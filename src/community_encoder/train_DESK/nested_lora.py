"""NestedLoRA as a differentiable TRAINING objective for DESK.

Companion to ``eigenbasis_diag``, which computes the same quantity as a read-only diagnostic. This
module is the loss: torch, differentiable, and intended to reach the optimizer.

Method: Ryu, Xu, Erol, Bu, Zheng & Wornell, "Operator SVD with Neural Networks via Nested Low-Rank
Approximation", ICML 2024 (https://openreview.net/forum?id=qESG5HaaoJ). Reference implementation at
``/Users/breallis/Dev/neural-svd``.

**Why DESK wants this at all.** DESK currently learns the eigenfunctions of the Ružička kernel
operator *indirectly*: ESK computes a Nyström eigenbasis offline, and DESK regresses onto those
coordinates (the stabilizing term) while separately matching dot products (the metric term). That
makes ESK a ceiling -- DESK can only be as good out of sample as the basis it is copying -- and it
pins Z to ESK's specific numbers including ESK's own approximation error. NestedLoRA solves the
eigenproblem directly, so it could replace both terms and remove the ceiling.

**The gradient subtlety, and why this implementation has no custom backward.**
The objective's operator term is ``-2 sum_l v_l <f_l, T f_l>``. Because ``T`` is self-adjoint and
``f`` appears on both sides, ``d/dtheta <f, Tf> = 2 <Tf, df/dtheta>``. The reference implementation
receives ``Tf`` as an already-computed tensor and therefore CANNOT see the second occurrence, so it
supplies the missing factor by hand: its backward returns ``-(4/B) v Tf`` where autograd on the same
forward would give ``-(2/B) v Tf``.

Verified against central finite differences on a free parameterisation (see
``tests/test_nested_lora.py``): the reference backward is correct to ~2e-9, plain autograd with a
detached ``Tf`` is wrong by ~6e-2, and building ``Tf`` INSIDE the autograd graph is correct to the
same ~2e-9. This module takes the third route. It needs no hand-written backward, which removes the
one part of the port that could be silently wrong, at the cost of backpropagating through one
``(B,B) @ (B,L)`` matmul against a constant Gram.

**Joint nesting only.** The sequential masks (``triu``) are asymmetric and their forward value is not
the objective their backward descends; the ordering there lives entirely in the hand-written
gradient. Since this module relies on autograd, only the symmetric joint masks are correct here.

**The split-half estimator is UNBOUNDED BELOW, and that is a production hazard.**
The metric term estimates ``E[.]^2``, so an unbiased estimate takes its two factors from
independent halves of the batch. But then putting all the magnitude in one half sends the other
half's ``Lambda`` to zero, so the entire quartic penalty vanishes while the operator term -- which
sees the whole batch -- can be driven arbitrarily negative. Measured on a 40-point toy problem: the
analytic optimum scores -0.0038, and a first-half-only configuration reaches -18.7 at scale 100 and
keeps falling. The full-batch form penalises the same configuration by +125620.

This is not a defect in the reference: the split is a variance device for a NETWORK's stochastic
gradient, where the network's dependence on covariates ties the two halves together, and it is
correct in that setting. It becomes exploitable exactly when the halves can move independently.

For DESK the batch is cells gathered from a spatial pool, so an index split would make the two
halves different REGIONS -- something an expressive network could plausibly learn to treat
differently. Two guards, both on by default: the halves are drawn by a fresh random permutation on
every call, so there is no stable target to specialise on; and ``return_parts`` reports the
imbalance between the two halves' Lambda norms, so an attempt to exploit it is visible in the
trajectory rather than only in a diverging loss.
"""
import torch


def ruzicka_gram(x, chunk_rows=256):
    """Pairwise Ružička similarity over a batch of raw communities. ``(B,B)``, no grad.

    Same estimand as ``desk_training._pair_kernel_loss``'s per-pair ratio and as
    ``eigenbasis_diag.ruzicka_gram``, over a full block. This is DATA -- it never requires grad, so
    it is computed once per batch and reused every step.

    Row-chunked because the pairwise ``|xi - xj|`` wants a ``(B,B,S)`` intermediate: 3 GiB at
    B=2048, S=96, a hundred times the size of the result.
    """
    x = torch.as_tensor(x)
    if x.dim() != 2:
        raise ValueError(f"x must be (B,S); got {tuple(x.shape)}")
    with torch.no_grad():
        b = x.shape[0]
        xs = x.sum(1)
        out = x.new_zeros((b, b))
        for i in range(0, b, chunk_rows):
            xb = x[i:i + chunk_rows]
            s = xb.sum(1)[:, None] + xs[None, :]
            d = (xb[:, None, :] - x[None, :, :]).abs().sum(2)
            num, den = 0.5 * (s - d), 0.5 * (s + d)
            out[i:i + chunk_rows] = torch.where(den > 1e-12, num / den.clamp_min(1e-30),
                                                torch.zeros_like(num))
    return out


def joint_nesting_masks(n_eig, step=1, device=None, dtype=None):
    """Joint nesting masks ``(v, M)`` with unit weight on every ``step``-th cut point.

    ``v[l] = sum_{k>=l} w_k`` and ``M[l,m] = min(v[l], v[m])``, which makes the objective exactly
    ``sum_k w_k L_LoRA(f_{1:k})`` -- the rank-k low-rank-approximation loss summed over prefixes.
    That is the nesting guarantee: at the optimum EVERY prefix is the best rank-k approximation, so
    the modes come out eigenvalue-ordered without any orthogonalisation step.

    ``step = n_eig`` puts all the weight on the full rank and recovers plain (un-nested) LoRA, whose
    optimum spans the right subspace but in no particular order -- useful as a control.
    """
    n, st = int(n_eig), int(step)
    # Checked before arange, which raises its own opaque "bound inconsistent with step sign"
    # for st > n rather than saying what is wrong.
    if not 1 <= st <= n:
        raise ValueError(f"step must be in [1, n_eig={n}] so it selects at least one cut point; "
                         f"got {step}")
    w = torch.zeros(n, dtype=dtype or torch.get_default_dtype(), device=device)
    w[torch.arange(st - 1, n, st, device=device)] = 1.0
    v = torch.flip(torch.cumsum(torch.flip(w, [0]), 0), [0])
    return v, torch.minimum(v[:, None], v[None, :])


def nested_lora_loss(f, gram, masks=None, step=1, return_parts=False, split_halves=True,
                     generator=None):
    """NestedLoRA (EVD) objective. Differentiable in ``f``. Lower is better.

    ``f``: ``(B, L)`` network outputs. ``gram``: ``(B, B)`` kernel over the same batch, treated as
    a constant. The empirical operator is ``T = gram / B``, so ``Tf = gram @ f / B`` -- built HERE,
    inside the graph, which is what makes plain autograd give the correct gradient (see the module
    docstring).

    ``split_halves=True`` (default, and the reference's behaviour) estimates the metric term's
    squared expectation from two independent halves, which is unbiased. The halves are chosen by a
    fresh random permutation each call: an index split would make them fixed subsets -- for DESK,
    different regions -- and the objective is unbounded below if the halves can move independently
    (see the module docstring). Re-permuting every call means there is no stable half to exploit.

    ``split_halves=False`` uses the full batch for both factors. That is biased upward by O(1/B) --
    it estimates ``E[X^2]`` where ``E[X]^2`` is wanted, inflating the term by a variance -- but it is
    bounded below, so it is the form to use when verifying that the objective's minimiser is the
    eigenbasis, and a safer choice if the imbalance diagnostic ever shows the split being gamed.
    """
    if f.dim() != 2:
        raise ValueError(f"f must be (B,L); got {tuple(f.shape)}")
    b, n_eig = f.shape
    if gram.shape != (b, b):
        raise ValueError(f"gram must be ({b},{b}); got {tuple(gram.shape)}")
    if b < 4:
        raise ValueError(f"need at least 4 rows to split into two usable halves; got {b}")
    v, m = (joint_nesting_masks(n_eig, step, f.device, f.dtype) if masks is None else masks)
    v, m = v.to(f.device, f.dtype), m.to(f.device, f.dtype)

    tf = gram.to(f.dtype) @ f / b                       # in-graph: f appears twice, as it must
    loss_operator = -2.0 * torch.einsum('l,bl,bl->b', v, f, tf).mean()
    if split_halves:
        half = b // 2
        perm = torch.randperm(b, generator=generator, device=f.device)
        f1, f2 = f[perm[:half]], f[perm[half:2 * half]]
        lam1, lam2 = f1.T @ f1 / half, f2.T @ f2 / half
    else:
        lam1 = lam2 = f.T @ f / b
    loss_metric = (m * lam1 * lam2).sum()
    loss = loss_operator + loss_metric
    if not return_parts:
        return loss
    n1, n2 = lam1.diagonal().sum().detach(), lam2.diagonal().sum().detach()
    return loss, {"operator": loss_operator.detach(), "metric": loss_metric.detach(),
                  # At the optimum the two satisfy metric = -operator/2, so this ratio's distance
                  # from 1 is a convergence check needing no reference basis.
                  "ratio": (loss_metric / (-loss_operator / 2)).detach(),
                  # Imbalance between the halves. 1.0 is balanced; a large value means one half
                  # carries the magnitude and the quartic penalty has been evaded -- the failure
                  # mode that makes the split form unbounded below. Watch it, do not assume it.
                  "half_imbalance": (torch.maximum(n1, n2)
                                     / torch.minimum(n1, n2).clamp_min(1e-30))}


def implied_eigenvalues(f):
    """``diag(f^T f / B)`` -- the eigenvalue each mode implies. Must be descending if ordered.

    NestedLoRA learns modes unnormalised so that ``||f_l||^2 = lambda_l``, which is what makes the
    prefix property testable: a non-monotone result means the modes are not in eigenvalue order, so
    a positional truncation ``f[:, :r]`` is not the top-r eigen-subspace.
    """
    return (f.detach() ** 2).mean(0)
