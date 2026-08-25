"""Is DESK's Z the ORDERED eigenbasis of the Ružička kernel, or merely a basis reproducing it?

Every metric DESK currently reports is a function of dot products alone, and dot products are
invariant to ``Z -> ZQ`` for orthogonal ``Q``. That invariance is not harmless. The population
model truncates **positionally** -- ``src/data/combine/model_inputs.py:470-471`` takes
``z[..., :M]`` -- and ``Z[:, :M]`` of a rotated ``ZQ`` is an arbitrary M-dimensional subspace
rather than the top-M eigen-subspace. ``src/model/age_priors.py:96-99`` accepts that input as long
as its metadata says ``top_eigenfeatures``, so a drifted basis passes every guard, produces a
worse rank-M Gaussian process, and shows up in no number anyone looks at.

The retained rank is also a moving target (24 today, plausibly 32 later), so nothing here is
allowed to key on a particular M. Every diagnostic below is either rank-agnostic or reported as a
curve over rank.

**These are diagnostics.** Nothing in this module is differentiable-by-intent or reachable from
the loss path; callers run it under ``no_grad``. Whether any of it should become a training
objective is a separate decision, to be made on the evidence it produces.

Ported from NeuralSVD / NestedLoRA -- Ryu, Xu, Erol, Bu, Zheng & Wornell, "Operator SVD with
Neural Networks via Nested Low-Rank Approximation", ICML 2024
(https://openreview.net/forum?id=qESG5HaaoJ), reference implementation at
``/Users/breallis/Dev/neural-svd``. Reimplemented rather than vendored: only the forward algebra
is needed, so the custom ``autograd.Function`` and its hand-written backward are not. Provenance
for each piece is given on the function that uses it.

**The three diagnostics are complementary, not redundant** -- measured on constructed cases:

===================  =============  ==================  ==============
transform            max_offdiag    spectrum ordered?   subspace @ r=8
===================  =============  ==================  ==============
none (ideal)              1e-16     yes                 0.000
swap components 3,11      1e-16     **no**              0.125
general rotation          0.375     no                  0.323
===================  =============  ==================  ==============

A component swap is a *permutation*, which is orthogonal -- so the orthogonality matrix is
completely blind to it, while the implied spectrum catches it immediately. A general rotation
correlates the components, which orthogonality catches. Subspace distance catches both. Drop any
one of the three and there is a real failure mode nothing reports.

DESK is in fact solving NeuralSVD's problem -- the top-L eigenfunctions of the Ružička kernel
operator -- but indirectly, by regressing onto a Nyström eigenbasis that ESK precomputes. That is
why NeuralSVD's diagnostics transfer without adaptation.
"""
import numpy as np


def ruzicka_gram(x, y=None, max_block_mib=256):
    """Ružička similarity between every row of ``x`` and every row of ``y``. ``(Bx, By)``.

    The same estimand as ``desk_training._pair_kernel_loss``'s per-pair ratio, over a full block
    instead of sampled pairs: ``sum(min) / sum(max)`` per pair, computed as
    ``(sum - |diff|) / (sum + |diff|)`` exactly as that function does, so the two cannot drift
    onto different definitions of the kernel DESK is fitted to.

    A full block is required because the nesting objective needs ``Tf = K f`` -- an operator
    applied to the feature map -- which random pairs cannot supply. The result is small (a 32 MB
    ``(B,B)`` matrix at B=2048) but the natural expression is not: the pairwise ``|xi - xj|`` needs
    a ``(Bx,By,S)`` intermediate, which at B=2048, S=96 is **3 GiB** -- a hundred times the output,
    in host memory, allocated inside the training loop. Computed in row blocks sized to cap that
    intermediate at ``max_block_mib`` instead, which changes nothing about the result and makes the
    cost scale with the answer rather than with the cube.

    Rows whose pair denominator vanishes (two all-zero communities) get similarity 0 rather than
    a divide-by-zero: undefined similarity is not high similarity.
    """
    x = np.asarray(x, dtype="float64")
    y = x if y is None else np.asarray(y, dtype="float64")
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"need (Bx,S) and (By,S) with matching S; got {x.shape} and {y.shape}")
    bx, by, sdim = x.shape[0], y.shape[0], x.shape[1]
    ysum = y.sum(1)
    out = np.zeros((bx, by), dtype="float64")
    # Rows per block, from the byte budget for the (rows, By, S) intermediate. At least 1, so a
    # very wide community set still makes progress instead of dividing to zero.
    per_row = max(by * sdim * 8, 1)
    rows = max(1, int(max_block_mib * 2 ** 20 // per_row))
    for i in range(0, bx, rows):
        xb = x[i:i + rows]
        # sum over species of min and max, via min = (a+b-|a-b|)/2 and max = (a+b+|a-b|)/2
        s = xb.sum(1)[:, None] + ysum[None, :]
        d = np.abs(xb[:, None, :] - y[None, :, :]).sum(2)
        num, den = 0.5 * (s - d), 0.5 * (s + d)
        ok = den > 1e-12
        blk = np.zeros_like(num)
        blk[ok] = num[ok] / den[ok]
        out[i:i + rows] = blk
    return out


def second_moment(f):
    """``f^T f / B`` -- the ``(L,L)`` second-moment matrix. NeuralSVD's ``compute_lambda``.

    Provenance: ``methods/nestedlora.py:10-11``,
    ``torch.einsum('bl...,bm...->lm', f, f) / f.shape[0]``.

    Everything cheap in this module comes from this one matrix, which is why it is worth
    computing even when the full nesting objective is skipped.
    """
    f = np.asarray(f, dtype="float64")
    if f.ndim != 2:
        raise ValueError(f"f must be (B,L); got {f.shape}")
    return f.T @ f / f.shape[0]


def implied_spectrum(f, kf=None):
    """Eigenvalue estimates implied by ``f``, and whether they are in descending order.

    Two independent estimators, both from NeuralSVD's ``methods/spectrum.py:86-90``:

    * ``norms`` -- ``diag(f^T f / B)``. NeuralSVD flags this one as specific to NestedLoRA, whose
      modes are learned unnormalised so that ``||f_l||^2 = lambda_l``.
    * ``rayleigh`` -- ``diag(f^T K f) / diag(f^T f)``, available only when ``kf`` is supplied.

    **Their disagreement is the diagnostic.** Both are exact at a genuine eigenbasis and diverge
    otherwise, so a gap between them is evidence independent of any comparison against ESK.

    ``descending`` is the property positional truncation actually depends on: if the implied
    spectrum is not monotone, then ``Z[:, :M]`` is not the top-M eigen-subspace for at least one
    M, and the downstream's rank-M GP is built on the wrong subspace. ``inversions`` counts the
    adjacent violations and ``worst_inversion_at`` locates the first, because *which* component
    drifted is what points at a cause.
    """
    cov = second_moment(f)
    norms = np.diag(cov).copy()
    out = {"norms": norms.tolist()}
    diffs = np.diff(norms)
    inv = np.flatnonzero(diffs > 0)
    out["descending"] = bool(inv.size == 0)
    out["inversions"] = int(inv.size)
    out["worst_inversion_at"] = (int(inv[0]) if inv.size else None)
    if kf is not None:
        kf = np.asarray(kf, dtype="float64")
        f = np.asarray(f, dtype="float64")
        quad = np.diag(f.T @ kf) / f.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            ray = np.where(norms > 1e-12, quad / np.maximum(norms, 1e-300), np.nan)
        out["rayleigh"] = ray.tolist()
        # Relative disagreement between the two estimators, on the components where both exist.
        fin = np.isfinite(ray) & (np.abs(norms) > 1e-12)
        out["estimator_disagreement"] = (
            float(np.median(np.abs(ray[fin] - norms[fin]) / np.abs(norms[fin])))
            if fin.any() else float("nan"))
    return out


def orthogonality(f):
    """How far ``f``'s components are from mutually orthogonal. ``1.0`` = perfectly orthogonal.

    The normalised second moment ``cov / sqrt(diag (x) diag^T)`` -- NeuralSVD's orthogonality
    matrix, ``methods/spectrum.py:145-146`` -- is the identity exactly when the components are
    orthogonal. Reported as summaries rather than an ``(L,L)`` matrix so it fits in a trajectory
    row: ``max_offdiag`` is the worst single leak and ``mean_abs_offdiag`` the overall level.

    This is the cheapest evidence of basis drift available: one ``f^T f``, no kernel, no ESK
    reference. A basis can reproduce every dot product while having correlated components, and
    that is precisely the state in which a positional prefix stops being an eigen-subspace.
    """
    cov = second_moment(f)
    d = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    norm = cov / np.outer(d, d)
    L = norm.shape[0]
    off = norm[~np.eye(L, dtype=bool)]
    return {"max_offdiag": float(np.max(np.abs(off))) if off.size else 0.0,
            "mean_abs_offdiag": float(np.mean(np.abs(off))) if off.size else 0.0,
            "frobenius_from_identity": float(np.linalg.norm(norm - np.eye(L)))}


def subspace_distance(a1, a2):
    """Normalised distance between the column spans of ``a1`` and ``a2``. 0 = same subspace.

    ``1 - tr(P1 P2)/k`` where ``Pi`` projects onto span(``ai``). Verbatim in form from NeuralSVD's
    ``examples/linalg.py:5-8``.

    This is the right comparison for the truncation question, and elementwise error is not: a
    rotation *within* the retained block is harmless to the downstream (its prior is isotropic,
    ``src/model/age_priors.py:224-240``, so the induced GP is unchanged), and subspace distance
    correctly scores that 0 while a coordinate-wise error would flag it. What is *not* harmless
    is a rotation that mixes retained with discarded components, and that is exactly what this
    detects.

    Uses ``lstsq``-based projectors rather than the paper's explicit ``inv(a^T a)``: the columns
    of a truncated DESK Z can be near-collinear, and an explicit inverse turns that into a silent
    garbage value instead of a rank-deficiency.
    """
    a1 = np.asarray(a1, dtype="float64")
    a2 = np.asarray(a2, dtype="float64")
    if a1.ndim != 2 or a2.ndim != 2 or a1.shape[1] != a2.shape[1]:
        raise ValueError(f"need (n,k) and (n,k) with matching k; got {a1.shape}, {a2.shape}")
    k = a1.shape[1]
    q1, _ = np.linalg.qr(a1)
    q2, _ = np.linalg.qr(a2)
    # tr(P1 P2) = ||q1^T q2||_F^2 for orthonormal bases q1, q2
    return float(1.0 - np.linalg.norm(q1.T @ q2) ** 2 / k)


def subspace_curve(z_desk, z_esk, ranks):
    """``{rank: subspace_distance(z_desk[:, :r], z_esk[:, :r])}`` -- the rank question, answered.

    A curve rather than a single rank, because the downstream's retained M is not fixed. Near-zero
    everywhere means truncation is safe at any rank; a rise at some ``r`` says the top-``r``
    subspace has drifted and a downstream truncating there would fit the wrong rank-``r`` kernel.

    Ranks above the available width are skipped rather than clipped: silently returning the value
    for a different rank than asked for is how a table comes to mean something other than its
    column headings.
    """
    out = {}
    L = min(z_desk.shape[1], z_esk.shape[1])
    for r in ranks:
        r = int(r)
        if 1 <= r <= L:
            out[r] = subspace_distance(z_desk[:, :r], z_esk[:, :r])
    return out


def joint_nesting_masks(n_eig):
    """NeuralSVD's JOINT nesting masks, unit weight on every cut point.

    Provenance: ``methods/nestedlora.py:40-46``. ``vector_mask[l] = sum_{k>=l} w_k`` and
    ``matrix_mask[l,m] = min(vector_mask[l], vector_mask[m])``.

    Deliberately not the *sequential* masks (``methods/nestedlora.py:49-54``,
    ``matrix_mask = triu(ones)``). That mask is asymmetric and its forward value is not a
    meaningful objective -- in the reference implementation the ordering it induces lives entirely
    in a hand-written ``backward`` (``methods/nestedlora.py:108-111``, an ``'lm,lm,bl->bm'``
    contraction that couples mode m only to modes l <= m). A diagnostic reads the forward scalar,
    so joint nesting is the only correct choice here.

    With unit weights the objective becomes ``sum_k L_LoRA(f_{1:k})`` over every prefix k, which
    is what makes one number cover all ranks.
    """
    w = np.ones(int(n_eig), dtype="float64")
    v = np.cumsum(w[::-1])[::-1]
    return v, np.minimum.outer(v, v)


def nesting_objective(f, kf, masks=None):
    """NeuralSVD's NestedLoRA objective, forward value only. Lower is better.

    ``-2 sum_l v_l E_b[f[b,l] (Kf)[b,l]] + sum_{l,m} M[l,m] Lam1[l,m] Lam2[l,m]``

    Provenance: ``methods/nestedlora.py:70-94`` (``NestedLoRALossFunctionEVD.forward``).

    ``Lam1``/``Lam2`` come from **independent halves** of the batch. The reference implementation
    requires this (``methods/nestedlora.py:84``, "warning: f1 and f2 must be independent") because
    the second term estimates a squared expectation: reusing one half for both factors estimates
    ``E[X^2]`` where ``E[X]^2`` is wanted, biasing the term upward by a variance.

    The value scales with the kernel and the batch, so it is meaningless in isolation. It is
    interpretable only against the same quantity on a reference basis -- ESK's own projection of
    the same points, which is an ordered eigenbasis by construction and therefore the achievable
    floor at this Nyström rank.

    ``operator_metric_ratio``: at the optimum the two terms satisfy
    ``loss_metric = -loss_operator / 2``, so this ratio's distance from 1 is a
    reference-free convergence check (cf. the term split returned at
    ``methods/nestedlora.py:306``).
    """
    f = np.asarray(f, dtype="float64")
    kf = np.asarray(kf, dtype="float64")
    if f.shape != kf.shape:
        raise ValueError(f"f and Kf must match; got {f.shape} and {kf.shape}")
    B, L = f.shape
    if B < 2:
        raise ValueError("need at least 2 rows to split into independent halves")
    v, m = joint_nesting_masks(L) if masks is None else masks
    half = B // 2
    lam1, lam2 = second_moment(f[:half]), second_moment(f[half:2 * half])
    loss_operator = -2.0 * float(np.sum(v * np.mean(f * kf, axis=0)))
    loss_metric = float(np.sum(m * lam1 * lam2))
    ratio = (loss_metric / (-loss_operator / 2.0)
             if abs(loss_operator) > 1e-300 else float("nan"))
    return {"nesting_loss": loss_operator + loss_metric,
            "loss_operator": loss_operator,
            "loss_metric": loss_metric,
            "operator_metric_ratio": ratio}


def eigenbasis_report(z, x, z_ref=None, ranks=(8, 16, 24, 32, 48, 64), gram=None):
    """Every diagnostic in this module for one batch. ``gram`` is reused if already computed.

    ``z`` is the encoder's output at the batch's points ``(B,L)``, ``x`` the matching raw
    communities ``(B,S)``, ``z_ref`` an optional reference basis at the SAME points (ESK's own
    projection) against which the subspace curve and the nesting gap are measured.

    Returns a flat dict of scalars plus the two curves, so a caller can drop it straight into a
    trajectory row. Pure numpy on already-detached arrays: nothing here can touch a weight.
    """
    k = ruzicka_gram(x) if gram is None else np.asarray(gram, dtype="float64")
    kf = k @ np.asarray(z, dtype="float64") / k.shape[1]
    out = {"orthogonality": orthogonality(z),
           "spectrum": implied_spectrum(z, kf=kf),
           "nesting": nesting_objective(z, kf)}
    if z_ref is not None:
        ref_kf = k @ np.asarray(z_ref, dtype="float64") / k.shape[1]
        out["subspace_vs_ref"] = subspace_curve(np.asarray(z), np.asarray(z_ref), ranks)
        out["nesting_ref"] = nesting_objective(z_ref, ref_kf)
        # The gap is the honest number: the reference is an ordered eigenbasis by construction,
        # so its value is the floor achievable at this Nyström rank on this batch.
        out["nesting_gap"] = out["nesting"]["nesting_loss"] - out["nesting_ref"]["nesting_loss"]
    return out
