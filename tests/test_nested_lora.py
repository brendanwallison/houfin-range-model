"""Does minimising the NestedLoRA loss actually recover a known eigenbasis, in order?

The forward value was already verified against the paper's own code to ~1e-14
(``tests/test_eigenbasis_diag.py``). That is necessary and not sufficient for a TRAINING objective:
a loss can compute the right number and still descend to the wrong place if its gradient is wrong,
and this objective's gradient has a subtlety that plain autograd gets wrong unless ``Tf`` is built
inside the graph.

So these tests optimise it on toy problems whose answer ``numpy.linalg.eigh`` supplies exactly, and
check the recovered basis against that -- subspace, ORDER, and eigenvalues. Toy data on purpose:
with a 40x40 kernel the truth is available to machine precision, which no test on real communities
can offer.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.nested_lora import (  # noqa: E402
    implied_eigenvalues, joint_nesting_masks, nested_lora_loss, ruzicka_gram)

NEURAL_SVD = "/Users/breallis/Dev/neural-svd"


def _sym_kernel(b, seed=0, decay=0.7):
    """A symmetric PSD kernel with a clearly separated, known spectrum."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(b, b)))
    lam = decay ** np.arange(b)
    return torch.tensor(q @ np.diag(lam) @ q.T, dtype=torch.float64), q, lam


def _fit(gram, n_eig, steps=4000, lr=0.05, step=1, seed=0, split_halves=False):
    """Minimise the loss over a free (B,L) parameterisation. Returns the fitted f.

    ``split_halves=False`` by default here, and that is load-bearing rather than incidental. The
    split-half estimator is UNBOUNDED BELOW over free per-row parameters -- put the magnitude in
    one half and the other half's Lambda goes to zero, killing the quartic penalty while the
    operator term keeps falling. Measured: it reaches -9556 against an analytic optimum of -0.0038.
    The full-batch form is biased by O(1/B) but bounded, so it is the one whose minimiser can be
    checked against numpy.linalg.eigh. See test_the_split_half_form_is_unbounded_below.
    """
    b = gram.shape[0]
    torch.manual_seed(seed)
    f = torch.randn(b, n_eig, dtype=torch.float64, requires_grad=True) * 0.1
    f = f.detach().requires_grad_(True)
    opt = torch.optim.Adam([f], lr=lr)
    masks = joint_nesting_masks(n_eig, step, f.device, f.dtype)
    for _ in range(steps):
        opt.zero_grad()
        nested_lora_loss(f, gram, masks=masks, split_halves=split_halves).backward()
        opt.step()
    return f.detach()


def _subspace_dist(a, b):
    qa, _ = np.linalg.qr(np.asarray(a))
    qb, _ = np.linalg.qr(np.asarray(b))
    return 1 - np.linalg.norm(qa.T @ qb) ** 2 / a.shape[1]


# --------------------------------------------------------------------------- gradient correctness

def test_the_gradient_matches_central_finite_differences():
    """THE test for a training objective: a right value with a wrong gradient descends wrong.

    Finite differences on the full objective -- with ``Tf`` recomputed from the parameters, because
    that is what the true derivative includes. This is the check that distinguishes building ``Tf``
    in the graph (correct) from passing it detached (wrong by a factor of 2 on the operator term,
    since ``<f, Tf>`` contains ``f`` twice and ``T`` is self-adjoint).
    """
    torch.manual_seed(0)
    b, n_eig = 24, 3
    gram, _, _ = _sym_kernel(b, seed=1)
    f0 = torch.randn(b, n_eig, dtype=torch.float64)
    masks = joint_nesting_masks(n_eig, 1, f0.device, f0.dtype)

    f = f0.clone().requires_grad_(True)
    nested_lora_loss(f, gram, masks=masks, split_halves=False).backward()
    g_auto = f.grad.clone()

    eps = 1e-6
    g_fd = torch.zeros_like(f0)
    for i in range(b):
        for j in range(n_eig):
            p, m = f0.clone(), f0.clone()
            p[i, j] += eps
            m[i, j] -= eps
            g_fd[i, j] = (nested_lora_loss(p, gram, masks=masks, split_halves=False)
                          - nested_lora_loss(m, gram, masks=masks,
                                             split_halves=False)) / (2 * eps)
    rel = (g_auto - g_fd).abs().max() / g_fd.abs().max()
    assert rel < 1e-7, f"gradient disagrees with finite differences by {rel:.2e}"


def test_a_detached_operator_term_gives_the_wrong_gradient():
    """Pins WHY Tf is built in-graph, so a future 'optimisation' cannot silently undo it.

    Detaching Tf halves the operator term's gradient. That still trains -- it just descends a
    different objective -- which is the kind of error that produces a plausible model and no error
    message.
    """
    torch.manual_seed(0)
    b, n_eig = 24, 3
    gram, _, _ = _sym_kernel(b, seed=1)
    f0 = torch.randn(b, n_eig, dtype=torch.float64)
    v, m = joint_nesting_masks(n_eig, 1, f0.device, f0.dtype)

    f = f0.clone().requires_grad_(True)
    nested_lora_loss(f, gram, masks=(v, m), split_halves=False).backward()
    g_ok = f.grad.clone()

    f = f0.clone().requires_grad_(True)
    tf = (gram @ f0 / b).detach()
    half = b // 2
    lamf = f.T @ f / b
    (-2 * torch.einsum('l,bl,bl->b', v, f, tf).mean() + (m * lamf * lamf).sum()).backward()
    g_bad = f.grad.clone()

    assert not torch.allclose(g_ok, g_bad), "detaching Tf must change the gradient"


# ------------------------------------------------------------------- does it recover the answer?

def test_minimising_it_recovers_the_top_eigen_subspace():
    """The fitted modes must span the true top-L eigen-subspace of the operator."""
    b, n_eig = 40, 4
    gram, q, lam = _sym_kernel(b, seed=2)
    f = _fit(gram, n_eig)
    truth = q[:, :n_eig]                       # eigh columns are already descending here
    d = _subspace_dist(f.numpy(), truth)
    assert d < 1e-3, f"subspace distance {d:.2e} -- did not recover the top-{n_eig} subspace"


def test_the_recovered_modes_are_in_eigenvalue_ORDER():
    """The nesting property, which is the whole reason for this objective.

    Un-nested LoRA recovers the right SUBSPACE in arbitrary order; nesting is what makes every
    prefix optimal, so that a positional truncation f[:, :r] is the top-r eigen-subspace. That is
    exactly the property DESK's downstream depends on and currently does not have.
    """
    b, n_eig = 40, 4
    gram, q, lam = _sym_kernel(b, seed=3)
    f = _fit(gram, n_eig, step=1)
    # every PREFIX must match the corresponding true prefix, not just the full set
    for r in range(1, n_eig + 1):
        d = _subspace_dist(f[:, :r].numpy(), q[:, :r])
        assert d < 5e-3, f"prefix r={r} subspace distance {d:.2e} -- modes are not ordered"
    ev = implied_eigenvalues(f).numpy()
    assert np.all(np.diff(ev) < 1e-8), f"implied spectrum not descending: {ev}"


def test_unnested_lora_gets_the_subspace_but_NOT_the_order():
    """The control that shows the ordering comes from nesting and not from the optimiser.

    With all the weight on the full rank (step = n_eig) the objective is plain LoRA: its optimum is
    any basis spanning the top-L subspace. If this ALSO came out ordered, the ordering in the test
    above would prove nothing about the nesting masks.
    """
    b, n_eig = 40, 4
    gram, q, lam = _sym_kernel(b, seed=4)
    f = _fit(gram, n_eig, step=n_eig)
    assert _subspace_dist(f.numpy(), q[:, :n_eig]) < 1e-3, "un-nested must still get the subspace"
    # but an intermediate prefix should NOT match, and the spectrum should not be sorted
    worst = max(_subspace_dist(f[:, :r].numpy(), q[:, :r]) for r in (1, 2, 3))
    ev = implied_eigenvalues(f).numpy()
    assert worst > 1e-2 or np.any(np.diff(ev) > 1e-6), (
        "un-nested LoRA came out ordered anyway; this control cannot distinguish the masks")


def test_the_implied_eigenvalues_match_the_true_spectrum():
    """``||f_l||^2 = lambda_l`` at the optimum, which is how eigenvalues are read off."""
    b, n_eig = 40, 4
    gram, q, lam = _sym_kernel(b, seed=5)
    f = _fit(gram, n_eig, steps=6000)
    got = implied_eigenvalues(f).numpy()
    # the empirical operator is gram/B, so its eigenvalues are lam/B
    want = lam[:n_eig] / b
    rel = np.abs(got - want) / want
    assert rel.max() < 0.05, f"implied {got} vs true {want} (rel {rel})"


def test_a_rank_deficient_kernel_does_not_blow_up():
    """Real Ružička blocks are near-singular; the loss must stay finite on one."""
    b, n_eig = 32, 6
    rng = np.random.default_rng(7)
    low = rng.normal(size=(b, 3))
    gram = torch.tensor(low @ low.T, dtype=torch.float64)     # rank 3, asking for 6 modes
    f = _fit(gram, n_eig, steps=2000)
    assert torch.isfinite(f).all()
    ev = implied_eigenvalues(f).numpy()
    # the modes beyond the true rank must collapse toward zero, not toward noise
    assert ev[:3].min() > 10 * max(ev[3:].max(), 1e-12), f"spurious modes did not collapse: {ev}"


# --------------------------------------------------------------- the pieces around the objective

def test_the_torch_gram_matches_the_numpy_diagnostic_one():
    """The loss and the diagnostic must be fitted to the SAME kernel.

    Two modules computing 'the Ružička similarity' independently is how this project previously
    ended up with two disagreeing definitions of one quantity.
    """
    from src.community_encoder.train_DESK.eigenbasis_diag import ruzicka_gram as np_gram

    rng = np.random.default_rng(11)
    x = rng.random((70, 9))
    a = ruzicka_gram(torch.tensor(x), chunk_rows=16).numpy()
    b = np_gram(x)
    assert np.allclose(a, b, atol=1e-12), np.abs(a - b).max()
    # chunking must not change it either
    assert np.allclose(ruzicka_gram(torch.tensor(x), chunk_rows=3).numpy(), b, atol=1e-12)
    assert not ruzicka_gram(torch.tensor(x)).requires_grad, "the kernel is data, never a parameter"


def test_the_masks_are_symmetric_and_step_recovers_unnested():
    v, m = joint_nesting_masks(5, step=1)
    assert torch.allclose(v, torch.tensor([5., 4., 3., 2., 1.]))
    assert torch.allclose(m, m.T)
    v2, m2 = joint_nesting_masks(5, step=5)
    assert torch.allclose(v2, torch.ones(5)), v2
    assert torch.allclose(m2, torch.ones(5, 5)), m2
    with pytest.raises(ValueError, match="at least one cut point"):
        joint_nesting_masks(5, step=9)
    with pytest.raises(ValueError, match="at least one cut point"):
        joint_nesting_masks(5, step=0)


def test_reusing_one_half_biases_the_metric_term_upward():
    """It estimates ``E[.]^2``; one half used twice estimates ``E[X^2]``, which is larger.

    In EXPECTATION, not per draw -- ``E[X^2] = E[X]^2 + Var[X]``, so the inflation is a variance and
    a single sample can fall either way. An earlier version of this test asserted it on one draw and
    duly failed (14.66 vs 14.98). Averaged over draws the sign is unambiguous.
    """
    n_eig, b, half = 5, 64, 32
    _v, m = joint_nesting_masks(n_eig, 1, dtype=torch.float64)
    splits, sames = [], []
    for s in range(200):
        g = torch.Generator().manual_seed(s)
        f = torch.randn(b, n_eig, dtype=torch.float64, generator=g)
        lam1 = f[:half].T @ f[:half] / half
        lam2 = f[half:].T @ f[half:] / half
        splits.append(float((m * lam1 * lam2).sum()))
        sames.append(float((m * lam1 * lam1).sum()))
    assert np.mean(sames) > np.mean(splits), (np.mean(sames), np.mean(splits))
    # and the inflation is the variance term, so it should be a clear effect, not a coin flip
    assert np.mean(sames) > np.mean(splits) * 1.02, (np.mean(sames), np.mean(splits))


def test_it_refuses_shapes_it_cannot_use():
    gram, _, _ = _sym_kernel(8, seed=0)
    with pytest.raises(ValueError, match=r"must be \(B,L\)"):
        nested_lora_loss(torch.randn(8, dtype=torch.float64), gram)
    with pytest.raises(ValueError, match="gram must be"):
        nested_lora_loss(torch.randn(8, 2, dtype=torch.float64),
                         torch.randn(5, 5, dtype=torch.float64))
    with pytest.raises(ValueError, match="at least 4 rows"):
        nested_lora_loss(torch.randn(3, 2, dtype=torch.float64),
                         torch.randn(3, 3, dtype=torch.float64))


@pytest.mark.skipif(not os.path.isdir(NEURAL_SVD), reason="reference not present")
def test_the_forward_still_matches_the_reference_after_the_torch_port():
    """The torch loss must agree with the paper's code, not just with our numpy diagnostic."""
    sys.path.insert(0, NEURAL_SVD)
    from methods.nestedlora import (NestedLoRALossFunctionEVD,  # noqa: E402
                                    get_joint_nesting_masks)
    rng = np.random.default_rng(0)
    for b, n_eig in ((64, 6), (128, 8)):
        f = torch.tensor(rng.normal(size=(b, n_eig)))
        gram, _, _ = _sym_kernel(b, seed=3)
        tf = gram @ f / b
        v_ref, m_ref = get_joint_nesting_masks(np.ones(n_eig))
        v_ref, m_ref = v_ref.double(), m_ref.double()
        # Like with like: our full-batch form against the reference given the full batch for both
        # Lambda factors. Comparing it against the reference's SPLIT call would compare two
        # different estimators and fail for that reason, which an earlier version of this test did.
        ref_full = float(NestedLoRALossFunctionEVD.apply(f, tf, f, f, v_ref, m_ref))
        mine_full = float(nested_lora_loss(f, gram, split_halves=False))
        assert abs(ref_full - mine_full) < 1e-10 * max(abs(ref_full), 1.0), (b, n_eig)
        # and the split estimator differs from it, which is the point of having both
        ref_split = float(NestedLoRALossFunctionEVD.apply(
            f, tf, f[:b // 2], f[b // 2:], v_ref, m_ref))
        assert abs(ref_split - ref_full) > 1e-6, "the two estimators should not coincide"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_the_split_half_form_is_unbounded_below():
    """The degeneracy that makes the split form unusable as a standalone objective.

    An unbiased estimate of ``E[.]^2`` needs two independent halves -- but then all the magnitude
    can go into one half, sending the other's ``Lambda`` to zero, so the quartic penalty vanishes
    entirely while the operator term still sees the whole batch and keeps falling. Measured here
    against the analytic optimum, and against the full-batch form which penalises the same
    configuration heavily.

    Not a defect in the reference: the split is a variance device for a network's stochastic
    gradient, where the network's dependence on covariates ties the halves together. It is
    exploitable exactly when the halves can move independently -- which is why this module
    re-permutes them every call and reports their imbalance.
    """
    b, n_eig = 40, 4
    gram, q, lam = _sym_kernel(b, seed=2)
    masks = joint_nesting_masks(n_eig, 1, dtype=torch.float64)
    f_true = (torch.tensor(q[:, :n_eig] * np.sqrt(b))
              * torch.tensor(np.sqrt(lam[:n_eig] / b)))
    at_opt = float(nested_lora_loss(f_true, gram, masks, split_halves=False))

    half = b // 2
    worst_split, worst_full = at_opt, at_opt
    for c in (1.0, 10.0, 100.0):
        f = torch.zeros(b, n_eig, dtype=torch.float64)
        f[:half] = torch.tensor(q[:half, :n_eig]) * c
        # index halves on purpose: this is the configuration a fixed split would permit
        v, m = masks
        tf = gram @ f / b
        lam1 = f[:half].T @ f[:half] / half
        lam2 = f[half:].T @ f[half:] / half
        split = float(-2 * torch.einsum('l,bl,bl->b', v, f, tf).mean() + (m * lam1 * lam2).sum())
        full = float(nested_lora_loss(f, gram, masks, split_halves=False))
        worst_split, worst_full = min(worst_split, split), max(worst_full, full)
    assert worst_split < at_opt - 1.0, (
        f"a one-sided configuration must beat the analytic optimum under the SPLIT form "
        f"({worst_split:.3f} vs {at_opt:.4f}) -- that is the unboundedness")
    assert worst_full > 1000, (
        f"the full-batch form must penalise it heavily, got {worst_full:.1f}")


def test_the_halves_are_re_permuted_every_call():
    """A fixed index split gives the network a stable subset to specialise on.

    For DESK the batch is cells from a spatial pool, so an index split makes the two halves
    different regions. Re-permuting each call means there is no consistent half to push the
    magnitude into.
    """
    torch.manual_seed(0)
    b, n_eig = 64, 4
    gram, _, _ = _sym_kernel(b, seed=0)
    f = torch.randn(b, n_eig, dtype=torch.float64)
    vals = {float(nested_lora_loss(f, gram, split_halves=True)) for _ in range(8)}
    assert len(vals) > 1, "the split must be re-randomised, so repeated calls differ"
    # the full-batch form is deterministic, which is what verification needs
    fixed = {float(nested_lora_loss(f, gram, split_halves=False)) for _ in range(4)}
    assert len(fixed) == 1, fixed


def test_the_half_imbalance_diagnostic_detects_the_exploit():
    """The guard must actually fire on the configuration that breaks the objective."""
    b, n_eig = 40, 4
    gram, q, _ = _sym_kernel(b, seed=2)
    balanced = torch.tensor(q[:, :n_eig], dtype=torch.float64)
    _l, p = nested_lora_loss(balanced, gram, return_parts=True, split_halves=True)
    assert p["half_imbalance"] < 5.0, p["half_imbalance"]

    lopsided = torch.zeros(b, n_eig, dtype=torch.float64)
    lopsided[:b // 2] = torch.tensor(q[:b // 2, :n_eig]) * 100
    # forced index halves, to show what the diagnostic reports when the split IS gamed
    v, m = joint_nesting_masks(n_eig, 1, dtype=torch.float64)
    half = b // 2
    n1 = (lopsided[:half].T @ lopsided[:half] / half).diagonal().sum()
    n2 = (lopsided[half:].T @ lopsided[half:] / half).diagonal().sum()
    assert (n1 / n2.clamp_min(1e-30)) > 1e6, "the exploit must show as a huge imbalance"
