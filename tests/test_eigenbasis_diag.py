"""Tests for the eigenbasis diagnostics: they must detect what the kernel metric cannot.

Every existing DESK metric is a function of dot products, and dot products are invariant to
``Z -> ZQ``. That invariance is not harmless, because the population model truncates POSITIONALLY
(``src/data/combine/model_inputs.py:470-471``) and a rotated prefix is not an eigen-subspace. So
the load-bearing assertion in this file is the pair: a transformed Z whose 64-dim Gram is
*bit-identical* must still be flagged. A diagnostic that merely runs, or that only fires on inputs
the old metric would also catch, is worthless here.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.eigenbasis_diag import (  # noqa: E402
    eigenbasis_report, implied_spectrum, joint_nesting_masks, nesting_objective, orthogonality,
    ruzicka_gram, second_moment, subspace_curve, subspace_distance)


def _ordered_basis(n=256, L=16, seed=0):
    """An exactly orthonormal, eigenvalue-descending Z: the ideal the real thing approximates."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(n, L)))
    lam = np.linspace(1.0, 0.05, L)                 # strictly descending
    return q * np.sqrt(lam) * np.sqrt(n)            # so diag(Z^T Z / n) == lam


def _kernel_eigenbasis(k, L):
    """Top-L eigenbasis of the EMPIRICAL OPERATOR ``T = K/B``, scaled as NestedLoRA's modes are.

    The convention matters and is easy to get wrong by a factor of B. The module estimates
    ``Tf`` as ``K @ f / B`` -- the Monte Carlo empirical operator -- so its eigenvalues are the
    Gram matrix's divided by B. NestedLoRA's modes are unnormalised with ``||f_l||^2 = lambda_l``,
    and here ``||f_l||^2`` means ``E_b[f_l^2]``, so the correct scaling is ``V * sqrt(vals)``
    with no extra ``sqrt(B)``: that gives ``E_b[f_l^2] = vals_l / B``, which is exactly the
    operator's eigenvalue. Scaling to the MATRIX eigenvalue instead makes the Rayleigh estimator
    disagree with the norm estimator by a factor of B at the true eigenbasis, which is not a
    finding about the basis.
    """
    vals, vecs = np.linalg.eigh(k)
    order = np.argsort(vals)[::-1][:L]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0.0))


def test_an_ordered_orthonormal_basis_scores_clean():
    """The ideal case must score clean, or every other assertion here is unanchored."""
    z = _ordered_basis()
    o = orthogonality(z)
    assert o["max_offdiag"] < 1e-8, o
    s = implied_spectrum(z)
    assert s["descending"] and s["inversions"] == 0, s
    assert np.allclose(s["norms"], np.linspace(1.0, 0.05, 16), atol=1e-8)
    assert subspace_distance(z, z) == pytest.approx(0.0, abs=1e-10)


def test_a_component_swap_is_caught_while_every_dot_product_is_identical():
    """THE test. Swapping two components leaves the 64-dim Gram bit-identical.

    A permutation is orthogonal, so ``dot(z_i, z_j)`` is unchanged for every pair -- the kernel
    metric DESK selects on cannot see this at all. But ``Z[:, :r]`` is now the wrong subspace for
    every r between the swapped indices, which is exactly the state that makes the downstream's
    positional truncation fit a worse rank-r GP while ``age_priors`` still accepts the input.
    """
    z = _ordered_basis()
    sw = z.copy()
    sw[:, [3, 11]] = sw[:, [11, 3]]

    # The quantity the current metric is a function of, unchanged. Not bit-identical: a
    # permutation reorders the summation in the matmul, so the last few bits move. That is float
    # arithmetic, not a difference in the Gram -- the mathematical claim is exact equality, and
    # the tolerance here is ~1e-14, far below anything the kernel metric could resolve.
    assert np.allclose(z @ z.T, sw @ sw.T, rtol=0, atol=1e-12)

    # but the spectrum is no longer descending, and it names where
    s = implied_spectrum(sw)
    assert not s["descending"], s
    assert s["worst_inversion_at"] == 3, s

    # and the subspace is wrong exactly between the swapped indices, not outside them
    curve = subspace_curve(sw, z, ranks=(2, 3, 4, 8, 11, 12, 16))
    assert curve[2] == pytest.approx(0.0, abs=1e-10), curve
    assert curve[3] == pytest.approx(0.0, abs=1e-10), curve      # swap is at index 3 -> rank 4
    assert curve[4] > 0.05, curve
    assert curve[8] > 0.05, curve
    assert curve[11] > 0.05, curve
    assert curve[12] == pytest.approx(0.0, abs=1e-10), curve     # both back inside the prefix
    assert curve[16] == pytest.approx(0.0, abs=1e-10), curve


def test_a_full_rotation_is_caught_while_the_gram_is_preserved():
    """A general rotation preserves every dot product and destroys the eigen-ordering."""
    rng = np.random.default_rng(3)
    z = _ordered_basis()
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    zq = z @ q

    assert np.allclose(z @ z.T, zq @ zq.T, atol=1e-10)           # kernel: blind
    assert orthogonality(zq)["max_offdiag"] > 0.05               # components now correlated
    curve = subspace_curve(zq, z, ranks=(4, 8, 12))
    assert all(v > 0.1 for v in curve.values()), curve


def test_a_rotation_inside_the_retained_block_is_correctly_ignored():
    """The invariance must be the RIGHT one, or the diagnostic cries wolf.

    The downstream's prior over features is isotropic (``src/model/age_priors.py:224-240``), so a
    rotation confined within the retained prefix leaves the induced GP exactly unchanged and must
    NOT be flagged. A coordinate-wise comparison would flag it; subspace distance does not.
    """
    rng = np.random.default_rng(5)
    z = _ordered_basis()
    r = 8
    q = np.eye(16)
    q[:r, :r] = np.linalg.qr(rng.normal(size=(r, r)))[0]
    zb = z @ q

    assert subspace_distance(zb[:, :r], z[:, :r]) == pytest.approx(0.0, abs=1e-10)
    # and it is still detected at ranks that cut THROUGH the rotated block
    assert subspace_distance(zb[:, :4], z[:, :4]) > 0.05


def test_the_ruzicka_gram_matches_the_trainer_pairwise_definition():
    """The block kernel must be the same estimand as the loss's per-pair ratio.

    Two modules computing "the Ružička similarity" in different orders is how this project
    previously ended up with two disagreeing definitions of one quantity, so the block form is
    checked against the trainer's own pairwise code path rather than against a fresh derivation.
    """
    import torch

    from src.community_encoder.train_DESK.desk_training import _pair_kernel_loss

    rng = np.random.default_rng(7)
    x = rng.random((32, 12)).astype("float32")
    g = ruzicka_gram(x)
    assert g.shape == (32, 32)
    assert np.allclose(np.diag(g), 1.0, atol=1e-9)              # self-similarity is 1
    assert np.allclose(g, g.T, atol=1e-12)

    # _pair_kernel_loss returns MSE(dot(zi,zj), ruzicka(xi,xj)); feeding z whose dot products are
    # exactly the block gram's entries must therefore give ~0 loss.
    i, j = np.triu_indices(32, k=1)
    xi, xj = torch.tensor(x[i]), torch.tensor(x[j])
    sim = torch.tensor(g[i, j], dtype=torch.float32)
    # a rank-1 z per pair reproducing that similarity exactly
    zi = torch.ones(len(i), 1)
    zj = sim.unsqueeze(1)
    loss = float(_pair_kernel_loss(zi, zj, xi, xj))
    assert loss < 1e-10, loss


def test_an_all_zero_pair_gets_zero_similarity_not_a_divide_by_zero():
    """Undefined similarity is not high similarity."""
    x = np.zeros((3, 5))
    x[0, 0] = 1.0
    g = ruzicka_gram(x)
    assert np.isfinite(g).all()
    assert g[1, 2] == 0.0 and g[0, 1] == 0.0
    assert g[0, 0] == pytest.approx(1.0)


def test_joint_nesting_masks_are_symmetric_and_sum_prefix_objectives():
    """Joint, not sequential: the forward value must be a real objective.

    The sequential mask (``triu``) is asymmetric and its forward scalar is not meaningful -- in
    the reference implementation the ordering it induces lives entirely in a hand-written
    backward. A diagnostic reads the forward, so the mask must be the symmetric ``min`` form,
    whose value is exactly the sum of the rank-k objectives over every prefix k.
    """
    v, m = joint_nesting_masks(5)
    assert np.array_equal(v, [5, 4, 3, 2, 1])                  # cumsum of unit weights, reversed
    assert np.allclose(m, m.T), m
    assert m[0, 4] == 1 and m[0, 0] == 5
    assert not np.allclose(m, np.triu(m)), "must not be the sequential/triu mask"


def test_the_nesting_objective_prefers_the_ordered_basis_over_a_rotation():
    """The reference basis must score at least as well as any rotation of it.

    This is the property that makes the scalar usable: its absolute value is uninterpretable, so
    it is only ever read as a gap against a reference that is an ordered eigenbasis by
    construction. If a rotation could beat that reference, the gap would mean nothing.
    """
    rng = np.random.default_rng(11)
    n, L, S = 128, 8, 10
    x = rng.random((n, S))
    k = ruzicka_gram(x)
    # the true ordered eigenbasis of this kernel, scaled as NestedLoRA's modes are
    z = _kernel_eigenbasis(k, L)

    base = nesting_objective(z, k @ z / n)["nesting_loss"]
    for s in range(4):
        q, _ = np.linalg.qr(np.random.default_rng(100 + s).normal(size=(L, L)))
        rot = nesting_objective(z @ q, k @ (z @ q) / n)["nesting_loss"]
        assert base <= rot + 1e-8, (base, rot, s)


def test_the_nesting_halves_must_be_independent():
    """The metric term estimates a SQUARED expectation, so reusing one half biases it upward.

    The reference implementation warns on this explicitly. Asserted by construction: computing
    both factors from the same half must give a strictly larger metric term than the split does,
    since it estimates E[X^2] where E[X]^2 is wanted.
    """
    rng = np.random.default_rng(13)
    f = rng.normal(size=(64, 6))
    half = 32
    v, m = joint_nesting_masks(6)
    split = float(np.sum(m * second_moment(f[:half]) * second_moment(f[half:])))
    same = float(np.sum(m * second_moment(f[:half]) * second_moment(f[:half])))
    assert same > split, (same, split)


def test_the_two_eigenvalue_estimators_agree_only_at_an_eigenbasis():
    """Their disagreement is a reference-free diagnostic, so it must actually discriminate."""
    rng = np.random.default_rng(17)
    n, L, S = 128, 8, 10
    x = rng.random((n, S))
    k = ruzicka_gram(x)
    z = _kernel_eigenbasis(k, L)

    at_basis = implied_spectrum(z, kf=k @ z / n)["estimator_disagreement"]
    q, _ = np.linalg.qr(rng.normal(size=(L, L)))
    rotated = implied_spectrum(z @ q, kf=k @ (z @ q) / n)["estimator_disagreement"]
    assert at_basis < rotated, (at_basis, rotated)


def test_the_report_is_pure_numpy_and_cannot_touch_a_weight():
    """The whole module is diagnostic. It must not be able to participate in autograd."""
    import torch

    rng = np.random.default_rng(19)
    x = rng.random((64, 9))
    z = torch.tensor(rng.normal(size=(64, 6)), requires_grad=True)
    # a caller must detach; the module operates on arrays and would raise on a grad-tracking
    # tensor rather than silently building a graph
    with pytest.raises((RuntimeError, TypeError)):
        eigenbasis_report(z, x)
    rep = eigenbasis_report(z.detach().numpy(), x, z_ref=rng.normal(size=(64, 6)),
                            ranks=(2, 4, 6))
    for key in ("orthogonality", "spectrum", "nesting", "subspace_vs_ref", "nesting_ref",
                "nesting_gap"):
        assert key in rep, key
    assert np.isfinite(rep["nesting_gap"])
    assert set(rep["subspace_vs_ref"]) == {2, 4, 6}


def test_ranks_beyond_the_available_width_are_skipped_not_clipped():
    """A column headed 'rank 64' must not silently hold the rank-16 value."""
    z = _ordered_basis(L=16)
    curve = subspace_curve(z, z, ranks=(8, 16, 32, 64))
    assert set(curve) == {8, 16}, curve


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_the_gram_is_chunked_and_bit_identical_to_the_unchunked_form():
    """The natural expression allocates a (Bx,By,S) cube: 3 GiB at B=2048, S=96.

    That is a hundred times the size of the (B,B) answer, in host memory, inside the training
    loop -- and the configured batch is 2048. Chunking must change nothing about the result, so
    this compares against the unchunked formula directly and at several block sizes: a chunked
    reduction that quietly changed the summation order would be a silent difference in the
    kernel every diagnostic is measured against.
    """
    rng = np.random.default_rng(0)
    x = rng.random((300, 96))
    s = x.sum(1)[:, None] + x.sum(1)[None, :]
    d = np.abs(x[:, None, :] - x[None, :, :]).sum(2)
    num, den = 0.5 * (s - d), 0.5 * (s + d)
    ref = np.zeros_like(num)
    ok = den > 1e-12
    ref[ok] = num[ok] / den[ok]
    for mb in (1, 8, 256):
        assert np.array_equal(ruzicka_gram(x, max_block_mib=mb), ref), mb

    # a block budget too small for even one row must still make progress, not divide to zero
    assert ruzicka_gram(x, max_block_mib=0).shape == (300, 300)
    # and an asymmetric pair of inputs still works
    y = rng.random((37, 96))
    g = ruzicka_gram(x, y, max_block_mib=1)
    assert g.shape == (300, 37) and np.isfinite(g).all()
    print("the chunked Gram is bit-identical at every block size")


NEURAL_SVD = "/Users/breallis/Dev/neural-svd"


@pytest.mark.skipif(not os.path.isdir(NEURAL_SVD),
                    reason="reference NeuralSVD implementation not present")
def test_the_nesting_forward_matches_the_reference_implementation_exactly():
    """Numerical equivalence with the paper's own code, which is the only decisive check.

    Every other test here compares against constructed cases -- an orthonormal basis, a component
    swap, a rotation. Those verify the diagnostic behaves sensibly, not that it computes the
    published objective. This compares against
    ``methods/nestedlora.py:NestedLoRALossFunctionEVD.forward`` directly, and the masks against
    ``get_joint_nesting_masks``.

    Skipped rather than vendored when the reference is absent: copying it in to keep a test green
    would mean testing our copy against our copy.
    """
    import torch

    sys.path.insert(0, NEURAL_SVD)
    from methods.nestedlora import (NestedLoRALossFunctionEVD,  # noqa: E402
                                    get_joint_nesting_masks)

    rng = np.random.default_rng(0)
    for B, L in ((64, 6), (128, 8), (256, 16)):
        f, tf = rng.normal(size=(B, L)), rng.normal(size=(B, L))
        v_ref, m_ref = get_joint_nesting_masks(np.ones(L))
        v_mine, m_mine = joint_nesting_masks(L)
        assert np.allclose(v_ref.numpy(), v_mine), (L, v_ref, v_mine)
        assert np.allclose(m_ref.numpy(), m_mine), L
        ref = float(NestedLoRALossFunctionEVD.apply(
            torch.tensor(f), torch.tensor(tf),
            torch.tensor(f[:B // 2]), torch.tensor(f[B // 2:2 * (B // 2)]), v_ref, m_ref))
        mine = nesting_objective(f, tf)["nesting_loss"]
        assert abs(ref - mine) < 1e-12 * max(abs(ref), 1.0), (B, L, ref, mine)


@pytest.mark.skipif(not os.path.isdir(NEURAL_SVD),
                    reason="reference NeuralSVD implementation not present")
def test_the_reference_backward_supplies_a_factor_autograd_cannot_see():
    """RESOLVED: the reference is right, and plain autograd on a detached Tf is wrong.

    The reference's custom backward returns ``-(4/B) * v * Tf`` for the operator term where plain
    autograd through the same forward gives ``-(2/B) * v * Tf`` -- a factor of two. The metric
    term's two gradients match exactly.

    The explanation: the operator term is ``-2 sum_l v_l <f_l, T f_l>``, and because ``T`` is
    self-adjoint with ``f`` on both sides, ``d/dtheta <f, Tf> = 2 <Tf, df/dtheta>``. The reference
    receives ``Tf`` already computed and so cannot see the second occurrence, and supplies the
    missing factor by hand.

    Confirmed against central finite differences on a free parameterisation
    (``tests/test_nested_lora.py::test_the_gradient_matches_central_finite_differences``): the
    reference backward is correct to ~2e-9, autograd with a detached ``Tf`` is wrong by ~6e-2, and
    building ``Tf`` inside the graph is correct to the same ~2e-9. ``nested_lora.py`` takes the
    third route and therefore needs no hand-written backward at all.

    Kept as a test so the factor cannot be "simplified" away by someone who notices autograd
    disagrees with it.
    """
    import torch

    sys.path.insert(0, NEURAL_SVD)
    from methods.nestedlora import (NestedLoRALossFunctionEVD,  # noqa: E402
                                    get_joint_nesting_masks)

    torch.manual_seed(0)
    B, L = 128, 8
    f0 = torch.randn(B, L, dtype=torch.float64)
    tf = torch.randn(B, L, dtype=torch.float64)
    v, m = get_joint_nesting_masks(np.ones(L))
    v, m = v.double(), m.double()

    a, a1, a2 = (f0.clone().requires_grad_(True), f0[:B // 2].clone().requires_grad_(True),
                 f0[B // 2:].clone().requires_grad_(True))
    NestedLoRALossFunctionEVD.apply(a, tf, a1, a2, v, m).backward()

    b, b1, b2 = (f0.clone().requires_grad_(True), f0[:B // 2].clone().requires_grad_(True),
                 f0[B // 2:].clone().requires_grad_(True))
    lam1, lam2 = b1.T @ b1 / b1.shape[0], b2.T @ b2 / b2.shape[0]
    (-2 * torch.einsum('l,bl,bl->b', v, b, tf).mean() + (m * lam1 * lam2).sum()).backward()

    # the metric term agrees exactly
    assert torch.allclose(a1.grad, b1.grad, atol=1e-12)
    assert torch.allclose(a2.grad, b2.grad, atol=1e-12)
    # the operator term is off by exactly 2x, and the reference is the correct one
    ratio = (a.grad / b.grad).flatten()
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-10), ratio[:5]


def test_the_esk_rank_curve_separates_a_noisy_tail_from_a_real_one():
    """``basis_domain_gap.rank_curve`` must reach opposite verdicts on the two cases it exists for.

    The trainer's rank curve is on DESK's z, so a flat result cannot distinguish a basis whose tail
    is noise from a basis whose tail is real signal the covariates cannot predict. Those imply
    opposite actions -- cut latent_dim and lose nothing, versus latent_dim is fine and the ENCODER is
    the ceiling. This function is what separates them, so a test that only checks it RUNS would miss
    the entire point; both verdicts are asserted against constructed answers.

    The construction: build z from the exact eigenvectors of the sample's Ružička matrix, then either
    replace the components past r8 with noise (tail is noise) or leave them exact (tail is real).
    """
    import numpy as np

    from scripts.diagnostics import basis_domain_gap as B

    rng = np.random.default_rng(1)
    X = rng.random((300, 30)) * rng.gamma(2, 1, (300, 1))

    # The stub must be a genuine FUNCTION OF ITS INPUT, not a precomputed array. rank_curve
    # subsamples before projecting, so a fixed array indexed in the original row order no longer
    # lines up with the permuted sample -- which is exactly how an earlier version of this test
    # broke when that (correct) ordering was introduced. Computing z from the rows actually handed
    # over keeps the stub honest under any sampling the function chooses.
    def make_stub(k_real, seed):
        def _proj(Xin, _zd, _ld):
            A = np.asarray(Xin, dtype="float64")
            R = B.ruzicka_pairs(A, A)
            w, V = np.linalg.eigh(R)
            o = np.argsort(w)[::-1]
            w, V = w[o], V[:, o]
            k = min(64, V.shape[1])
            z = V[:, :k] * np.sqrt(np.maximum(w[:k], 0))
            if k_real < k:
                z = z.copy()
                z[:, k_real:] = np.random.default_rng(seed).normal(
                    scale=np.sqrt(np.maximum(w[:k], 0))[k_real:] * 3,
                    size=(len(A), k - k_real))
            return z.astype("float32")
        return _proj

    def curve_with(k_real):
        real = B.project_points_to_z
        B.project_points_to_z = make_stub(k_real, 7)
        try:
            return B.rank_curve("t", X, "unused", 64, np.random.default_rng(0), n=300)
        finally:
            B.project_points_to_z = real

    c_noise = curve_with(8)
    c_real = curve_with(64)

    assert min(c_noise, key=c_noise.get) <= 8, (
        f"a tail replaced by noise still favoured rank {min(c_noise, key=c_noise.get)}; the curve "
        f"cannot detect a noisy tail and its verdict would be backwards")
    assert min(c_real, key=c_real.get) == max(c_real), (
        f"an EXACT tail favoured rank {min(c_real, key=c_real.get)} rather than the full width; the "
        f"curve would report real signal as noise and send latent_dim the wrong way")
    # monotone improvement when the tail is real -- the property the verdict text claims
    ks = sorted(c_real)
    assert all(c_real[a] >= c_real[b] for a, b in zip(ks, ks[1:])), c_real
    print(f"noisy tail -> best rank {min(c_noise, key=c_noise.get)}; "
          f"real tail -> best rank {min(c_real, key=c_real.get)}")
