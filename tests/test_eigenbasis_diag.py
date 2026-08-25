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
