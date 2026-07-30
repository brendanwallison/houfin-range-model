"""Tests for the turnover-metric fixes that came out of the Mexico zero-turnover hunt.

Three defects, all confirmed by measurement on the real cube and all fixed here:

1. **Norm drift contaminated temporal turnover.** ``encoder_diagnostics`` computed the Z-side
   turnover as ``1 - Z.Z'`` while ``validate_spacetime`` used ``1 - cos``. Measured ``||Z||^2``
   medians are 0.73/0.81 rather than the contract's 1.0, so the dot version attributed ~73% of its
   value to the norm deficit rather than to any rotation of Z -- and where ``||Z||^2 > 1`` it went
   negative, which a similarity-based turnover cannot be. The two scripts must agree.
2. **An undefined Ružička pair scored as maximal turnover.** Two empty communities gave
   similarity 0, hence turnover 1.0, for a pair carrying no information at all.
3. **The BBS coverage gate disagreed with the trajectory it gates.** A (species, cell) with a BBS
   rate but no abundance counted toward ``min_coverage`` while the trajectory silently held it
   constant, emitting points whose deep and recent communities are bit-identical.

Runs standalone or under pytest; no GPU, no cluster data.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.trend_community import backward_trajectory, has_bbs


def _diag():
    """Exec encoder_diagnostics' pure helpers without triggering its config load."""
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "viz",
                        "encoder_diagnostics.py")
    src = open(path).read()
    ns = {"__name__": "encoder_diagnostics_pure", "__file__": path}
    exec(compile(src[:src.index("def main()")], path, "exec"), ns)   # noqa: S102
    return ns


def test_ruzicka_undefined_pair_is_nan():
    rp = _diag()["_ruzicka_pairs"]
    X = np.array([[0., 0.], [1., 2.], [2., 1.], [0., 0.]])
    out = rp(X, np.array([0, 1, 3]), np.array([3, 2, 0]))
    assert np.isnan(out[0]) and np.isnan(out[2]), "empty-vs-empty must be NaN, not 0"
    assert np.isfinite(out[1]) and abs(float(out[1]) - 0.5) < 1e-12, "real pairs unchanged"
    assert np.isnan(1.0 - out[0]), "turnover must propagate NaN, not claim complete turnover"
    print("ruzicka undefined pair -> NaN OK")


def test_temporal_turnover_uses_cosine_not_dot():
    """The Z-side turnover must be norm-invariant; the spatial kernel test must not be."""
    ns = _diag()
    cos_pairs, dot_pairs = ns["_cos_pairs"], ns["_feature_kernel_pairs"]
    i, j = np.array([0]), np.array([1])

    # Same directions, shrunk magnitudes -- exactly the measured ||Z||^2 ~ 0.73 situation.
    unit = np.array([[1.0, 0.0], [0.8, 0.6]])          # cos = 0.8
    shrunk = unit * np.sqrt(0.73)
    assert abs(float(cos_pairs(unit, i, j)[0]) - 0.8) < 1e-12
    assert abs(float(cos_pairs(shrunk, i, j)[0]) - 0.8) < 1e-12, "cosine must ignore the norm"
    # the dot, by contrast, moves with the norm -- which is why it inflated turnover
    assert abs(float(dot_pairs(shrunk, i, j)[0]) - 0.73 * 0.8) < 1e-12

    # An over-unit norm drives 1 - dot NEGATIVE (impossible turnover) but never 1 - cos.
    big = unit * np.sqrt(1.32)
    assert 1.0 - float(dot_pairs(big, i, j)[0]) < 0.0
    assert 0.0 <= 1.0 - float(cos_pairs(big, i, j)[0]) <= 2.0

    # identical vectors -> zero turnover; opposed -> 2; zero vector -> NaN, not a spurious 0
    same = np.array([[0.3, 0.4], [0.3, 0.4]])
    assert abs(float(cos_pairs(same, i, j)[0]) - 1.0) < 1e-12
    opp = np.array([[0.3, 0.4], [-0.3, -0.4]])
    assert abs(float(cos_pairs(opp, i, j)[0]) + 1.0) < 1e-12
    assert np.isnan(float(cos_pairs(np.array([[0., 0.], [1., 1.]]), i, j)[0]))
    print("temporal turnover uses cosine (norm-invariant) OK")


def test_paired_turnover_sides():
    """Observed side stays Ružička; both Z sides are cosine."""
    ns = _diag()
    paired = ns["paired_turnover"]
    rows = np.array([0, 0]); cols = np.array([0, 0]); years = np.array([1966, 2025])
    # Z shrunk by a factor but pointing the same way: cosine turnover must be ~0.
    Z = np.array([[1.0, 0.0], [0.5, 0.0]])
    X = np.array([[2.0, 0.0], [1.0, 0.0]])            # Ruzicka = 1/2 -> obs turnover 0.5
    r, c, fused, esk, desk = paired(X, Z, Z, rows, cols, years, 1966, 2025)
    assert len(r) == 1
    assert abs(float(desk[0])) < 1e-12, "collinear Z must give ~0 turnover regardless of magnitude"
    assert abs(float(esk[0])) < 1e-12
    assert abs(float(fused[0]) - 0.5) < 1e-12, "observed side must remain Ruzicka"
    # no matched cell -> empty, not a crash
    r0, _, _, _, d0 = paired(X, Z, Z, rows, cols, np.array([1966, 1966]), 1966, 2025)
    assert r0.size == 0 and d0.size == 0
    print("paired_turnover metric sides OK")


def test_bbs_gate_matches_trajectory():
    assert bool(has_bbs(np.array(2.0), np.array(5.0)))
    assert not bool(has_bbs(np.array(2.0), np.array(0.0)))       # abundance must be POSITIVE
    assert not bool(has_bbs(np.array(2.0), np.array(np.nan)))    # the old gate said True here
    assert not bool(has_bbs(np.array(np.nan), np.array(5.0)))

    anchor, k = np.array([[10.0]]), np.array([[1.0]])

    def traj(rate, abund, ebird):
        yrs, N = backward_trajectory(anchor, np.array([[rate]]), np.array([[ebird]]),
                                     np.array([[abund]]), k, [1966, 2025], 2025, 1966,
                                     2010.0, 1.5, np.log(100.0))
        return {int(y): float(N[i][0, 0]) for i, y in enumerate(yrs)}

    # BBS rate, no abundance, no eBird rate -> held constant: deep == recent, so Ružička is
    # exactly 1.0 and observed turnover exactly 0.0. This is the fabricated point.
    held = traj(2.0, np.nan, np.nan)
    assert abs(held[1966] - held[2025]) < 1e-12, held
    assert not bool(has_bbs(np.array(2.0), np.array(np.nan)))

    # with an abundance the BBS branch is usable and the trajectory genuinely varies
    real = traj(2.0, 5.0, np.nan)
    assert abs(real[1966] - real[2025]) > 1e-6, real
    print("BBS coverage gate matches the trajectory OK")


if __name__ == "__main__":
    test_ruzicka_undefined_pair_is_nan()
    test_temporal_turnover_uses_cosine_not_dot()
    test_paired_turnover_sides()
    test_bbs_gate_matches_trajectory()
    print("\nALL TURNOVER-METRIC CHECKS PASSED")
