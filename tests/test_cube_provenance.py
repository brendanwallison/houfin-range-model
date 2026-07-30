"""Tests for the zero-turnover artifact: cube fill provenance and the BBS coverage gate.

`turnover_maps.png` showed exactly 0.0 predicted turnover over Mexico and parts of Canada.
Two independent causes, both covered here:

1. **The cube gap-fills invalid land cells and never recorded that it had.** Stage-2 fill
   assigns a *year-invariant* static field, so such a cell has identical Z in every year and
   turnover ``1 - Z.Z'`` is forced to ``1 - ||Z||^2`` ~= 0 by the kernel contract. Indistinguishable
   from a real prediction of "no community change" until provenance is written.
2. **The trend reconstruction's coverage gate disagreed with its own trajectory** about what
   counts as BBS coverage, so cells that get held constant passed the gate and emitted points
   whose deep and recent community vectors are bit-identical -> Ružička 1.0 -> turnover 0.0.

Runs standalone or under pytest; no GPU, no cluster data.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from community_encoder.build_final_z_cube import (FILL_NEAREST, FILL_NODATA, FILL_PREDICTED,
                                                  FILL_SPATIAL, FILL_STATIC, fill_provenance)
from src.community_encoder.train_DESK.trend_community import backward_trajectory, has_bbs


def _load_encoder_diagnostics():
    """Exec encoder_diagnostics' pure functions without triggering its config load."""
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "viz",
                        "encoder_diagnostics.py")
    src = open(path).read()
    ns = {"__name__": "encoder_diagnostics_pure", "__file__": path}
    exec(compile(src[:src.index("def main()")], path, "exec"), ns)   # noqa: S102
    return ns


def test_fill_provenance_partitions_land():
    H, W = 40, 60
    land = np.zeros((H, W), bool); land[2:38, 3:57] = True
    valid = np.zeros((H, W), bool); valid[5:25, 10:40] = True     # DESK forward
    m1 = valid.copy(); m1[25:30, 10:40] = True                    # stage 1 added
    m2 = m1.copy(); m2[30:34, 10:40] = True                       # stage 2 added

    stage = fill_provenance(land, valid, m1, m2)
    classes = (FILL_PREDICTED, FILL_SPATIAL, FILL_STATIC, FILL_NEAREST)

    # exactly one class per land cell, nothing else anywhere
    hits = sum((stage == c).astype(int) for c in classes)
    assert (hits[land] == 1).all(), "a land cell must have exactly one provenance class"
    assert (hits[~land] == 0).all()
    assert (stage[~land] == FILL_NODATA).all()
    assert not (stage[land] == FILL_NODATA).any()
    assert int(land.sum()) == sum(int((stage == c).sum()) for c in classes)

    # stage == FILL_PREDICTED IS the per-year valid mask (that is why no separate file)
    assert np.array_equal(stage == FILL_PREDICTED, land & valid)
    # each class matches the mask difference it is meant to represent
    assert np.array_equal(stage == FILL_SPATIAL, land & m1 & ~valid)
    assert np.array_equal(stage == FILL_STATIC, land & m2 & ~m1)
    assert np.array_equal(stage == FILL_NEAREST, land & ~m2)

    # degenerate ends: nothing to fill, and nothing predicted
    allv = fill_provenance(land, land, land, land)
    assert (allv[land] == FILL_PREDICTED).all()
    none = np.zeros((H, W), bool)
    nothing = fill_provenance(land, none, none, none)
    assert (nothing[land] == FILL_NEAREST).all()
    print("fill provenance partitions the land mask OK")


def test_zero_turnover_artifact_reproduced_then_excluded():
    """Build the artifact deliberately, then show the provenance filter removes it."""
    ed = _load_encoder_diagnostics()
    paired_turnover, matched_cells = ed["paired_turnover"], ed["matched_cells"]

    # cell (0,0): Z changes between years -> real turnover.
    # cell (1,1): stage-2 filled, so Z is the SAME static vector in both years.
    rows = np.array([0, 0, 1, 1]); cols = np.array([0, 0, 1, 1])
    years = np.array([1966, 2025, 1966, 2025])
    Z = np.array([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
    X = Z.copy()

    _, _, _, _, desk = paired_turnover(X, Z, Z, rows, cols, years, 1966, 2025)
    assert any(abs(float(v)) < 1e-12 for v in desk), \
        "expected a year-invariant cell to yield turnover exactly 0 (the artifact)"

    predicted = np.array([True, True, False, False])
    r, c, _, _, desk2 = paired_turnover(X, Z, Z, rows, cols, years, 1966, 2025,
                                        predicted=predicted)
    assert len(r) == 1 and (int(r[0]), int(c[0])) == (0, 0)
    assert not any(abs(float(v)) < 1e-12 for v in desk2), "the fake zero must be gone"

    # the withheld count reported in the figure title
    _, all_cells = matched_cells(rows, cols, years, 1966, 2025)
    _, kept = matched_cells(rows, cols, years, 1966, 2025, predicted)
    assert len(all_cells) - len(kept) == 1

    # excluding everything must not raise
    r0, _, _, _, d0 = paired_turnover(X, Z, Z, rows, cols, years, 1966, 2025,
                                      predicted=np.zeros(4, bool))
    assert r0.size == 0 and d0.size == 0
    print("zero-turnover artifact reproduced then excluded OK")


def test_ruzicka_undefined_pair_is_nan():
    """Two empty communities are UNDEFINED, not maximally different."""
    ed = _load_encoder_diagnostics()
    rp = ed["_ruzicka_pairs"]
    X = np.array([[0., 0.], [1., 2.], [2., 1.], [0., 0.]])
    out = rp(X, np.array([0, 1, 3]), np.array([3, 2, 0]))
    assert np.isnan(out[0]) and np.isnan(out[2]), "empty-vs-empty must be NaN, not 0"
    assert np.isfinite(out[1]) and abs(float(out[1]) - 0.5) < 1e-12, "real pairs unchanged"
    # turnover 1 - sim therefore propagates NaN rather than claiming complete turnover
    assert np.isnan(1.0 - out[0])
    print("ruzicka undefined pair -> NaN OK")


def test_bbs_gate_matches_trajectory():
    """A BBS rate without a positive abundance is not coverage -- in BOTH places."""
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

    # BBS rate, no abundance, no eBird rate -> the trajectory holds it constant, which is
    # exactly the false-stability point the coverage gate exists to reject.
    held = traj(2.0, np.nan, np.nan)
    assert abs(held[1966] - held[2025]) < 1e-12, held
    assert not bool(has_bbs(np.array(2.0), np.array(np.nan))), \
        "the gate must agree with the trajectory that this is NOT coverage"

    # with an abundance the BBS branch is usable and the trajectory genuinely varies
    real = traj(2.0, 5.0, np.nan)
    assert abs(real[1966] - real[2025]) > 1e-6, real
    print("BBS coverage gate matches the trajectory OK")


if __name__ == "__main__":
    test_fill_provenance_partitions_land()
    test_zero_turnover_artifact_reproduced_then_excluded()
    test_ruzicka_undefined_pair_is_nan()
    test_bbs_gate_matches_trajectory()
    print("\nALL CUBE-PROVENANCE CHECKS PASSED")
