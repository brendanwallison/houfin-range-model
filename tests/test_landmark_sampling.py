"""Joint-ESK landmark sampling: the stratified and diversity-aware Nystrom draws.

These are ABLATIONS, not the production default (``esk.spacetime.landmark_mode`` is
``"random"``), but they are working alternatives on the LIVE joint ESK rather than
scaffolding for a retired target, so they stay.

Was ``test_enrich_direction.py``. Two of its five tests covered the retired enrich
direction-of-change target (``_weighted_median_cols`` and the scatter-add weighted mean)
and were removed with it.
"""
import numpy as np

from src.community_encoder.train_DESK.esk_kernel import diverse_landmarks, stratified_landmarks




def test_stratified_landmarks_recent_heavy():
    # 16k recent (label 0, ~9% of points) + 5 historical decade strata; recent_frac=0.5
    strata = np.concatenate([np.zeros(16000, int)] + [np.full(31400, d) for d in range(1, 6)])
    rng = np.random.default_rng(0)
    lm = stratified_landmarks(strata, 30000, rng, recent_label=0, recent_frac=0.5)
    assert len(lm) == 30000
    assert len(set(lm.tolist())) == len(lm)                       # no duplicates
    assert abs(np.mean(strata[lm] == 0) - 0.5) < 0.02             # recent boosted 9% -> ~50%
    # reproducible
    lm2 = stratified_landmarks(strata, 30000, np.random.default_rng(0), 0, 0.5)
    assert np.array_equal(lm, lm2)


def test_stratified_landmarks_exact_when_oversized():
    strata = np.array([0, 0, 1, 1, 2])
    lm = stratified_landmarks(strata, 100, np.random.default_rng(0))
    assert sorted(lm.tolist()) == [0, 1, 2, 3, 4]                 # all points are landmarks


def test_diverse_landmarks_cover_occupied_strata_and_are_reproducible():
    # Four deliberately distinct space/time/magnitude strata, with enough budget
    # that each must contribute at least one landmark.
    X = np.vstack([
        np.full((10, 2), 0.0), np.full((10, 2), 1.0),
        np.full((10, 2), 10.0), np.full((10, 2), 100.0),
    ])
    pidx = np.vstack([
        np.column_stack((np.zeros(10), np.zeros(10), np.full(10, 1966))),
        np.column_stack((np.zeros(10), np.full(10, 20), np.full(10, 1985))),
        np.column_stack((np.full(10, 20), np.zeros(10), np.full(10, 2005))),
        np.column_stack((np.full(10, 20), np.full(10, 20), np.full(10, 2025))),
    ]).astype(int)
    lm = diverse_landmarks(X, pidx, 8, np.random.default_rng(7),
                           spatial_bins=2, abundance_bins=4)
    assert len(lm) == len(np.unique(lm)) == 8
    assert {int(i // 10) for i in lm} == {0, 1, 2, 3}
    lm2 = diverse_landmarks(X, pidx, 8, np.random.default_rng(7),
                            spatial_bins=2, abundance_bins=4)
    assert np.array_equal(lm, lm2)
