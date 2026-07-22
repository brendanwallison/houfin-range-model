"""Joint-ESK stratified landmarks + direction-of-change target primitives."""
import numpy as np

from src.community_encoder.train_DESK.esk_kernel import diverse_landmarks, stratified_landmarks
from src.community_encoder.train_DESK.desk_training import _weighted_median_cols


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


def test_weighted_median_cols():
    V = np.array([[1., 10.], [2., 20.], [3., 30.]])
    assert np.allclose(_weighted_median_cols(V, np.array([1., 1., 1.])), [2., 20.])  # = plain median
    assert _weighted_median_cols(V, np.array([5., 1., 1.]))[0] == 1.  # heavy weight pulls the median


def test_direction_scatter_add_matches_weighted_mean():
    import torch
    rng = np.random.default_rng(0)
    n_cell, L, npre = 3, 4, 7
    gathered = torch.tensor(rng.normal(size=(npre, L)).astype("float32"))
    cell = torch.tensor([0, 0, 1, 1, 1, 2, 2]); w = torch.tensor(rng.random(npre).astype("float32"))
    acc = torch.zeros(n_cell, L); ws = torch.zeros(n_cell, 1)
    acc.index_add_(0, cell, gathered * w[:, None]); ws.index_add_(0, cell, w[:, None])
    zp = (acc / ws.clamp_min(1e-8)).numpy()
    for c in range(n_cell):
        m = cell.numpy() == c
        ref = (gathered.numpy()[m] * w.numpy()[m, None]).sum(0) / w.numpy()[m].sum()
        assert np.allclose(zp[c], ref, atol=1e-5)
