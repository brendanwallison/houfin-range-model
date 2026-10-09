"""research/exp/atlas.py in a world where every answer is known: a LINEAR kernel and an identity ESK.

Spatial structure lives in components 0-3 and temporal change in 4-7, measured through noisy halves.
Rank 4 must keep ~all spatial and ~no temporal signal (the matched reading must say so), rank 8 both;
a DESK that moves half as far as the truth in time must read slope 0.5 with correlation 1, despite
the halves' noise -- that is what the cross-half estimator is for.
"""
import json
import os
import sys

import numpy as np
import pytest


def _world(tmp_path, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    n = 30
    cells = np.array([(r, c) for r in range(n) for c in range(n)], "int32")
    rr, cc = cells[:, 0] / n, cells[:, 1] / n
    spatial = np.column_stack([np.sin(3 * rr), np.cos(2 * cc), rr * cc, np.sin(2 * (rr + cc))])
    spatial = np.hstack([spatial * 3.0, np.zeros((len(cells), 4))])
    # one change shared by every cell: it cancels between neighbours, so SPATIAL differences live in
    # components 0-3 only while the TEMPORAL change lives in 4-7 only
    change = np.tile(np.r_[np.zeros(4), [0.8, -0.6, 0.5, 0.7]], (len(cells), 1))
    groups_x, groups_cells, groups_epoch = [], [], []
    truth = {}
    for e in (0, 1):
        x = spatial + (change if e == 1 else 0.0)
        for i, c in enumerate(cells):
            groups_x.append(x[i])
            groups_cells.append(c)
            groups_epoch.append(e)
            truth[(tuple(c), e)] = x[i]
    X = np.array(groups_x)
    xa = X + noise * rng.normal(size=X.shape)
    xb = X + noise * rng.normal(size=X.shape)
    E = {"cells": np.array(groups_cells, "int32"), "epoch": np.array(groups_epoch, "int8"),
         "withheld": np.zeros(len(X), bool), "n_years": np.full(len(X), 10, "int16"),
         "x_full": X, "x_a": xa, "x_b": xb, "z_full": X, "z_a": xa, "z_b": xb,
         "rows_flat": np.arange(len(X)), "rows_ptr": np.arange(len(X) + 1)}
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "esk_epochs.npz", **E)
    ho = np.zeros((n, n), bool)
    ho[: n // 2] = True
    np.savez(cache / "split.npz", holdout=ho)
    # planted DESK: the true communities with the TEMPORAL change halved
    desk = np.array([spatial[i] + (0.5 * change[i] if e == 1 else 0.0)
                     for e in (0, 1) for i in range(len(cells))])
    return str(cache), desk


def test_atlas_recovers_planted_retention_and_desk_attenuation(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "exp"))
    import atlas
    cache, desk = _world(tmp_path)
    monkeypatch.setattr(atlas, "KERNEL", lambda X, Y: (np.asarray(X) * np.asarray(Y)).sum(1))
    monkeypatch.setattr(atlas, "desk_epoch_z", lambda c, E, w: desk)
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["atlas", "--cache", cache, "--out", out, "--ranks", "4", "8",
                                      "--lags", "1", "2", "4"])
    atlas.main()
    r = json.load(open(os.path.join(out, "atlas.json")))
    t = r["temporal"]["all_cells"]["esk"]
    assert abs(t["4"]["retention"]) < 0.05 and t["8"]["retention"] == pytest.approx(1.0, abs=1e-9)
    # unbiased, not exact: at short lags the true spatial differences are small and the halves'
    # noise cross-terms in components 4-7 move the ratio -- the block-bootstrap CI must cover 1
    for lag in ("1", "2", "4"):
        lo, hi = r["spatial"][lag]["esk"]["4"]["retention_ci"]
        # 95% intervals over ~25 blocks, three lags: a near-miss is expected sometimes; a gross bias
        # (the overlapping-pair version read 0.50 with an interval ending at 0.74) must not pass
        assert lo - 0.05 < 1.0 < hi + 0.05, (lag, lo, hi)
        assert r["spatial"][lag]["esk"]["8"]["retention"] == pytest.approx(1.0, abs=1e-9)
    assert r["spatial"]["4"]["esk"]["4"]["retention"] == pytest.approx(1.0, abs=0.1)
    m = r["matched"]["4"]
    assert m["temporal_retention"] < 0.05 and m["spatial_retention_at_temporal_size"] > 0.85
    d = r["temporal"]["heldout_desk_raw"]["desk"]["8"]
    assert d["slope"] == pytest.approx(0.5, abs=0.05)
    assert d["corr"] == pytest.approx(1.0, abs=0.05)
    # at lag 1 the true spatial difference is tiny next to the halves' noise and the ratio is wild
    # (measured 0.50 on this world): it must at least carry a CI that admits the truth...
    s1 = r["spatial"]["1_heldout_desk_raw"]["desk"]["8"]
    assert s1["slope_ci"][0] < 1.0 < s1["slope_ci"][1] or s1["slope"] == pytest.approx(1.0, abs=0.1)
    # ...and where the spatial signal is real, DESK's (unattenuated) spatial slope is recovered
    s = r["spatial"]["4_heldout_desk_raw"]["desk"]["8"]
    assert s["slope"] == pytest.approx(1.0, abs=0.1)
    assert s["slope_ci"][0] < 1.0 < s["slope_ci"][1]
    share_t = np.array(r["components"]["temporal_share"])
    assert share_t[:4].sum() < 0.05 and share_t[4:].sum() > 0.95
