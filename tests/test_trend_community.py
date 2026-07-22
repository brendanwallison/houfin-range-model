"""Trend-product community construction: blend weight, rate blend, backward
compounding, and point assembly (the pure numerical core)."""
import numpy as np

from src.community_encoder.train_DESK.trend_community import (
    assemble_points, backward_trajectory, blend_weight, blended_rate,
)


def test_blend_weight_monotone_and_bounds():
    yrs = np.arange(2000, 2024)
    w = blend_weight(yrs, crossover=2010.0, width=1.5)
    assert np.all((w > 0) & (w < 1))
    assert np.all(np.diff(w) > 0)                          # increasing toward present
    assert abs(blend_weight(2010.0, 2010.0, 1.5) - 0.5) < 1e-9   # 0.5 at crossover
    assert blend_weight(2022, 2010.0, 1.5) > 0.9           # eBird-heavy in its domain
    assert blend_weight(2000, 2010.0, 1.5) < 0.01          # BBS-dominant long ago


def test_blended_rate_nan_handling():
    bbs = np.array([2.0, np.nan, 5.0, np.nan])
    eb = np.array([-1.0, 3.0, np.nan, np.nan])
    r = blended_rate(bbs, eb, w=0.75)
    assert np.isclose(r[0], 0.75 * -1.0 + 0.25 * 2.0)      # both -> blend
    assert np.isclose(r[1], 3.0)                            # only eBird
    assert np.isclose(r[2], 5.0)                            # only BBS
    assert np.isnan(r[3])                                   # neither


def test_backward_trajectory_constant_rate():
    # Both products agree at +10%/yr (blend == 10 for any w); anchor 100.
    anchor = np.array([100.0])
    bbs = np.array([10.0]); eb = np.array([10.0])
    yrs, N = backward_trajectory(anchor, bbs, eb, sample_years=[2020, 2021, 2022, 2023],
                                 anchor_year=2023, first_year=2020,
                                 crossover=2010.0, width=1.5, clip_pct=100.0)
    assert yrs == [2020, 2021, 2022, 2023]
    assert np.isclose(N[3, 0], 100.0)                       # anchor year
    assert np.isclose(N[2, 0], 100.0 / 1.1)                 # one year back
    assert np.isclose(N[1, 0], 100.0 / 1.1**2)
    assert np.isclose(N[0, 0], 100.0 / 1.1**3)


def test_backward_trajectory_winsorize_and_hold():
    anchor = np.array([100.0, 50.0])
    bbs = np.array([500.0, np.nan])                         # 500%/yr artifact; and no data
    eb = np.array([np.nan, np.nan])
    yrs, N = backward_trajectory(anchor, bbs, eb, sample_years=[2021, 2022, 2023],
                                 anchor_year=2023, first_year=2021,
                                 crossover=2010.0, width=1.5, clip_pct=15.0)
    # species 0: clipped to +15%/yr -> divide by 1.15 each step back
    assert np.isclose(N[2, 0], 100.0)
    assert np.isclose(N[1, 0], 100.0 / 1.15)
    assert np.isclose(N[0, 0], 100.0 / 1.15**2)
    # species 1: neither product -> rate 0 -> held constant at anchor
    assert np.allclose(N[:, 1], 50.0)


def test_assemble_points_shapes_and_recent_first():
    S, H, W = 2, 2, 3
    rng = np.random.default_rng(0)
    anchor = rng.random((S, H, W)).astype("float32")
    anchor[:, 0, 2] = np.nan                                # one invalid cell (both species)
    bbs = np.full((S, H, W), 1.0); eb = np.full((S, H, W), 2.0)
    valid = np.any(np.isfinite(anchor), axis=0)
    M = int(valid.sum())                                    # 5 valid cells
    cfg = dict(anchor_year=2023, first_year=2019, stride=2,
               crossover=2010.0, width=1.5, clip_pct=15.0)
    X, pidx, years = assemble_points(anchor, bbs, eb, valid, cfg)
    # sample years: 2023 (recent) + 2021, 2019 (stride 2) -> 3 year-blocks
    assert years[0] == 2023                                 # recent first
    assert set(years) == {2023, 2021, 2019}
    assert X.shape == (M * 3, S)
    assert pidx.shape == (M * 3, 3)
    assert np.all(pidx[:M, 2] == 2023)                      # first block is the anchor year
    assert not np.isnan(X).any()                            # nan_to_num applied
    # recent block == the anchor's valid cells
    rr, cc = np.where(valid)
    assert np.allclose(X[:M], np.stack([anchor[s][rr, cc] for s in range(S)]).T)


def test_load_trend_grid_reindex(tmp_path):
    from src.community_encoder.train_DESK.trend_community import _load_trend_grid
    H, W = 2, 2
    rate = np.arange(3 * H * W, dtype="float32").reshape(3, H, W)
    p = tmp_path / "bbs.npz"
    np.savez(p, rate=rate, species_code=np.array(["aaa", "bbb", "ccc"]))
    out, missing = _load_trend_grid(str(p), ["ccc", "zzz", "aaa"], "rate")
    assert out.shape == (3, H, W)
    assert np.allclose(out[0], rate[2])                     # ccc
    assert np.isnan(out[1]).all()                           # zzz absent -> NaN
    assert np.allclose(out[2], rate[0])                     # aaa
    assert missing == ["zzz"]
