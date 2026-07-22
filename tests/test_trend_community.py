"""Trend-product community construction: blend weight, rate blend, backward
compounding, and point assembly (the pure numerical core)."""
import numpy as np

from src.community_encoder.train_DESK.trend_community import (
    assemble_points, backward_trajectory, blend_weight, blended_rate, soft_clip,
)

# A no-op cap (knee/asymptote far above any test magnitude) for exact-compounding checks.
_NOCAP = (np.log(1e6), np.log(1e9))


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


def test_backward_method_b_deep_and_anchor():
    # Method B: deep past -> k*B (BBS absolute), present -> anchor. Flat rates.
    E = np.array([5.0]); rate_b = np.array([0.0]); rate_e = np.array([0.0])
    B = np.array([20.0]); k = np.array([0.5])                # k*B = 10 (deep target)
    yrs, N = backward_trajectory(E, rate_b, rate_e, B, k, [1966, 2023], 2023, 1966,
                                 2010.0, 1.5, *_NOCAP)
    assert np.isclose(N[yrs.index(2023), 0], 5.0)            # anchor exact
    assert np.isclose(N[yrs.index(1966), 0], 10.0, rtol=1e-3)  # deep -> k*B


def test_backward_method_b_absent_and_fallback():
    # sp0 absent now (E=0) -> 0 all years; sp1 has no BBS -> follows eBird (rate 0 -> held).
    E = np.array([0.0, 5.0]); rate_b = np.array([np.nan, np.nan])
    rate_e = np.array([np.nan, 0.0]); B = np.array([np.nan, np.nan]); k = np.array([1.0, 1.0])
    yrs, N = backward_trajectory(E, rate_b, rate_e, B, k, [1966, 2023], 2023, 1966,
                                 2010.0, 1.5, *_NOCAP)
    assert np.allclose(N[:, 0], 0.0)                        # absent -> 0
    assert np.allclose(N[:, 1], 5.0)                        # no BBS, eBird flat -> held


def test_backward_method_b_soft_cap():
    # Steep BBS decline on a nonzero base blows up deep; soft cap bounds the fold near asymptote.
    E = np.array([5.0]); rate_b = np.array([-20.0]); rate_e = np.array([0.0])
    B = np.array([5.0]); k = np.array([1.0])
    kn, asy = np.log(10.0), np.log(100.0)
    yrs, N = backward_trajectory(E, rate_b, rate_e, B, k, [1966, 2023], 2023, 1966,
                                 2010.0, 1.5, kn, asy)
    fold = N[yrs.index(1966), 0] / 5.0                      # decline -> past larger
    assert 10.0 < fold <= 100.0 * (1 + 1e-6)


def test_soft_clip_identity_then_saturates():
    kn, asy = np.log(10.0), np.log(100.0)
    assert np.isclose(soft_clip(np.log(5.0), kn, asy), np.log(5.0))   # 5x < knee -> untouched
    assert np.isclose(soft_clip(1.0, kn, asy), 1.0)
    x = np.log(40.0)                                          # 40x: past the 10x knee
    assert kn < soft_clip(x, kn, asy) < x                     # compressed, but not below the knee
    assert soft_clip(50.0, kn, asy) <= asy + 1e-9            # never exceeds the asymptote
    assert np.isclose(soft_clip(1e9, kn, asy), asy)          # saturates to the asymptote
    assert np.isclose(soft_clip(-2.0, kn, asy), -soft_clip(2.0, kn, asy))  # odd symmetry


def test_assemble_points_shapes_and_recent_first():
    S, H, W = 2, 2, 3
    rng = np.random.default_rng(0)
    anchor = rng.random((S, H, W)).astype("float32")
    anchor[:, 0, 2] = np.nan                                # one invalid cell (both species)
    bbs = np.full((S, H, W), 1.0); eb = np.full((S, H, W), 2.0)
    ba = np.full((S, H, W), 1.0); k = np.ones(S)            # BBS abundance + unit scale
    valid = np.any(np.isfinite(anchor), axis=0)
    M = int(valid.sum())                                    # 5 valid cells
    cfg = dict(anchor_year=2023, first_year=2019, stride=2, crossover=2010.0, width=1.5,
               soft_knee=np.log(1e6), soft_asymptote=np.log(1e9))
    X, pidx, years = assemble_points(anchor, bbs, eb, ba, k, valid, cfg, log1p=False)
    # sample years: 2023 (recent) + 2021, 2019 (stride 2) -> 3 year-blocks
    assert years[0] == 2023                                 # recent first
    assert set(years) == {2023, 2021, 2019}
    assert X.shape == (M * 3, S)
    assert pidx.shape == (M * 3, 3)
    assert np.all(pidx[:M, 2] == 2023)                      # first block is the anchor year
    assert not np.isnan(X).any()                            # nan_to_num applied
    # recent block == the anchor's valid cells (log1p off)
    rr, cc = np.where(valid)
    anchor_recent = np.stack([anchor[s][rr, cc] for s in range(S)]).T
    assert np.allclose(X[:M], anchor_recent)
    # log1p=True emits log-abundance
    Xl, _, _ = assemble_points(anchor, bbs, eb, ba, k, valid, cfg, log1p=True)
    assert np.allclose(Xl[:M], np.log1p(anchor_recent))


def test_coverage_gate_drops_sparse_history():
    # 2 species at 2 cells. Deep (BBS) coverage: cell 0 has both species' trend, cell 1 only one.
    S, H, W = 2, 1, 2
    anchor = np.ones((S, H, W))                             # both species present at both cells
    bbs = np.full((S, H, W), np.nan)
    bbs[0, 0, 0] = 1.0; bbs[1, 0, 0] = 1.0                  # cell 0: 2/2 covered
    bbs[0, 0, 1] = 1.0                                      # cell 1: 1/2 covered
    eb = np.full((S, H, W), np.nan)
    ba = np.ones((S, H, W)); k = np.ones(S)
    valid = np.ones((H, W), bool)
    base = dict(anchor_year=2023, first_year=1923, stride=100, crossover=2010.0, width=1.5,
                soft_knee=np.log(1e6), soft_asymptote=np.log(1e9))

    _, pidx, years = assemble_points(anchor, bbs, eb, ba, k, valid, {**base, "min_coverage": 0.75}, log1p=False)
    assert set(years) == {2023, 1923}
    assert (pidx[:, 2] == 2023).sum() == 2                  # anchor keeps both cells
    hist = pidx[pidx[:, 2] == 1923]
    assert len(hist) == 1 and hist[0, 1] == 0              # only the fully-covered cell survives

    _, pidx0, _ = assemble_points(anchor, bbs, eb, ba, k, valid, {**base, "min_coverage": 0.0}, log1p=False)
    assert (pidx0[:, 2] == 1923).sum() == 2                # gate off -> both cells kept


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
