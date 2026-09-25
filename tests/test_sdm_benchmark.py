"""Data-free tests for the correlative-SDM benchmark.

Everything here runs without ${HOUFIN_DATA} / ${HOUFIN_PROCESSED}, matching the
suite's standing rule that a bare ``pytest`` completes on a laptop with no data
tree. The pieces that need real grids (covariate assembly, the biolith fits) are
exercised only through their refusal paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.sdm_benchmark import baselines, designation


# ---------------------------------------------------------------- NB2 likelihood

def test_nb2_loglik_matches_scipy():
    nbinom = pytest.importorskip("scipy.stats").nbinom
    y = np.array([0, 1, 4, 12, 57])
    mu = np.array([0.5, 2.0, 3.0, 17.0, 40.0])
    phi = 2.5
    assert np.allclose(baselines.nb2_loglik(y, mu, phi),
                       nbinom.logpmf(y, phi, phi / (phi + mu)))


def test_nb2_tends_to_poisson_as_concentration_grows():
    poisson = pytest.importorskip("scipy.stats").poisson
    y = np.array([0, 3, 9])
    mu = np.array([1.0, 4.0, 8.0])
    assert np.allclose(baselines.nb2_loglik(y, mu, 1e7), poisson.logpmf(y, mu),
                       atol=1e-4)


@pytest.mark.parametrize("true_phi", [0.5, 2.0, 8.0])
def test_nb2_concentration_mle_recovers_truth(true_phi):
    rng = np.random.default_rng(0)
    mu = rng.gamma(2.0, 3.0, size=60_000)
    y = rng.negative_binomial(true_phi, true_phi / (true_phi + mu))
    est = baselines.nb2_concentration_mle(y, mu)
    assert abs(est - true_phi) / true_phi < 0.06


def test_nb2_deviance_explained_is_zero_for_the_null_mean():
    y = np.array([0.0, 1.0, 5.0, 2.0, 9.0])
    mu = np.full_like(y, y.mean())
    assert abs(baselines.nb2_deviance_explained(y, mu, 2.0)) < 1e-9


# ------------------------------------------------------------ spatial block CV

def _valid_grid():
    v = np.zeros((60, 80), dtype=bool)
    v[5:55, 5:75] = True
    return v


def test_block_folds_partition_validation_exactly_once():
    valid = _valid_grid()
    folds = list(baselines.spatial_block_folds(valid, 6, 5, buffer_cells=0, seed=0))
    assert len(folds) == 5
    stack = np.stack([v for _, v in folds])
    assert stack.sum() == valid.sum()                     # covers every valid cell
    assert stack.sum(axis=0).max() <= 1                   # and never twice


def test_block_folds_train_and_val_are_disjoint_and_buffered():
    valid = _valid_grid()
    for tr, va in baselines.spatial_block_folds(valid, 6, 4, buffer_cells=2, seed=1):
        assert not (tr & va).any()
        assert tr.sum() + va.sum() < valid.sum()          # a buffer was withheld
        assert not (tr & ~valid).any()


def test_rows_in_selects_by_cell():
    g = np.zeros((10, 10), dtype=bool)
    g[3, 4] = True
    sel = baselines.rows_in(g, np.array([3, 3, 0]), np.array([4, 5, 0]))
    assert sel.tolist() == [True, False, False]


# ------------------------------------------------------------ the 2x2 and metrics

def test_contingency_codes_follow_hypothesis_scenarios():
    niche = np.array([[True, False], [True, False]])
    occupied = np.array([[True, True], [False, False]])
    mask = np.ones((2, 2), dtype=bool)
    r = designation.contingency(niche, occupied, mask)
    assert r["counts"]["occupied_source"] == 1
    assert r["counts"]["occupied_sink"] == 1
    assert r["counts"]["unoccupied_source"] == 1
    assert r["counts"]["unoccupied_sink"] == 1
    assert r["occupied_sink_fraction"] == 0.25


def test_occupied_sink_is_the_cell_a_correlative_model_cannot_reach():
    """An occupancy-derived niche axis yields a diagonal table by construction."""
    rng = np.random.default_rng(0)
    occupied = rng.random((40, 40)) < 0.5
    mask = np.ones((40, 40), dtype=bool)
    r = designation.contingency(occupied.copy(), occupied, mask)   # niche := occupancy
    assert r["counts"]["occupied_sink"] == 0
    assert r["counts"]["unoccupied_source"] == 0


def test_cohens_kappa_bounds():
    a = np.array([[True, False], [True, False]])
    mask = np.ones((2, 2), dtype=bool)
    assert designation.cohens_kappa(a, a, mask) == pytest.approx(1.0)
    assert designation.cohens_kappa(a, ~a, mask) == pytest.approx(-1.0)


def test_auc_is_half_for_noise_and_one_for_perfect_separation():
    rng = np.random.default_rng(3)
    labels = np.array([True] * 300 + [False] * 300)
    assert designation.auc(rng.random(600), labels) == pytest.approx(0.5, abs=0.06)
    assert designation.auc(labels.astype(float), labels) == pytest.approx(1.0)


def test_max_sss_threshold_separates_a_clean_split():
    scores = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
    labels = np.concatenate([np.zeros(50, bool), np.ones(50, bool)])
    thr = designation.threshold_max_sss(scores, labels)
    assert 0.1 < thr <= 0.9
    assert designation.tss(scores, labels, thr) == pytest.approx(1.0)


def test_p10_threshold_is_the_tenth_percentile_of_presences():
    scores = np.linspace(0, 1, 101)
    labels = scores > 0.5
    assert designation.threshold_p10(scores, labels) == pytest.approx(
        np.percentile(scores[labels], 10))


def test_niche_from_lambda_treats_nan_as_not_niche_and_reports_finiteness():
    lam = np.array([[0.5, 1.5], [np.nan, 1.0]])
    niche, finite = designation.niche_from_lambda(lam)
    assert niche.tolist() == [[False, True], [False, True]]
    assert finite.tolist() == [[True, True], [False, True]]


# -------------------------------------------------- abundance -> occupancy step

def test_per_draw_conversion_differs_from_naive_mean_by_jensen():
    """1 - exp(-E[lambda]) is biased; the per-draw mean is the correct one."""
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    lam = np.array([[0.1], [10.0]])                      # 2 draws, 1 site
    per_draw = float((1.0 - np.exp(-lam)).mean(axis=0)[0])
    naive = float(1.0 - np.exp(-lam.mean(axis=0))[0])

    class R:
        samples = {"abundance": lam}

    got = occupancy.psi_from_nmixture(R())
    assert got[0] == pytest.approx(per_draw)
    assert got[0] != pytest.approx(naive)
    assert occupancy.analytic_poisson_check(R())["max_abs_gap"] > 0.05


def test_suggest_max_abundance_exceeds_the_largest_count():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    counts = np.array([[0.0, 5.0], [np.nan, 490.0]])
    mx, need = occupancy.suggest_max_abundance(counts, detection_floor=0.25)
    assert need == 1960
    assert mx > 490                      # the biolith default of 100 would truncate


def test_build_inputs_shapes_and_nan_policy():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    counts = np.array([[1.0, np.nan, 3.0], [0.0, 2.0, np.nan]])
    X = np.zeros((2, 4))
    out = occupancy.build_inputs(None, counts, [2015, 2016, 2017], X, binary=True)
    assert out["obs"].shape == (1, 2, 1, 3)
    assert out["obs_covs"].shape == (2, 1, 3, 1)
    assert np.isnan(out["obs"]).sum() == 2                # missing surveys preserved
    assert np.nanmax(out["obs"]) == 1.0                   # binarized

    with pytest.raises(ValueError, match="mask EVERY observation"):
        occupancy.build_inputs(None, counts, [2015, 2016, 2017],
                               np.full((2, 4), np.nan))


# ------------------------------------------------------------- refusal paths

def test_route_years_refuses_the_invasion_transient():
    data = pytest.importorskip("src.analysis.sdm_benchmark.data")
    with pytest.raises(ValueError, match="invasion transient"):
        data.route_years(1966, 1985)


# ------------------------------------------------- climate summary derivation

def test_climate_token_parses_the_bioyear_scheme():
    """Token is {base}_b{kk}m{MM}_{lvl} (src/data/preprocess/climate_grid.py)."""
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    m = cov._CLIM_TOKEN.match("Tmax_b01m08_q50")
    assert m.group("base") == "Tmax"
    assert int(m.group("m")) == 8
    assert m.group("lvl") == "q50"
    assert cov._CLIM_TOKEN.match("PPT_b06m01_q50").group("base") == "PPT"
    assert cov._CLIM_TOKEN.match("elev_q50") is None       # not a monthly channel


def test_climate_summary_groups_builds_annual_and_seasonal_means(monkeypatch):
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    # bio-year runs Aug(T-1)..Jul(T): b01=Aug ... b12=Jul
    months = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
    toks = [f"{base}_b{i+1:02d}m{mm:02d}_q50"
            for base in ("Tmax", "PPT") for i, mm in enumerate(months)]
    monkeypatch.setattr(cov, "_discover", lambda d, level=None: toks)

    g = cov.climate_summary_groups()
    assert len(g["Tmax_ann"]) == 12
    # Jun/Jul/Aug all fall in the Aug..Jul window: Aug=b01, Jun=b11, Jul=b12
    # Jun/Jul/Aug all fall in the Aug..Jul window (Aug=b01, Jun=b11, Jul=b12);
    # members come out in SUMMER_MONTHS order, which is irrelevant to the mean.
    assert g["Tmax_summer"] == ["Tmax_b11m06_q50", "Tmax_b12m07_q50",
                                "Tmax_b01m08_q50"]
    assert set(g) >= {"Tmax_ann", "Tmax_summer", "Tmax_winter",
                      "PPT_ann", "PPT_summer", "PPT_winter"}
    # winter is Dec/Jan/Feb whichever bio-year slots they land in
    assert len(g["Tmax_winter"]) == 3


def test_derived_source_is_unavailable_when_groups_are_empty():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    d = cov.DerivedSource("climate", "/nope/{var}_{year}_grid.tif", groups={})
    assert not d.available([2020])


def test_discover_handles_every_source_type_in_the_standard_tier():
    """Regression: discover() reached for .variables, which DerivedSource lacks.

    build_design calls discover() FIRST, so this crashed every `brt --tier
    standard` run before the climate rasters were ever opened. It stayed hidden
    locally because with no climate dir the derived source is empty and nothing
    exercised discover() on a mixed source list.
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    sources = [
        cov.Source("static", "/nope/{var}.tif", variables=("a", "b")),
        cov.Source("annual", "/nope/{var}_{year}.tif", variables=("c",), annual=True),
        cov.DerivedSource("derived", "/nope/{var}_{year}.tif",
                          groups={"x_ann": ["x_b01m08_q50"], "x_summer": ["x_b11m06_q50"]}),
    ]
    rep = cov.discover([2020, 2021], sources)          # must not raise
    assert [r["n_vars"] for r in rep] == [2, 1, 2]
    assert all(r["available"] is False for r in rep)   # nothing on disk


def test_every_source_type_exposes_n_features():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    assert cov.Source("s", "/t/{var}.tif", variables=("a", "b")).n_features == 2
    assert cov.Source("s", "/t.tif").n_features == 1
    assert cov.DerivedSource("d", "/t/{var}_{year}.tif",
                             groups={"a": ["m1"], "b": ["m2"]}).n_features == 2


def test_standard_tier_survives_discover_end_to_end():
    """The real standard_sources() list must pass through discover() cleanly."""
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rep = cov.discover([2020], cov.standard_sources())   # must not raise
    assert {r["name"] for r in rep} >= {"elev", "hyde", "soil", "luh3", "climate"}


def _write_grid_tif(path, value):
    """A raster matching the model grid exactly, filled with ``value``."""
    import rasterio
    from src.config_utils import load_data_config
    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as ref:
        profile = ref.profile
        shape, transform, crs = ref.shape, ref.transform, ref.crs
    profile.update(dtype="float32", count=1, nodata=None, compress="lzw")
    arr = np.full(shape, float(value), dtype="float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def test_build_design_averages_derived_groups(tmp_path):
    """The DerivedSource branch of build_design -- the code that runs on TACC.

    It had never executed anywhere: locally the climate dir is absent so the
    group dict is empty, and the unit tests only touched discover(). This writes
    real rasters on the model grid and checks the monthly members are averaged.
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    pytest.importorskip("rasterio")
    pd = pytest.importorskip("pandas")
    try:
        _write_grid_tif(tmp_path / "probe.tif", 1.0)
    except Exception as e:                      # no ref grid in this checkout
        pytest.skip(f"model grid reference unavailable: {e}")

    # Two monthly members per feature, constant-valued so the mean is exact.
    for var, val in [("T_b01m08_q50", 10.0), ("T_b11m06_q50", 20.0)]:
        _write_grid_tif(tmp_path / f"{var}_2020_grid.tif", val)

    src = cov.DerivedSource("climate", str(tmp_path / "{var}_{year}_grid.tif"),
                            groups={"T_summer": ["T_b01m08_q50", "T_b11m06_q50"]})
    assert src.n_features == 1
    assert src.available([2020])

    df = pd.DataFrame({"row": [0, 5, 10], "col": [0, 7, 21], "Year": [2020] * 3})
    X, names = cov.build_design(df, sources=[src])
    assert names == ["climate:T_summer"]
    assert X.shape == (3, 1)
    assert np.allclose(X[:, 0], 15.0)           # (10 + 20) / 2


def test_build_design_reads_the_matching_year_per_row(tmp_path):
    """Annual sources must sample each row's OWN year, not a single raster."""
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    pytest.importorskip("rasterio")
    pd = pytest.importorskip("pandas")
    try:
        _write_grid_tif(tmp_path / "probe.tif", 1.0)
    except Exception as e:
        pytest.skip(f"model grid reference unavailable: {e}")

    _write_grid_tif(tmp_path / "v_2020_grid.tif", 3.0)
    _write_grid_tif(tmp_path / "v_2021_grid.tif", 9.0)
    src = cov.Source("s", str(tmp_path / "{var}_{year}_grid.tif"),
                     variables=("v",), annual=True)
    df = pd.DataFrame({"row": [0, 0], "col": [0, 0], "Year": [2020, 2021]})
    X, _ = cov.build_design(df, sources=[src])
    assert X[:, 0].tolist() == [3.0, 9.0]
