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


def test_suggest_max_abundance_follows_the_unmarked_convention():
    """K = max(y) + 100, as unmarked::pcount uses.

    The old rule was max(y)/detection_floor with a 0.25 floor, giving 1960 for a
    490-bird western route -- over 3x the convention, and cost is LINEAR in the
    ceiling, so every NUTS gradient step was 3x more expensive for no gain.
    """
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    counts = np.array([[0.0, 5.0], [np.nan, 490.0]])
    mx, need = occupancy.suggest_max_abundance(counts)
    assert (mx, need) == (590, 590)
    assert mx > 490                      # the biolith default of 100 would truncate
    # the old behaviour stays reachable when asked for explicitly
    assert occupancy.suggest_max_abundance(counts, detection_floor=0.25)[1] == 1960


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


def test_probing_is_cheap_enough_for_a_login_node(monkeypatch):
    """Availability probing must not open every raster for every year.

    preflight is meant to run on a TACC login node. Checking geometry per file
    meant ~144 climate channels x 26 years = ~3,700 rasterio.open calls, each a
    Lustre metadata round-trip -- an I/O storm, and slow enough to look hung.
    Geometry is a property of how a source was BUILT, so one open per source is
    enough; existence is a cheap stat on sampled years.
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    years = list(range(2000, 2026))
    toks = [f"{b}_b{i+1:02d}m{m:02d}_q50"
            for b in ("Tmax", "Tmin", "Tave", "PPT", "CMD", "DD5")
            for i, m in enumerate([8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7])]
    monkeypatch.setattr(cov, "_discover", lambda d, level=None: toks)
    src = cov.DerivedSource("climate", "/x/{var}_{year}_grid.tif",
                            groups=cov.climate_summary_groups())

    n = {"stat": 0, "open": 0}
    monkeypatch.setattr(cov, "_exists_ok",
                        lambda p: (n.__setitem__("stat", n["stat"] + 1), True)[1])
    monkeypatch.setattr(cov, "_geometry_ok",
                        lambda p: (n.__setitem__("open", n["open"] + 1), True)[1])

    assert src.available(years) is True
    assert n["open"] == 1                               # geometry checked once
    assert n["stat"] <= len(toks) * cov.PROBE_YEARS     # years are sampled
    assert n["stat"] < len(toks) * len(years) / 4       # far below the naive cost


def test_sampled_probe_years_span_the_window():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    ys = cov._sample_years(list(range(2000, 2026)))
    assert len(ys) == cov.PROBE_YEARS
    assert ys[0] == 2000 and ys[-1] == 2025             # endpoints always probed
    assert cov._sample_years([2020, 2021]) == [2020, 2021]   # short windows kept whole


def test_probe_distinguishes_absent_from_off_grid(tmp_path):
    """The two need different fixes, so preflight must not conflate them."""
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    pytest.importorskip("rasterio")
    missing = cov._why(str(tmp_path / "nope.tif"), "src")
    assert "MISSING" in missing
    try:
        _write_grid_tif(tmp_path / "ok.tif", 1.0)
    except Exception as e:
        pytest.skip(f"model grid reference unavailable: {e}")
    assert cov._geometry_ok(str(tmp_path / "ok.tif"))


def test_processed_root_is_not_derived_from_the_data_root():
    """On HPC the two roots are different filesystems.

    HOUFIN_DATA lives under $SCRATCH and HOUFIN_PROCESSED under $WORK, so
    building the processed root as datasets_root/processed pointed at a
    directory that does not exist and reported every encoder tier as missing.
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    cfg = pytest.importorskip("src.config_utils").load_data_config()
    assert cov._PROC == cfg["processed_root"]


def test_last_complete_year_walks_back_past_trailing_gaps(tmp_path, monkeypatch):
    """Covariate streams end before BBS does; find the newest usable year."""
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    present = {2020, 2021, 2022}
    monkeypatch.setattr(cov, "standard_sources", lambda: [
        cov.Source("s", str(tmp_path / "{var}_{year}_grid.tif"),
                   variables=("v",), annual=True)])
    monkeypatch.setattr(cov, "_exists_ok",
                        lambda p: any(f"_{y}_" in p for y in present))
    assert cov.last_complete_year([2020, 2021, 2022, 2023, 2024], "standard") == 2022
    assert cov.last_complete_year([2023, 2024], "standard") is None


def test_last_complete_year_rejects_unknown_tier():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    with pytest.raises(ValueError, match="unknown tier"):
        cov.last_complete_year([2020], "nope")


def test_preflight_parser_scopes_tiers():
    """An unbuilt tier must not block a submission that never asks for it."""
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_sdm", repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    ap = mod.build_parser()
    assert ap.parse_args(["preflight"]).tiers == ["standard", "full", "latent"]
    assert ap.parse_args(["preflight", "--tiers", "standard", "full"]).tiers == \
        ["standard", "full"]
    with pytest.raises(SystemExit):
        ap.parse_args(["preflight", "--tiers", "bogus"])


def test_latent_tier_prefers_the_npz_the_model_actually_reads(tmp_path):
    """Z_latent_*.npy is an intermediate; the age model ingests Z_disp_*.npz.

    model_inputs globs Z_disp_*.npz and reads peek['Z_raw'], so a tree with
    completed MAP runs can legitimately have no .npy left. Looking only for
    .npy reported the tier unbuilt on a machine that had just fitted the model.
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    pd = pytest.importorskip("pandas")

    path, key = cov._latent_year_path(str(tmp_path), 2015)
    assert path.endswith("Z_latent_2015.npy") and key is None   # fallback

    np.savez(tmp_path / "Z_disp_2015.npz",
             Z_raw=np.zeros((1, 133, 224, 64), dtype="float32"))
    path, key = cov._latent_year_path(str(tmp_path), 2015)
    assert path.endswith("Z_disp_2015.npz") and key == "Z_raw"  # preferred


def test_latent_design_reads_z_raw_and_truncates(tmp_path):
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    pd = pytest.importorskip("pandas")
    z = np.zeros((1, 133, 224, 64), dtype="float32")
    z[0, 5, 7, :] = np.arange(64)
    np.savez(tmp_path / "Z_disp_2015.npz", Z_raw=z)

    df = pd.DataFrame({"row": [5], "col": [7], "Year": [2015]})
    X, names = cov.latent_z_design(df, z_dir=str(tmp_path), latent_dim=24)
    assert X.shape == (1, 24) and len(names) == 24
    assert X[0].tolist() == list(range(24))      # leading axis dropped, truncated


def test_latent_design_accepts_the_bare_npy_intermediate(tmp_path):
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    pd = pytest.importorskip("pandas")
    z = np.zeros((133, 224, 64), dtype="float32")
    z[5, 7, :] = np.arange(64) * 2.0
    np.save(tmp_path / "Z_latent_2015.npy", z)
    df = pd.DataFrame({"row": [5], "col": [7], "Year": [2015]})
    X, _ = cov.latent_z_design(df, z_dir=str(tmp_path), latent_dim=24)
    assert X[0].tolist() == [i * 2.0 for i in range(24)]


# ------------------------------------------------- nmixture enumeration cost

def test_enumeration_bytes_matches_the_observed_oom():
    """Verified against a real failure on an A100.

    3853 sites at max_abundance=1960 requested exactly 55.20 GiB; biolith's
    Categorical log_prob broadcast makes the cost QUADRATIC in max_abundance,
    which is why a ceiling derived from a 490-bird western route is unusable.
    """
    occ = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                              reason="needs numpyro/jax")
    gib = occ.enumeration_bytes(3853, 1960) / 2 ** 30
    assert gib == pytest.approx(55.20, abs=0.01)
    # quadratic, not linear: halving the ceiling quarters the cost
    assert occ.enumeration_bytes(3853, 979) == pytest.approx(
        occ.enumeration_bytes(3853, 1959) / 4, rel=0.01)


def test_budget_guard_refuses_before_the_gpu_allocates():
    occ = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                              reason="needs numpyro/jax")
    with pytest.raises(MemoryError, match="QUADRATIC"):
        occ.check_enumeration_budget(3853, 1960, budget_gib=8.0)
    assert occ.check_enumeration_budget(3853, 100, budget_gib=8.0) > 0


def test_max_abundance_for_budget_is_the_largest_that_fits():
    occ = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                              reason="needs numpyro/jax")
    n, budget = 3853, 8 * 2 ** 30
    m = occ.max_abundance_for_budget(n, budget)
    assert occ.enumeration_bytes(n, m) <= budget
    assert occ.enumeration_bytes(n, m + 2) > budget      # and it is tight


def test_biolith_parser_defaults_to_the_affordable_nmixture():
    """The abundance fit runs by DEFAULT again.

    It was briefly made opt-in because biolith's enumerated implementation
    needed 55 GiB at this data's ceiling. The direct marginal sum is linear and
    needs ~0.3 GiB, so disabling the model is no longer the price of running the
    benchmark; --enumerate-nmixture keeps biolith's version for cross-checking.
    """
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_sdm2", repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    ap = mod.build_parser()
    a = ap.parse_args(["biolith"])
    assert a.no_nmixture is False            # abundance runs by default
    assert a.enumerate_nmixture is False     # via the direct marginal sum
    assert a.budget_gib == 8.0
    assert ap.parse_args(["biolith", "--no-nmixture"]).no_nmixture is True
    assert ap.parse_args(["biolith", "--enumerate-nmixture"]).enumerate_nmixture
    assert ap.parse_args(["biolith", "--max-abundance", "460"]).max_abundance == 460


# --------------------------------------- direct-sum N-mixture (Royle 2004)

def test_direct_sum_is_linear_not_quadratic_in_the_ceiling():
    """This is the whole point: biolith's enumeration is quadratic, this is not."""
    nm = pytest.importorskip("src.analysis.sdm_benchmark.nmixture_marginal",
                             reason="needs numpyro/jax")
    occ = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                              reason="needs numpyro/jax")
    n, rep = 3853, 10
    # doubling the ceiling doubles the direct sum, but quadruples enumeration
    assert nm.enumeration_free_bytes(n, 1959, rep) == pytest.approx(
        2 * nm.enumeration_free_bytes(n, 979, rep), rel=0.01)
    assert occ.enumeration_bytes(n, 1959) == pytest.approx(
        4 * occ.enumeration_bytes(n, 979), rel=0.01)
    # at the ceiling this data implies, 55 GiB becomes well under 1 GiB
    assert occ.enumeration_bytes(n, 1960) / 2 ** 30 > 50
    assert nm.enumeration_free_bytes(n, 1960, rep) / 2 ** 30 < 1


def test_log_binom_pmf_matches_scipy_and_is_minus_inf_above_n():
    nm = pytest.importorskip("src.analysis.sdm_benchmark.nmixture_marginal",
                             reason="needs numpyro/jax")
    binom = pytest.importorskip("scipy.stats").binom
    y = np.array([0.0, 2.0, 5.0])
    n = np.array([5.0, 5.0, 5.0])
    p = np.array([0.3, 0.3, 0.3])
    got = np.asarray(nm._log_binom_pmf(y, n, p))
    assert np.allclose(got, binom.logpmf(y, n, p), atol=1e-5)
    # N must be at least the observed count
    assert np.isneginf(float(nm._log_binom_pmf(np.array(6.0), np.array(5.0),
                                               np.array(0.3))))


def test_marginal_likelihood_ignores_unsurveyed_visits():
    """NaN visits must not contribute; masking them changes nothing else."""
    nm = pytest.importorskip("src.analysis.sdm_benchmark.nmixture_marginal",
                             reason="needs numpyro/jax")
    jnp = pytest.importorskip("jax.numpy")
    y = jnp.array([[2.0, 3.0, 0.0]])
    lam = jnp.array([4.0])
    p = jnp.array([[0.5, 0.5, 0.5]])
    full = nm.marginal_log_likelihood(y, jnp.array([[True, True, True]]), lam, p, 40)
    part = nm.marginal_log_likelihood(y, jnp.array([[True, True, False]]), lam, p, 40)
    two = nm.marginal_log_likelihood(y[:, :2], jnp.array([[True, True]]),
                                     lam, p[:, :2], 40)
    assert float(part[0]) == pytest.approx(float(two[0]), rel=1e-5)
    assert float(part[0]) != pytest.approx(float(full[0]), rel=1e-5)


def test_marginal_likelihood_is_a_proper_log_probability():
    nm = pytest.importorskip("src.analysis.sdm_benchmark.nmixture_marginal",
                             reason="needs numpyro/jax")
    jnp = pytest.importorskip("jax.numpy")
    # one site, one visit: summing exp(logL) over all observable y must give 1
    lam = jnp.array([3.0])
    p = jnp.array([[0.4]])
    tot = 0.0
    for yv in range(0, 60):
        ll = nm.marginal_log_likelihood(jnp.array([[float(yv)]]),
                                        jnp.array([[True]]), lam, p, 200)
        tot += float(np.exp(np.asarray(ll)[0]))
    assert tot == pytest.approx(1.0, abs=1e-4)


def test_sampler_diagnostics_flags_treedepth_saturation():
    """A run pinned at 1023 leapfrog steps is hitting max_tree_depth.

    That is a badly conditioned posterior -- slow AND poorly mixed -- and is
    what the first full run showed. Reporting it beats reading it off a
    progress bar afterwards.
    """
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")

    class _M:
        def get_extra_fields(self):
            return {"diverging": np.array([False, True, False, True]),
                    "num_steps": np.array([1023, 1023, 1023, 7])}

    d = occupancy.sampler_diagnostics(_M())
    assert d["divergences"] == 2
    assert d["frac_at_max_treedepth"] == pytest.approx(0.75)
    assert d["mean_steps"] == pytest.approx((1023 * 3 + 7) / 4)


def test_sampler_diagnostics_is_silent_without_extra_fields():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")

    class _M:
        def get_extra_fields(self):
            raise RuntimeError("not collected")

    assert occupancy.sampler_diagnostics(_M()) == {}


def test_max_abundance_sensitivity_reports_no_shift_for_identical_fits():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    a = np.array([0.1, 0.5, 0.9])
    r = occupancy.max_abundance_sensitivity(a, a)
    assert r["max_abs_shift"] == 0.0 and r["correlation"] == pytest.approx(1.0)
    r2 = occupancy.max_abundance_sensitivity(a, a + 0.2)
    assert r2["max_abs_shift"] == pytest.approx(0.2)


def test_skip_existing_flag_is_available_on_the_fitting_stages():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_sdm3", repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    ap = mod.build_parser()
    assert ap.parse_args(["brt"]).skip_existing is False
    assert ap.parse_args(["brt", "--skip-existing"]).skip_existing is True
    assert ap.parse_args(["biolith", "--skip-existing"]).skip_existing is True


# ------------------------------ NB latent abundance (unmarked's mixture="NB")

def test_nb2_latent_pmf_matches_scipy():
    nm = pytest.importorskip("src.analysis.sdm_benchmark.nmixture_marginal",
                             reason="needs numpyro/jax")
    jnp = pytest.importorskip("jax.numpy")
    nbinom = pytest.importorskip("scipy.stats").nbinom
    n = jnp.arange(0.0, 20.0)
    got = np.asarray(nm._log_nb2_pmf(n, jnp.float32(4.0), jnp.float32(2.5)))
    assert np.allclose(got, nbinom.logpmf(np.arange(20), 2.5, 2.5 / (2.5 + 4.0)),
                       atol=1e-5)


@pytest.mark.parametrize("phi", [None, 2.0])
def test_marginal_likelihood_integrates_to_one_for_both_mixtures(phi):
    """Poisson and NB latents must each give a proper likelihood."""
    nm = pytest.importorskip("src.analysis.sdm_benchmark.nmixture_marginal",
                             reason="needs numpyro/jax")
    jnp = pytest.importorskip("jax.numpy")
    lam, p = jnp.array([3.0]), jnp.array([[0.4]])
    ph = None if phi is None else jnp.float32(phi)
    tot = sum(float(np.exp(np.asarray(nm.marginal_log_likelihood(
        jnp.array([[float(y)]]), jnp.array([[True]]), lam, p, 300, phi=ph))[0]))
        for y in range(0, 150))
    assert tot == pytest.approx(1.0, abs=1e-4)


def test_nb_latent_is_overdispersed_and_poisson_is_not():
    """Binomial thinning of a Poisson stays Poisson (var/mean == 1 exactly)."""
    nm = pytest.importorskip("src.analysis.sdm_benchmark.nmixture_marginal",
                             reason="needs numpyro/jax")
    jnp = pytest.importorskip("jax.numpy")

    def var_over_mean(phi):
        ys = np.arange(200)
        ll = np.asarray(nm.marginal_log_likelihood(
            jnp.array([[float(y)] for y in ys]), jnp.ones((200, 1), bool),
            jnp.full((200,), 3.0), jnp.full((200, 1), 0.4), 400, phi=phi))
        w = np.exp(ll); w /= w.sum()
        m = (w * ys).sum()
        return ((w * (ys - m) ** 2).sum()) / m

    assert var_over_mean(None) == pytest.approx(1.0, abs=1e-3)      # Poisson
    assert var_over_mean(jnp.float32(5.0)) > 1.1                    # NB, mild
    assert var_over_mean(jnp.float32(1.0)) > 2.0                    # NB, strong


def test_biolith_parser_defaults_to_nb_without_site_random_effects():
    """Per-site effects compete with the covariates that make the map."""
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_sdm4", repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    ap = mod.build_parser()
    a = ap.parse_args(["biolith"])
    assert a.mixture == "NB"
    assert a.site_random_effects is False
    assert ap.parse_args(["biolith", "--mixture", "P"]).mixture == "P"
    assert ap.parse_args(["biolith", "--site-random-effects"]).site_random_effects


def test_convergence_uses_per_chain_draws_not_concatenated_ones():
    """get_samples() concatenates chains, which made every R-hat come back nan.

    Gelman-Rubin needs (n_chains, n_draws, ...). Feeding it the flattened array
    compares a chain with itself, and the first real run reported r_hat_max=nan
    for every fit -- so the benchmark had no convergence signal at all.
    """
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    rng = np.random.default_rng(0)

    class _MCMC:
        def __init__(self, grouped):
            self._g = grouped

        def get_samples(self, group_by_chain=False):
            if group_by_chain:
                return self._g
            return {k: v.reshape((-1,) + v.shape[2:]) for k, v in self._g.items()}

    class _R:
        pass

    good = {"beta": rng.normal(size=(4, 200, 3)),
            "beta0": rng.normal(size=(4, 200))}
    r = _R(); r.mcmc = _MCMC(good)
    cv = occupancy.convergence(r)
    assert "error" not in cv
    assert np.isfinite(occupancy.worst_r_hat(cv))
    assert occupancy.worst_r_hat(cv) < 1.1
    assert cv["worst"]["param"] in {"beta", "beta0"}


def test_convergence_reports_an_error_rather_than_nan_when_it_cannot_compute():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")

    class _R:
        pass

    cv = occupancy.convergence(_R())                 # no .mcmc at all
    assert "error" in cv
    assert np.isnan(occupancy.worst_r_hat(cv))

    class _MCMC1:
        def get_samples(self, group_by_chain=False):
            return {"beta0": np.zeros((1, 100))}     # a single chain

    r = _R(); r.mcmc = _MCMC1()
    assert "error" in occupancy.convergence(r)       # R-hat needs >= 2 chains


def test_convergence_skips_the_thousands_of_derived_site_quantities():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    rng = np.random.default_rng(1)

    class _MCMC:
        def get_samples(self, group_by_chain=False):
            return {"beta0": rng.normal(size=(4, 50)),
                    "abundance": rng.normal(size=(4, 50, 3853)),
                    "psi": rng.normal(size=(4, 50, 3853))}

    class _R:
        pass

    r = _R(); r.mcmc = _MCMC()
    cv = occupancy.convergence(r)
    assert "abundance" not in cv and "psi" not in cv
    assert "beta0" in cv


def test_dump_design_flag_exists():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_sdm5", repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    ap = mod.build_parser()
    assert ap.parse_args(["biolith"]).dump_design is False
    assert ap.parse_args(["biolith", "--dump-design"]).dump_design is True


# ---------------------------------------- PCA for the linear (biolith) models

def test_pca_removes_the_collinearity_a_linear_model_chokes_on():
    """The encoder bank is 12 bases x 12 months plus duplicate quantile levels.

    A boosted tree is indifferent to that; a linear model is not -- collinear
    columns make a ridge in the coefficient posterior, which is what leaves NUTS
    saturating max tree depth every iteration.
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(0)
    blocks = []
    for _ in range(8):
        base = rng.normal(size=(500, 1))
        blocks.append(base + 0.15 * rng.normal(size=(500, 12)))   # near-duplicates
    X = np.hstack(blocks)

    raw = cov.condition_number(X)
    Z, names, info = cov.pca_reduce(X, var_target=0.99)
    assert raw > 10                                   # genuinely ill-conditioned
    assert cov.condition_number(Z) == pytest.approx(1.0, abs=1e-3)   # orthogonal
    assert Z.shape[1] < X.shape[1]                    # and it reduces
    assert 0.98 <= info["explained_variance"] <= 1.0
    assert names == [f"pc{i:03d}" for i in range(Z.shape[1])]


def test_pca_rotation_can_be_applied_to_heldout_rows():
    """Fit the rotation on train only, then transform -- no leakage."""
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 12))
    Z, _, info = cov.pca_reduce(X, n_components=5)
    Z2, _, _ = cov.pca_reduce(X[:20], basis=info["basis"], center=info["center"],
                              mu=info["mu"], sd=info["sd"])
    assert Z2.shape == (20, 5)
    assert np.allclose(Z2, Z[:20], atol=1e-4)


def test_pca_component_count_is_controllable():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 20))
    assert cov.pca_reduce(X, n_components=7)[0].shape[1] == 7
    # a lower variance target keeps fewer components
    lo = cov.pca_reduce(X, var_target=0.5)[0].shape[1]
    hi = cov.pca_reduce(X, var_target=0.99)[0].shape[1]
    assert lo < hi


def test_condition_number_flags_duplicate_columns():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(3)
    good = rng.normal(size=(200, 5))
    assert cov.condition_number(good) < 5
    dup = np.hstack([good, good[:, :1] + 1e-6 * rng.normal(size=(200, 1))])
    assert cov.condition_number(dup) > 100


def test_pca_flags_parse():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_sdm6", repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    ap = mod.build_parser()
    assert ap.parse_args(["biolith"]).pca is None            # off unless asked
    assert ap.parse_args(["biolith", "--pca"]).pca == -1     # bare = use var target
    assert ap.parse_args(["biolith", "--pca", "30"]).pca == 30
    assert ap.parse_args(["biolith", "--pca-var", "0.95"]).pca_var == 0.95


def test_per_stream_pca_protects_small_streams_when_truncating():
    """Per-stream matters only when components are being DISCARDED.

    Climate is ~80% of the bank, so a variance budget applied globally can
    squeeze the ~10 human-footprint channels out of the retained set invisibly.
    That is the one thing per-stream buys; it does NOT fix conditioning (see
    test_per_stream_pca_leaves_cross_stream_collinearity).
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(0)
    cols, names = [], []

    def add(stream, n, redundant=False):
        if redundant:
            base = rng.normal(size=(400, 1))
            a = base + 0.15 * rng.normal(size=(400, n))
        else:
            a = rng.normal(size=(400, n))
        cols.append(a)
        names.extend(f"{stream}:{i:03d}" for i in range(n))

    for _ in range(20):
        add("climate", 12, redundant=True)        # 240 near-duplicate channels
    add("soil", 16); add("bui", 7); add("hyde", 3)
    X = np.hstack(cols)

    Z, out_names, info = cov.pca_reduce_by_stream(X, names, var_target=0.99)
    per = info["per_stream"]
    assert per["climate"]["n_in"] == 240
    assert per["climate"]["n_out"] < 240           # redundancy collapsed
    # the small streams survive intact
    assert per["hyde"]["n_out"] == 3
    assert per["bui"]["n_out"] == 7
    # stream identity is preserved in the names, so loadings stay interpretable
    assert any(n.startswith("hyde:pc") for n in out_names)
    assert cov.condition_number(Z) < cov.condition_number(X)


def test_per_stream_rotation_applies_to_heldout_rows():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 12))
    names = [f"a:{i}" for i in range(6)] + [f"b:{i}" for i in range(6)]
    Z, _, info = cov.pca_reduce_by_stream(X, names, var_target=0.99)
    Z2, _, _ = cov.pca_reduce_by_stream(X[:20], names, fitted=info["fitted"])
    assert Z2.shape[1] == Z.shape[1]
    assert np.allclose(Z2, Z[:20], atol=1e-4)


def test_stream_of_parses_channel_names():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    assert cov.stream_of("climate:012") == "climate"
    assert cov.stream_of("hyde:popd_2020") == "hyde"
    assert cov.stream_of("z00") == "_"          # latent tier has no stream prefix


def test_pca_per_stream_is_opt_in_and_global_is_the_default():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_sdm7", repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    ap = mod.build_parser()
    assert ap.parse_args(["biolith", "--pca"]).pca_per_stream is False
    assert ap.parse_args(["biolith", "--pca", "--pca-per-stream"]).pca_per_stream


def test_rotation_alone_fixes_conditioning_so_truncation_is_not_required():
    """The default keeps every numerically meaningful component.

    Conditioning is fixed by the ROTATION -- components are standardized before
    fitting, so the design is orthonormal however many are kept. Truncating is a
    separate, optional decision.
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(0)
    cols = []
    for _ in range(10):
        base = rng.normal(size=(600, 1))
        cols.append(base + 0.15 * rng.normal(size=(600, 12)))
    X = np.hstack(cols)

    Z, _, _ = cov.pca_reduce(X)                       # default: no truncation
    assert Z.shape[1] == X.shape[1]                   # nothing dropped
    assert cov.condition_number(Z) == pytest.approx(1.0, abs=1e-3)
    assert cov.condition_number(X) > 5                # the raw matrix was not
    # a variance budget is opt-in and does truncate
    assert cov.pca_reduce(X, var_target=0.99)[0].shape[1] < X.shape[1]


def test_variance_budget_can_discard_the_predictive_direction():
    """The reason the 99% default was wrong, as an executable demonstration."""
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(0)
    cols = []
    for _ in range(20):
        base = rng.normal(size=(1500, 1))
        cols.append(base + 0.15 * rng.normal(size=(1500, 12)))
    X = np.hstack(cols)
    Z, _, _ = cov.pca_reduce(X)                       # rotate, keep all
    kept_99 = cov.pca_reduce(X, var_target=0.99)[0].shape[1]
    # a component beyond the 99% budget still exists and carries variance
    assert kept_99 < Z.shape[1]
    assert np.std(Z[:, kept_99 + 10]) > 0             # not numerically dead


def test_select_n_components_cv_prefers_parsimony_on_a_flat_curve():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(800, 40))
    # signal entirely in the first component -> small k should suffice
    y = (rng.random(800) < 1 / (1 + np.exp(-Z[:, 0])))
    k, rep = cov.select_n_components_cv(Z, y, n_folds=4)
    assert 1 <= k <= 40
    assert rep["chosen_k"] <= rep["best_k"]           # one-SE rule never grows k
    assert "mean_auc" in rep


def test_numerical_rank_drops_only_degenerate_directions():
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 8))
    assert cov.numerical_rank(X) == 8
    X2 = np.hstack([X, X[:, :1]])                     # an exact duplicate column
    assert cov.numerical_rank(X2) == 8


def test_per_stream_pca_leaves_cross_stream_collinearity():
    """Per-stream rotation does nothing ACROSS streams, and the streams correlate.

    Temperature with elevation through the lapse rate; HYDE population with
    HISDAC built-up and urban land use all measuring human footprint. An earlier
    version of this benchmark made per-stream the default on the strength of a
    test whose streams were independent noise -- which assumed away exactly this.
    """
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(0)
    S = 1500
    temp = rng.normal(size=(S, 1))
    relief = temp * 0.8 + 0.6 * rng.normal(size=(S, 1))     # elevation <-> climate
    human = rng.normal(size=(S, 1))                          # hyde <-> bui

    cols, names = [], []

    def add(stream, n, driver, noise):
        cols.append(driver + noise * rng.normal(size=(S, n)))
        names.extend(f"{stream}:{i:03d}" for i in range(n))

    for _ in range(8):
        add("climate", 12, temp, 0.2)
    add("elevation", 3, relief, 0.1)
    add("hyde", 3, human, 0.1)
    add("bui", 7, human, 0.15)
    X = np.hstack(cols)

    Zs, ns, _ = cov.pca_reduce_by_stream(X, names)
    Zg, _, _ = cov.pca_reduce(X)
    assert cov.condition_number(Zg) == pytest.approx(1.0, abs=1e-3)
    assert cov.condition_number(Zs) > 5        # per-stream stays ill-conditioned

    # and a specific cross-stream pair is nearly collinear
    Zn = (Zs - Zs.mean(0)) / Zs.std(0)
    corr = np.abs(np.corrcoef(Zn.T))
    np.fill_diagonal(corr, 0.0)
    st = np.array([n.split(":")[0] for n in ns])
    for stream in set(st):
        corr[np.ix_(st == stream, st == stream)] = 0.0
    assert corr.max() > 0.9


# ------------------------------------------- what the naive SDM actually estimates

def test_identifiability_measures_the_lambda_p_ridge():
    """Posterior corr(beta0, alpha0) near +-1 means lambda and p are not separable.

    This is the decisive test for WHY an N-mixture samples badly: near +-1 the
    ridge is intrinsic and no covariate rotation helps; modest means the design
    matrix was the problem.
    """
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    rng = np.random.default_rng(0)
    t = rng.normal(size=500)

    class _Ridged:
        samples = {"beta0": t, "alpha0": -t + 0.01 * rng.normal(size=500)}

    class _Clean:
        samples = {"beta0": rng.normal(size=500), "alpha0": rng.normal(size=500)}

    assert occupancy.identifiability(_Ridged())["corr(beta0,alpha0)"] < -0.9
    assert abs(occupancy.identifiability(_Clean())["corr(beta0,alpha0)"]) < 0.2


def test_detection_summary_flags_degenerate_estimates():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")

    class _R:
        samples = {"prob_detection": np.full(200, 0.45)}

    d = occupancy.detection_summary(_R())
    assert d["mean"] == pytest.approx(0.45) and not d["degenerate"]

    class _Deg:
        samples = {"prob_detection": np.full(200, 0.995)}

    assert occupancy.detection_summary(_Deg())["degenerate"] is True

    class _None:
        samples = {}

    assert occupancy.detection_summary(_None()) == {}


def test_sampler_diagnostics_threshold_follows_the_tree_depth():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")

    class _M:
        def get_extra_fields(self):
            return {"num_steps": np.array([1023, 1023, 7, 7])}

    at10 = occupancy.sampler_diagnostics(_M(), max_tree_depth=10)
    assert at10["frac_at_max_treedepth"] == pytest.approx(0.5)
    assert at10["max_treedepth_steps"] == 1023
    # raising the depth means those same runs are no longer at the limit
    at12 = occupancy.sampler_diagnostics(_M(), max_tree_depth=12)
    assert at12["frac_at_max_treedepth"] == 0.0


def test_compare_subcommand_is_registered():
    """The benchmark's primary stage: SDM designation vs the dynamic model."""
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_run_sdm8", repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    ap = mod.build_parser()
    a = ap.parse_args(["compare", "--run-dir", "/tmp/x"])
    assert a.run_dir == "/tmp/x" and a.lam_threshold == 1.0


def test_zero_probability_matches_the_latent_distribution():
    """P(N=0) is exp(-lambda) for Poisson but (phi/(phi+lambda))^phi for NB2.

    The latent was switched to negative binomial while the conversion still used
    the Poisson formula, which OVERSTATES occupancy -- badly at small phi. At
    lambda=8 the Poisson puts 0.0003 on zero and NB2(phi=1) puts 0.111.
    """
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    nbinom = pytest.importorskip("scipy.stats").nbinom
    poisson = pytest.importorskip("scipy.stats").poisson
    lam = np.array([[8.0], [8.0]])                 # 2 draws, 1 site

    class _P:
        samples = {"abundance": lam}

    class _NB:
        samples = {"abundance": lam, "concentration": np.array([1.0, 1.0])}

    got_p = occupancy.psi_from_nmixture(_P())[0]
    got_nb = occupancy.psi_from_nmixture(_NB())[0]
    assert got_p == pytest.approx(1 - poisson.pmf(0, 8.0), abs=1e-6)
    assert got_nb == pytest.approx(1 - nbinom.pmf(0, 1.0, 1.0 / (1.0 + 8.0)),
                                   abs=1e-6)
    assert got_nb < got_p - 0.05          # NB puts far more mass on zero


def test_nb_concentration_is_picked_up_automatically_from_the_fit():
    occupancy = pytest.importorskip("src.analysis.sdm_benchmark.occupancy",
                                    reason="needs numpyro/jax")
    lam = np.full((4, 3), 2.0)

    class _NB:
        samples = {"abundance": lam, "concentration": np.full(4, 1.5)}

    auto = occupancy.psi_from_nmixture(_NB())
    explicit = occupancy.psi_from_nmixture(
        _NB(), concentration=np.full(4, 1.5).reshape(4, 1))
    assert np.allclose(auto, explicit)
    assert np.all((auto >= 0) & (auto <= 1))


def test_compare_reports_the_threshold_free_quantities_first():
    """psi and mean lambda compare directly; thresholding psi destroys the result.

    A model reporting psi=0.83 in the Great Plains reads as "35% niche" once
    maxSSS is applied. The cut is a convention, so the headline is the
    un-thresholded pair and the contingency table is supporting detail.
    """
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "scripts" / "run_sdm_benchmark.py").read_text()
    # the threshold-free block must be built and printed before the 2x2
    i_free = src.index("OCCUPANCY vs SELF-SUSTAINING")
    i_thr = src.index("Designation comparison (maxSSS-thresholded")
    assert i_free < i_thr
    assert "occupancy_vs_demography" in src
    # and the un-thresholded surfaces must be retained, not discarded
    assert "continuous[tag] = grid" in src

    spec = importlib.util.spec_from_file_location("_run_sdm9",
                                                  repo / "scripts" / "run_sdm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")
    assert mod.build_parser().parse_args(["compare", "--run-dir", "/tmp/x"])


# --------------------------------------------------------------- PDF report

def test_report_textpage_wraps_instead_of_overflowing():
    """matplotlib's wrap=True does nothing for text in FIGURE coordinates.

    Long covariate descriptions ran off the page and collided at the bottom
    margin, so the page wraps to a character count itself.
    """
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "scripts" / "viz" / "sdm_benchmark_report.py").read_text()
    assert "import textwrap" in src
    assert "def _textpage" in src
    # the docstring MENTIONS wrap=True to explain why it was dropped, so match
    # the call form, not the bare string
    import re
    assert not re.search(r"fig\.text\([^)]*wrap\s*=\s*True", src, re.S)
    assert "textwrap.wrap(" in src


def test_report_draws_correlative_surfaces_unthresholded():
    """Binarising psi at maxSSS would manufacture the answer the report exists
    to show; the report must not do it."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "scripts" / "viz" / "sdm_benchmark_report.py").read_text()
    assert "threshold_max_sss" not in src
    assert "unthresholded" in src.lower() or "UNTHRESHOLDED" in src


def test_report_reuses_the_shared_geo_and_regrid_helpers():
    """_geo for the basemap/Great Plains outline, regrid for eBird's EPSG:8857
    raster -- both already existed and should not be reimplemented."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "scripts" / "viz" / "sdm_benchmark_report.py").read_text()
    assert "import _geo" in src and "GeoContext" in src
    assert "reproject_to_ref" in src
    assert "_validation_style" in src


def test_report_cli_parses():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_sdm_report", repo / "scripts" / "viz" / "sdm_benchmark_report.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"report module not importable here: {e}")
    assert set(mod.MODEL_NOTES) >= {"biolith_standard", "brt_standard"}
    for n in mod.MODEL_NOTES.values():
        assert {"kind", "estimates", "covars", "prep"} <= set(n)


def test_biolith_saves_do_not_clobber_each_other():
    """cmd_biolith writes the prediction file twice: after occu, then after
    nmixture. The second must EXTEND the first payload, not rebuild it.

    Rebuilding dropped grid_psi -- the occu save wrote the continental surface
    and the nmixture save overwrote the file without it, so a run that logged
    "grid surface: 17,209 of 17,209 land cells" still shipped a file with no
    grid in it.
    """
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "scripts" / "run_sdm_benchmark.py").read_text()
    body = src[src.index("def cmd_biolith"):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body

    saves = re.findall(r"np\.savez_compressed\((.*?)\)\n", body, re.S)
    assert len(saves) >= 2, "expected an occu save and an nmixture save"
    # every save of the prediction file must go through the shared payload dict
    for s in saves:
        if "_pred.npz" in s:
            assert "**pay" in s, f"save rebuilds the payload instead of extending it: {s[:120]}"


def test_grid_prediction_reuses_the_fitted_transforms():
    """The grid must be standardized with the SITE mean/sd, not its own.

    Refitting the transform on grid cells expresses the model's coefficients in
    a different basis: psi at SURVEYED cells came back 0.91-0.94 where the site
    predictions for those same cells were 0.82-0.83, and the zone ordering
    inverted. Same model, same cells, different answers.
    """
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    src = (repo / "scripts" / "run_sdm_benchmark.py").read_text()
    body = src[src.index("def _biolith_grid_psi"):]
    body = body[:body.index("\ndef ", 1)]
    # the grid standardization must pass the fitted mu/sd through
    assert re.search(r"standardize\(\s*Xg\s*,\s*mu\s*,\s*sd\s*\)", body), \
        "grid standardization refits on the grid instead of reusing site mu/sd"
    # and the PCA rotation likewise
    assert "basis=pca_info" in body and "center=pca_info" in body
    # the caller must supply them
    call = src[src.index("grid_psi = _biolith_grid_psi"):]
    assert "mu=mu" in call[:400] and "sd=sd" in call[:400]


def test_standardize_with_supplied_stats_is_not_refitted():
    """covariates.standardize(X, mu, sd) must apply, not re-estimate."""
    cov = pytest.importorskip("src.analysis.sdm_benchmark.covariates")
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 2.0, size=(200, 3))
    _, mu, sd = cov.standardize(train)
    # a shifted block must come out shifted, not re-centred on itself
    other = train + 10.0
    out, _, _ = cov.standardize(other, mu, sd)
    assert out.mean() > 3.0                      # retains the shift
    refit, _, _ = cov.standardize(other)
    assert abs(refit.mean()) < 0.1               # refitting would erase it


def test_every_subcommand_defines_the_args_its_handler_reads():
    """Guards the class of bug that killed all three BRT stages.

    --no-grid was added to the biolith subparser only, while cmd_brt read
    args.no_grid -- so the BRT died with AttributeError at runtime, on HPC,
    after the covariates had already been assembled. Parsers and handlers are
    edited in different places and drift silently; this checks them against
    each other.
    """
    import importlib.util
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "run_sdm_benchmark.py"
    spec = importlib.util.spec_from_file_location("_run_sdm_args", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"CLI not importable here: {e}")

    src = path.read_text()
    ap = mod.build_parser()
    subparsers = [a for a in ap._actions if hasattr(a, "choices") and a.choices
                  and all(hasattr(v, "parse_args") for v in a.choices.values())]
    assert subparsers, "no subparsers found"

    # attributes the CLI sets itself at runtime rather than through argparse
    runtime_only = {"fn", "_mcmc"}

    for name, sub in subparsers[0].choices.items():
        handler = {"brt": "cmd_brt", "biolith": "cmd_biolith",
                   "premise": "cmd_premise", "designation": "cmd_designation",
                   "compare": "cmd_compare", "preflight": "cmd_preflight"}.get(name)
        if handler is None or f"def {handler}" not in src:
            continue
        body = src[src.index(f"def {handler}"):]
        nxt = body.index("\ndef ", 1) if "\ndef " in body[1:] else len(body)
        body = body[:nxt]
        read = set(re.findall(r"\bargs\.([A-Za-z_][A-Za-z0-9_]*)", body)) - runtime_only
        defined = {a.dest for a in sub._actions}
        missing = sorted(read - defined)
        assert not missing, (
            f"{handler} reads args that `{name}` does not define: {missing}")


def test_every_structure_equation_renders_in_mathtext():
    """matplotlib's mathtext is not LaTeX: \\big and bmatrix both fail.

    Rendering each equation in isolation names the offender instead of failing
    the whole page with one traceback.
    """
    pytest.importorskip("matplotlib")
    import importlib.util
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_sdm_struct", repo / "scripts" / "viz" / "sdm_benchmark_report.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"report module not importable here: {e}")

    bad = []
    for title, _latent, eqs, _note in mod.STRUCTURE:
        for eq, gloss in eqs:
            for frag in (eq, gloss):
                fig = plt.figure()
                fig.text(0.1, 0.5, frag)
                try:
                    fig.canvas.draw()
                except Exception as e:
                    bad.append((title, frag[:50], str(e).strip().splitlines()[-1]))
                finally:
                    plt.close(fig)
    assert not bad, f"equations that do not render: {bad}"


def test_structure_notes_contain_no_mathtext():
    """Wrapped prose cannot hold math.

    _structure wraps the notes with textwrap, so a $...$ pair split across two
    lines renders as literal source. Equations keep their mathtext; notes use
    plain words.
    """
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_sdm_struct2", repo / "scripts" / "viz" / "sdm_benchmark_report.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"report module not importable here: {e}")
    offenders = [t for t, _l, _e, note in mod.STRUCTURE if "$" in note]
    assert not offenders, f"notes containing math that textwrap will break: {offenders}"


def test_structure_covers_every_model_family():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_sdm_struct3", repo / "scripts" / "viz" / "sdm_benchmark_report.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        pytest.skip(f"report module not importable here: {e}")
    titles = " ".join(t for t, _l, _e, _n in mod.STRUCTURE).lower()
    for family in ("boosted", "occupancy", "n-mixture", "ebird", "dynamic"):
        assert family in titles, f"no structure entry for {family}"
    # the latent state is the organising idea; every entry must declare one
    assert all(latent for _t, latent, _e, _n in mod.STRUCTURE)
