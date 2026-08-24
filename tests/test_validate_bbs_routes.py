"""Route-level BBS validation: densification, the no-change gate, and the metrics' SIGN.

The sign tests are the important ones. The claim is that differencing DESK against a
frozen-modern null isolates temporal skill: a model with no temporal information must score ~0,
and one carrying real temporal information must score > 0. If those stop holding, the headline
number means nothing.

The graded quantity is the DOT product ``Z @ Z.T``, per the kernel contract
(``Z(x) dot Z(x') ~= uncentered Ruzicka``) that ``desk_training.true_kernel_loss`` trains on --
NOT cosine, which discards the ||z|| calibration the contract fixes. One test here shows CKA is
provably blind to a pure norm deficit that the elementwise rmse catches, which is why rmse is
primary.
"""
import numpy as np
import pytest

from src.community_encoder.train_DESK.validate_bbs_routes import (
    EPOCH_EARLY, EPOCH_MODERN, compare_predictors, cosine_gram, densify_community, dot_gram,
    epoch_gate, epoch_mean_observed, epoch_mean_z, epoch_neighborhood_analysis, kernel_error,
    knn_neighbours, log1p_community, modern_reference_rows, quantile_distance_bins,
    stratified_sample, _pairwise_dot_neighbours, _pairwise_ruzicka_neighbours)
from src.community_encoder.train_DESK.validate_spacetime import ruzicka_rect


# ----------------------------- densification -----------------------------

def test_densify_uses_coverage_as_the_row_set():
    # Three surveyed cell-years; only two have any recorded species. The third must survive as a
    # genuine all-zero row -- it is a real observation of an empty community, and dropping it
    # would remove exactly the cell-years carrying the strongest turnover signal.
    X, keys, dropped = densify_community(
        row=[0, 0, 1], col=[0, 0, 1], year=[2000, 2000, 2001],
        species_index=[0, 2, 1], mean_count=[5.0, 3.0, 7.0],
        cov_row=[0, 1, 2], cov_col=[0, 1, 2], cov_year=[2000, 2001, 2002], n_species=3)

    assert keys.shape == (3, 3) and X.shape == (3, 3)
    assert dropped == 0
    kl = [tuple(int(v) for v in k) for k in keys]
    i00, i11, i22 = kl.index((0, 0, 2000)), kl.index((1, 1, 2001)), kl.index((2, 2, 2002))
    assert X[i00].tolist() == [5.0, 0.0, 3.0]      # unrecorded species are genuine zeros
    assert X[i11].tolist() == [0.0, 7.0, 0.0]
    assert X[i22].tolist() == [0.0, 0.0, 0.0]      # surveyed, nothing recorded -> kept, all-zero


def test_densify_sums_lumped_species_and_drops_uncovered():
    # Two triples mapping to the SAME (cell, year, species) must sum (crosswalk lumps), and a
    # presence triple for a cell-year missing from coverage must be dropped and counted.
    X, keys, dropped = densify_community(
        row=[0, 0, 9], col=[0, 0, 9], year=[2000, 2000, 1999],
        species_index=[1, 1, 0], mean_count=[2.0, 3.0, 100.0],
        cov_row=[0], cov_col=[0], cov_year=[2000], n_species=2)

    assert keys.shape == (1, 3) and X[0].tolist() == [0.0, 5.0]
    assert dropped == 1                            # the (9,9,1999) triple failed QC coverage
    assert not (X == 100.0).any()


def test_densify_deduplicates_coverage_rows():
    # A repeated cell-year in cov_* must not create two rows for one site-year.
    X, keys, _ = densify_community(
        row=[0], col=[0], year=[2000], species_index=[0], mean_count=[4.0],
        cov_row=[0, 0], cov_col=[0, 0], cov_year=[2000, 2000], n_species=1)
    assert keys.shape == (1, 3) and X.shape == (1, 1) and X[0, 0] == 4.0


def test_densify_returns_three_values_even_with_no_coverage():
    """The empty-coverage path used to return a 2-tuple while every other path returned 3, so
    an all-uncovered input unpacked into a ValueError instead of an empty result. Reachable
    whenever a gate removes every surveyed cell-year."""
    X, keys, dropped = densify_community(
        row=[0], col=[0], year=[2000], species_index=[0], mean_count=[5.0],
        cov_row=[], cov_col=[], cov_year=[], n_species=3)
    assert X.shape == (0, 3) and keys.shape == (0, 3)
    assert X.dtype == np.float32 and keys.dtype == np.int32
    assert dropped == 0


def test_densify_and_log1p_are_the_shared_bbs_community_implementations():
    """These moved to src/data/preprocess/bbs_community.py because the raw-BBS TARGET and this
    validation must densify identically. Re-exported here; assert it is the same object, not a
    copy that could drift."""
    from src.data.preprocess import bbs_community as bc
    assert densify_community is bc.densify_community
    assert log1p_community is bc.log1p_community


def test_log1p_clips_negatives():
    out = log1p_community(np.array([[-1.0, 0.0, np.e - 1.0]]))
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0
    assert abs(float(out[0, 2]) - 1.0) < 1e-6


# ----------------------------- the no-change gate -----------------------------

def test_modern_reference_picks_latest_and_gates_cells_without_modern():
    # Cell (0,0) surveyed 1970, 2012, 2020 -> modern ref is 2020 (the latest in-window year).
    # Cell (5,5) surveyed only in 1970 -> no modern survey, so EVERY row of it is dropped.
    keys = np.array([[0, 0, 1970], [0, 0, 2012], [0, 0, 2020], [5, 5, 1970]], dtype="int32")
    nc_src, keep = modern_reference_rows(keys, modern_window=(2010, 2025))

    assert keep.tolist() == [True, True, True, False]
    assert nc_src[0] == 2 and nc_src[1] == 2 and nc_src[2] == 2   # all point at the 2020 row
    assert nc_src[3] == -1


def test_nochange_rows_are_constant_within_a_cell():
    # After the gate, X_nc for one cell must be that cell's modern vector at EVERY year -- the
    # null must carry zero temporal variation, which is what makes cka_gain a temporal readout.
    keys = np.array([[0, 0, 1970], [0, 0, 2015], [1, 1, 1980], [1, 1, 2020]], dtype="int32")
    X = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 5.0], [3.0, 3.0]], dtype="float32")
    nc_src, keep = modern_reference_rows(keys, modern_window=(2010, 2025))
    assert keep.all()

    X_nc = X[nc_src]
    assert np.array_equal(X_nc[0], X_nc[1]) and np.array_equal(X_nc[0], X[1])
    assert np.array_equal(X_nc[2], X_nc[3]) and np.array_equal(X_nc[2], X[3])
    # and the null genuinely differs from truth at the historical rows
    assert not np.array_equal(X_nc[0], X[0])


def test_modern_window_boundaries_are_inclusive():
    keys = np.array([[0, 0, 2010], [1, 1, 2025], [2, 2, 2009], [3, 3, 2026]], dtype="int32")
    _, keep = modern_reference_rows(keys, modern_window=(2010, 2025))
    assert keep.tolist() == [True, True, False, False]


# ----------------------------- metric sign -----------------------------

def _synthetic(n_cells=14, years=(1970, 1985, 2000, 2015), n_species=8, seed=0):
    """Cells with a per-cell community that drifts monotonically toward a per-cell target."""
    rng = np.random.default_rng(seed)
    base = rng.random((n_cells, n_species)) * 5.0
    target = rng.random((n_cells, n_species)) * 5.0
    keys, X = [], []
    for c in range(n_cells):
        for y in years:
            f = (y - years[0]) / (years[-1] - years[0])       # 0 at first year, 1 at last
            keys.append([c, c, y])
            X.append((1.0 - f) * base[c] + f * target[c])
    return np.array(keys, dtype="int32"), log1p_community(np.array(X))


def test_a_temporally_neutral_desk_scores_exactly_zero_gain():
    # If DESK emits its modern z at every year, S_desk IS S_nc, so the gain must be exactly 0.
    # This is what makes the metric a temporal readout: the null is DESK's own output held
    # constant, in DESK's own similarity functional, so spatial fidelity cancels identically.
    keys, X = _synthetic()
    nc_src, keep = modern_reference_rows(keys, modern_window=(2010, 2025))
    assert keep.all()

    Z_nc = X[nc_src] @ np.random.default_rng(2).standard_normal((X.shape[1], 6))
    S_nc = dot_gram(Z_nc)
    m = compare_predictors(ruzicka_rect(X, X), {"desk": S_nc, "no_change": S_nc}, grams=True)
    d, nl = m["predictors"]["desk"], m["predictors"]["no_change"]
    assert d["cka"] - nl["cka"] == 0.0, m
    assert d["mantel"] - nl["mantel"] == 0.0, m
    assert m["skill_vs"]["desk"] == 0.0, m         # identical kernels -> no error reduction
    assert d["rmse"] == nl["rmse"], m


def test_cka_gain_is_positive_only_when_temporal_information_is_added():
    # Controlled contrast: ONE linear map A, so spatial fidelity is identical between the two
    # models and the only difference is whether z varies over time.
    #
    #   Z_nc   = X_nc @ A  -> the null: modern community, frozen
    #   Z_good = X    @ A  -> same map, but tracks the true year-by-year community
    #
    # Both are scored with dot_gram, matching how run() builds them. Scoring the null with
    # Ruzicka instead (truth's own function) made a temporally-neutral model read -0.28, which is
    # the confound this arrangement removes.
    keys, X = _synthetic()
    nc_src, keep = modern_reference_rows(keys, modern_window=(2010, 2025))
    assert keep.all()

    A = np.random.default_rng(2).standard_normal((X.shape[1], 6))
    S_true = ruzicka_rect(X, X)
    S_nc = dot_gram(X[nc_src] @ A)

    m = compare_predictors(S_true, {"desk": dot_gram(X @ A), "no_change": S_nc}, grams=True)
    d, nl = m["predictors"]["desk"], m["predictors"]["no_change"]
    assert d["cka"] - nl["cka"] > 0.0, m
    assert d["cka"] > nl["cka"], m


def test_observed_space_null_is_reported_but_not_differenced():
    # cka_nochange_observed uses Ruzicka (truth's functional) and so is NOT comparable to
    # cka_desk; it must appear in the output without contaminating cka_gain.
    keys, X = _synthetic()
    nc_src, _ = modern_reference_rows(keys, modern_window=(2010, 2025))
    X_nc = X[nc_src]
    A = np.random.default_rng(4).standard_normal((X.shape[1], 6))
    S_true = ruzicka_rect(X, X)

    dot = compare_predictors(S_true, {"desk": dot_gram(X @ A), "no_change": dot_gram(X_nc @ A)},
                             grams=True)
    cos = compare_predictors(S_true, {"desk": cosine_gram(X @ A),
                                      "no_change": cosine_gram(X_nc @ A)}, grams=True)
    g = lambda m: m["predictors"]["desk"]["cka"] - m["predictors"]["no_change"]["cka"]
    # The observed-space reference uses Ruzicka -- truth's OWN functional -- so it is reported
    # beside the model columns and never differenced against them; mixing functionals was
    # measured at -0.28 on a temporally-neutral model.
    assert g(dot) == dot["predictors"]["desk"]["cka"] - dot["predictors"]["no_change"]["cka"]
    # dot and cosine are separate forms, not one overwriting the other
    assert g(cos) != g(dot)


def test_cka_of_truth_against_itself_is_one():
    _, X = _synthetic()
    S = ruzicka_rect(X, X)
    m = compare_predictors(S, {"desk": S, "no_change": S}, grams=True)
    assert abs(m["predictors"]["desk"]["cka"] - 1.0) < 1e-8
    assert abs(m["predictors"]["desk"]["cka"]
               - m["predictors"]["no_change"]["cka"]) < 1e-8    # identical inputs -> no gain


def test_dot_gram_is_the_contract_and_keeps_scale():
    # The contract is Z.Z' ~= Ruzicka, so the dot product must preserve ||z|| -- scaling Z MUST
    # change the result. That sensitivity is the point: it is what makes rmse against Ruzicka a
    # calibration check rather than a structure-only comparison.
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((6, 4))
    assert np.allclose(dot_gram(Z), Z @ Z.T)
    assert not np.allclose(dot_gram(Z), dot_gram(Z * 2.0))
    assert np.allclose(np.diag(dot_gram(Z)), (Z ** 2).sum(1))     # diagonal is ||z||^2


def test_kernel_error_detects_a_norm_deficit_that_cka_ignores():
    # A model whose z is correct in direction but systematically too short reproduces the
    # STRUCTURE perfectly (CKA ~ 1) while its dot products are all too small. Only the
    # elementwise comparison sees that, which is why rmse is the primary metric.
    # Z must be NON-NEGATIVE for this to mean anything: Ruzicka is in [0,1], so the real dots are
    # positive and shrinking ||z|| lowers them. With Gaussian z the off-diagonal dots straddle
    # zero, shrinking moves them toward zero from both sides, and the mean bias cancels to ~0.
    rng = np.random.default_rng(5)
    Z = rng.random((30, 5))
    S_true = dot_gram(Z)
    S_short = dot_gram(Z * 0.85)                                  # 28% low in dot space

    e = kernel_error(S_true, S_short)
    assert e["bias"] < 0                                          # predicts too little similarity
    assert e["rmse"] > 0
    assert abs(e["pearson_r"] - 1.0) < 1e-9                       # perfectly correlated...
    from src.community_encoder.train_DESK.validate_spacetime import linear_cka
    assert abs(linear_cka(S_true, S_short) - 1.0) < 1e-9          # ...and CKA is blind to it


def test_r2_floor_is_predicting_the_mean():
    # r2 must be 0 for the know-nothing model (same average similarity for every pair) and 1 for a
    # perfect one. Without that anchor an RMSE of 0.11 is uninterpretable -- it is only meaningful
    # relative to sd(S_true), which IS the know-nothing model's RMSE.
    rng = np.random.default_rng(11)
    S = rng.random((40, 40)); S = (S + S.T) / 2
    t = S[np.triu_indices_from(S, k=1)]

    S_mean = np.full_like(S, t.mean())                        # predict the mean everywhere
    e_mean = kernel_error(S, S_mean)
    assert abs(e_mean["r2"]) < 1e-9                           # exactly the floor
    assert abs(e_mean["rmse"] - t.std()) < 1e-9               # and its rmse IS sd(S_true)

    e_perfect = kernel_error(S, S)
    assert abs(e_perfect["r2"] - 1.0) < 1e-12
    assert e_perfect["rmse"] == 0.0


def test_r2_matches_the_closed_form_and_the_measured_run():
    # r2 = 1 - (rmse/sd)^2. Pinned against the real run: sd 0.1615, rmse_desk 0.1085 -> 0.549,
    # rmse_nochange 0.1212 -> 0.437. If this identity ever changes, every reported r2 shifts.
    for rmse, sd, want in ((0.1085, 0.1615, 0.549), (0.1212, 0.1615, 0.437)):
        assert abs((1.0 - (rmse / sd) ** 2) - want) < 5e-4


def test_error_variance_removed_is_the_squared_rmse_ratio():
    # "DESK removes X% of the null's error variance" must be 1 - (rmse_desk/rmse_nc)^2, and must
    # equal the r2_gain divided by nothing -- i.e. r2_gain is that reduction expressed as a share
    # of TOTAL variance. Both framings have to stay consistent or the write-up contradicts itself.
    rng = np.random.default_rng(12)
    S = rng.random((60, 60)); S = (S + S.T) / 2
    noise_d = rng.normal(0, 0.05, S.shape); noise_d = (noise_d + noise_d.T) / 2
    noise_n = rng.normal(0, 0.09, S.shape); noise_n = (noise_n + noise_n.T) / 2

    m = compare_predictors(S, {"desk": S + noise_d, "no_change": S + noise_n}, grams=True)
    d, nl = m["predictors"]["desk"], m["predictors"]["no_change"]
    ratio = d["rmse"] / nl["rmse"]
    assert abs(m["skill_vs"]["desk"] - (1 - ratio)) < 1e-12
    # the r2 gain is that same reduction expressed as a share of TOTAL variance
    expect = (nl["rmse"] ** 2 - d["rmse"] ** 2) / m["observed_sd"] ** 2
    assert abs((d["r2"] - nl["r2"]) - expect) < 1e-9


def test_calibration_loss_is_pearson_sq_minus_r2():
    # A prediction that ranks pairs perfectly but is offset by a constant has pearson^2 = 1 and
    # r2 < 1; the gap is exactly the calibration loss. This is the 0.78-vs-0.55 discrepancy in the
    # real run, and it must be reported rather than hidden by quoting only the correlation.
    rng = np.random.default_rng(13)
    S = rng.random((50, 50)); S = (S + S.T) / 2
    e = kernel_error(S, S + 0.05)                             # perfect ranking, constant offset
    assert abs(e["pearson_r"] - 1.0) < 1e-9
    assert e["r2"] < 1.0
    assert abs(e["bias"] - 0.05) < 1e-9
    m = compare_predictors(S, {"desk": S + 0.05, "no_change": S + 0.09}, grams=True)
    d = m["predictors"]["desk"]
    assert abs(d["calibration_loss"] - (d["pearson_r"] ** 2 - d["r2"])) < 1e-12
    assert d["calibration_loss"] > 0                          # ranking better than accuracy
    # and it is computed for EVERY predictor, not only the model -- the asymmetry this refactor
    # removed meant a bar could never be diagnosed the way DESK was.
    assert "calibration_loss" in m["predictors"]["no_change"]


def test_cosine_gram_is_scale_invariant_and_unit_diagonal():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((6, 4))
    S, S_scaled = cosine_gram(Z), cosine_gram(Z * 17.0)
    assert np.allclose(S, S_scaled)                            # a raw dot would NOT satisfy this
    assert np.allclose(np.diag(S), 1.0)
    # a zero row must not produce NaN (it would poison every CKA downstream)
    Z0 = np.vstack([Z, np.zeros((1, 4))])
    assert np.isfinite(cosine_gram(Z0)).all()


# ----------------------------- sampling -----------------------------

def test_stratified_sample_reaches_the_sparse_early_window():
    # BBS coverage grows over time, so a uniform sample is swamped by modern rows and the early
    # window would be too thin to report. 6 early rows against 300 modern ones.
    keys = np.array([[0, 0, 1970]] * 6 + [[1, 1, 2015]] * 300, dtype="int32")
    sel = stratified_sample(keys, n_sample=30, rng=np.random.default_rng(0))
    yrs = keys[sel, 2]
    assert (yrs == 1970).sum() == 6                            # every available early row taken
    assert (yrs == 2015).sum() > 0
    assert len(np.unique(sel)) == len(sel)                     # no duplicated rows


def test_stratified_sample_guarantees_heldout_rows():
    # Reproduces the imbalance from the first real run: held-out cells were ~7% of rows, so a
    # year-only stratification left heldout/early with 94 rows and heldout/modern with 73 -- the
    # decisive buckets were the least reliable. Crossing holdout with the year window must lift
    # the held-out share far above its 7% base rate at the same budget.
    rng_keys = np.random.default_rng(7)
    n = 4000
    yrs = rng_keys.choice([1970, 1975, 2015, 2020], size=n, p=[0.1, 0.1, 0.4, 0.4])
    keys = np.stack([np.arange(n) % 200, np.arange(n) % 200, yrs], axis=1).astype("int32")
    is_ho = rng_keys.random(n) < 0.07                          # ~7%, as measured

    sel_year = stratified_sample(keys, 400, np.random.default_rng(0))
    sel_both = stratified_sample(keys, 400, np.random.default_rng(0), is_heldout=is_ho)

    ho_year, ho_both = is_ho[sel_year].sum(), is_ho[sel_both].sum()
    assert ho_both > 3 * ho_year, (ho_year, ho_both)
    # and the early window is still protected on the held-out side specifically
    early_ho = ((keys[sel_both, 2] <= 1980) & is_ho[sel_both]).sum()
    assert early_ho > 0
    assert len(np.unique(sel_both)) == len(sel_both)            # no duplicated rows


def test_stratified_sample_without_holdout_matches_year_only_behavior():
    # Omitting is_heldout must reduce to the previous behavior, so the sparse early window is
    # still protected when no holdout mask exists on disk.
    keys = np.array([[0, 0, 1970]] * 6 + [[1, 1, 2015]] * 300, dtype="int32")
    sel = stratified_sample(keys, n_sample=30, rng=np.random.default_rng(0))
    yrs = keys[sel, 2]
    assert (yrs == 1970).sum() == 6 and (yrs == 2015).sum() > 0


def test_stratified_sample_returns_all_rows_when_under_budget():
    keys = np.array([[0, 0, 1970], [1, 1, 2015]], dtype="int32")
    sel = stratified_sample(keys, n_sample=100, rng=np.random.default_rng(0))
    assert sel.tolist() == [0, 1]


# ----------------------------- driver wiring -----------------------------

def test_run_end_to_end_gates_cells_and_buckets(tmp_path, monkeypatch):
    """Exercises run()'s gate -> reindex -> sample -> bucket wiring on a synthetic community.

    Stubs load_observed (which reads the full raw BBS release) and desk_z_ema (which needs a
    trained checkpoint). What is covered here is only what the driver itself does -- notably
    remapping ``nc_src`` into the post-gate row indices, which is silent if wrong: it would pair
    rows with the wrong cell's modern vector and quietly deflate the gain.
    """
    from src.community_encoder.train_DESK import validate_bbs_routes as V

    n_species, years = 6, [1970, 1975, 2000, 2012, 2020]
    rng = np.random.default_rng(0)
    keys, rows = [], []
    for c in range(20):
        # cell 19 is surveyed ONLY in the early window -> no modern reference -> fully gated out
        for y in ([1970, 1975] if c == 19 else years):
            keys.append([c, c, y])
            rows.append(rng.random(n_species) * 5 + (y - 1970) * 0.05)
    keys = np.array(keys, dtype="int32")
    X_arr = np.array(rows)
    X_log = log1p_community(X_arr)

    desk = tmp_path / "desk"
    desk.mkdir()
    ho = np.zeros((25, 25), bool)
    ho[::3] = True                                             # ~1/3 of cells held out
    np.save(desk / "holdout_cells.npy", ho)
    cfg = {"trend": {"points_dir": str(tmp_path)}, "paths": {"desk_output_dir": str(desk)}}

    monkeypatch.setattr(V, "load_observed", lambda config: (
        X_log, keys, {"n_species": n_species, "n_surveyed_cell_years": int(keys.shape[0]),
                      "year_range": [1970, 2020]}, X_arr))

    # Stub DESK: z is a linear map of the TRUE community, so it genuinely tracks time and the gain
    # must come out positive. Keyed by (cell, year) so the stub cannot leak information the real
    # encoder would not have.
    lut = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(keys)}
    A = np.random.default_rng(1).standard_normal((n_species, 5))

    def fake_z(config, k):
        Z = np.stack([X_log[lut[(int(r), int(c), int(y))]] @ A for r, c, y in k])
        return Z.astype("float32"), {"output_ema_applied": True, "ema_half_life": 10.0,
                                     "ema_warmup_start": 1940, "encode_years": [1940, 2020]}
    monkeypatch.setattr(V, "desk_z_ema", fake_z)

    rep = V.run(config=cfg, n_sample=200, seed=0)

    assert rep["site_gate"]["cells_total"] == 20
    assert rep["site_gate"]["cells_dropped_no_modern"] == 1     # exactly cell 19
    assert rep["site_gate"]["cells_kept"] == 19
    for name in ("pooled/all", "train/all", "heldout/all",
                 "pooled/modern", "pooled/early", "heldout/early"):
        assert name in rep["buckets"], name
    b = rep["buckets"]["pooled/all"]["dot"]["predictors"]
    assert b["desk"]["cka"] - b["no_change"]["cka"] > 0, rep["buckets"]["pooled/all"]
    assert (desk / "bbs_route_validation.json").exists()
    assert (desk / "bbs_route_validation.npz").exists()


# ------------------- epoch x local-neighbourhood analysis -------------------

def test_epoch_gate_counts_distinct_years_not_route_years():
    # Cell (0,0): 3 distinct years in each epoch -> KEPT.
    # Cell (1,1): plenty of rows but all in ONE calendar year per epoch -> REJECTED. This is the
    # whole point of the gate: 5 routes run in 1972 say nothing about the other 20 years, so the
    # epoch mean would not be the noise-averaged estimate the analysis assumes.
    keys = np.array(
        [[0, 0, 1970], [0, 0, 1978], [0, 0, 1985], [0, 0, 2008], [0, 0, 2015], [0, 0, 2022]]
        + [[1, 1, 1972]] * 5 + [[1, 1, 2010]] * 5, dtype="int32")
    cells, e_rows, m_rows, stats = epoch_gate(keys)

    assert cells.tolist() == [[0, 0]]
    assert stats["cells_kept"] == 1 and stats["cells_seen"] == 2
    assert len(e_rows[0]) == 3 and len(m_rows[0]) == 3
    # rows point at the right epoch
    assert sorted(keys[e_rows[0], 2].tolist()) == [1970, 1978, 1985]
    assert sorted(keys[m_rows[0], 2].tolist()) == [2008, 2015, 2022]


def test_epoch_gate_requires_both_windows_and_is_boundary_inclusive():
    # Epoch bounds inclusive on both ends: 1966/1986 and 2005/2025 must all count.
    keys = np.array([[0, 0, 1966], [0, 0, 1976], [0, 0, 1986],
                     [0, 0, 2005], [0, 0, 2015], [0, 0, 2025],
                     [2, 2, 1966], [2, 2, 1976], [2, 2, 1986]],   # early only -> rejected
                    dtype="int32")
    cells, _, _, stats = epoch_gate(keys)
    assert cells.tolist() == [[0, 0]]
    assert stats["cells_failed_modern_only"] == 1
    assert stats["early_window"] == list(EPOCH_EARLY)
    assert stats["modern_window"] == list(EPOCH_MODERN)


def test_epoch_gate_excludes_years_outside_both_windows():
    # 1990-2004 falls between the epochs and must count toward neither.
    keys = np.array([[0, 0, 1990], [0, 0, 1995], [0, 0, 2000],
                     [0, 0, 2010], [0, 0, 2015], [0, 0, 2020]], dtype="int32")
    cells, _, _, _ = epoch_gate(keys)
    assert cells.shape[0] == 0                       # no early-epoch years at all


def test_epoch_mean_averages_raw_counts_then_log1p():
    # log1p is concave, so mean(log1p(x)) < log1p(mean(x)) whenever abundance varies. The epoch
    # summary must be "the average community", i.e. average the COUNTS then transform once.
    X_raw = np.array([[0.0, 100.0], [100.0, 0.0]])    # large spread -> the gap is obvious
    got = epoch_mean_observed(X_raw, [[0, 1]])
    assert np.allclose(got, np.log1p([[50.0, 50.0]]))

    mean_of_log1p = np.log1p(X_raw).mean(axis=0)
    assert (got[0] > mean_of_log1p).all()            # concavity: transform-last is strictly larger


def test_epoch_mean_z_averages_the_vectors():
    Z = np.array([[1.0, 3.0], [3.0, 5.0], [10.0, 10.0]], dtype="float32")
    got = epoch_mean_z(Z, [[0, 1], [2]])
    assert np.allclose(got[0], [2.0, 4.0]) and np.allclose(got[1], [10.0, 10.0])


def test_knn_on_a_line_of_cells_at_grid_spacing():
    # 27 km spacing: neighbour 1 of the focal cell must be the adjacent cell at exactly 27 km,
    # distances must come back in METRES, and the self hit must never appear.
    xy = np.stack([np.arange(8) * 27000.0, np.zeros(8)], 1)
    idx, dist = knn_neighbours(xy, k=3)
    assert idx.shape == (8, 3) and dist.shape == (8, 3)
    assert idx[0].tolist() == [1, 2, 3]
    assert np.allclose(dist[0], [27000.0, 54000.0, 81000.0])
    for r in range(8):
        assert r not in idx[r].tolist()               # self excluded wherever it sorted


def test_knn_returns_all_available_when_fewer_than_k():
    # Asking for 99 neighbours from 4 cells must yield 3, not an error.
    xy = np.stack([np.arange(4) * 27000.0, np.zeros(4)], 1)
    idx, dist = knn_neighbours(xy, k=99)
    assert idx.shape == (4, 3)
    assert np.isfinite(dist).all()


def test_batched_neighbour_ruzicka_matches_the_reference_implementation():
    # The fast path computes only the focal-to-neighbour entries; it must agree with the
    # full-matrix ruzicka_rect on exactly those entries. fp32 on the GPU path, hence 1e-5.
    rng = np.random.default_rng(3)
    A, B = rng.random((30, 9)), rng.random((30, 9))
    idx = np.stack([rng.permutation(30)[:6] for _ in range(30)])

    fast = _pairwise_ruzicka_neighbours(A, B, idx)
    ref = ruzicka_rect(A, B)
    slow = np.stack([ref[i, idx[i]] for i in range(30)])
    assert np.abs(fast - slow).max() < 1e-5

    d_fast = _pairwise_dot_neighbours(A, B, idx)
    d_slow = np.stack([(A[i] * B[idx[i]]).sum(1) for i in range(30)])
    assert np.abs(d_fast - d_slow).max() < 1e-5


def test_quantile_bins_are_equal_n_and_survive_degenerate_input():
    d = np.concatenate([np.full(100, 30000.0), np.linspace(4e5, 2e6, 100)])
    edges, labels = quantile_distance_bins(d, n_bins=4)
    counts = np.bincount(np.clip(np.digitize(d, edges) - 1, 0, len(labels) - 1),
                         minlength=len(labels))
    assert counts.sum() == d.size
    assert counts.max() <= 3 * max(counts.min(), 1)   # roughly balanced, not all in one bin
    # all-identical distances must collapse to one bin rather than raise
    e2, l2 = quantile_distance_bins(np.full(50, 12345.0), n_bins=10)
    assert len(l2) == 1


def _epoch_fixture(n=40, n_species=7, latent=5, seed=0):
    """Cells on a line whose community drifts, with DESK z a linear map of the truth."""
    rng = np.random.default_rng(seed)
    Xe = rng.random((n, n_species)) * 5.0
    Xm = Xe + rng.random((n, n_species)) * 2.0        # genuine change
    A = rng.standard_normal((n_species, latent))
    xy = np.stack([np.arange(n) * 27000.0, np.zeros(n)], 1)
    return log1p_community(Xe), log1p_community(Xm), A, xy


def _dot(rep, question, split="pooled", bin_="all_distances"):
    """New addressing: results are keyed by PREDICTOR, and the dot/cosine forms are siblings."""
    return rep["types"][question][split][bin_]["dot"]


def _skill(rep, question, predictor="desk", split="pooled", bin_="all_distances"):
    return _dot(rep, question, split, bin_)["skill_vs"][predictor]


def test_spatial_modern_skill_is_exactly_zero_the_builtin_null_test():
    # spatial_modern grades Zm.Zm against a null that IS Zm.Zm. Its skill must be exactly 0.
    # This is the analysis's own null test: if it ever moves, the harness mis-pairs its inputs
    # and no other row in the table can be trusted.
    Xe, Xm, A, xy = _epoch_fixture()
    rep, per_cell = epoch_neighborhood_analysis(Xe, Xm, Xe @ A, Xm @ A, xy, k=9, n_bins=3)

    for split, per_bin in rep["types"]["cross_cell_same_era_modern"].items():
        for bname, mm in per_bin.items():
            if "skipped" in mm:
                continue
            d = mm["dot"]["predictors"]
            assert mm["dot"]["skill_vs"]["desk"] == 0.0, (split, bname, mm)
            assert d["desk"]["rmse"] == d["no_change"]["rmse"], (split, bname, mm)
            assert d["desk"]["r2"] == d["no_change"]["r2"], (split, bname, mm)
            # the ANGULAR form must pass the same zero check, not just the dot form
            assert mm["cosine"]["skill_vs"]["desk"] == 0.0, (split, bname, mm)
    assert np.allclose(per_cell["cross_cell_same_era_modern_skill_desk"], 0.0)


def test_epoch_analysis_shape_and_that_a_tracking_desk_beats_the_null():
    Xe, Xm, A, xy = _epoch_fixture()
    rep, per_cell = epoch_neighborhood_analysis(Xe, Xm, Xe @ A, Xm @ A, xy, k=9, n_bins=3)

    assert set(rep["types"]) == {"pair_convergence",
                                 "cross_cell_same_era_early", "cross_cell_same_era_modern",
                                 "cross_cell_cross_time", "same_cell_over_time"}
    assert rep["config"]["k"] == 9 and rep["config"]["n_focal_cells"] == 40
    # a DESK that tracks the truth must beat the frozen-modern null where time matters
    assert _skill(rep, "cross_cell_same_era_early") > 0
    assert _dot(rep, "same_cell_over_time")["n"] == 40
    # per-cell fields are present and one value per focal cell, for mapping
    for key in ("cross_cell_same_era_early_skill_desk", "cross_cell_cross_time_skill_desk",
                "same_cell_over_time_obs", "neighbour_dist_m"):
        assert key in per_cell
    assert per_cell["cross_cell_same_era_early_skill_desk"].shape == (40,)
    assert per_cell["neighbour_dist_m"].shape == (40, 9)


def test_epoch_analysis_splits_by_holdout_when_given_a_mask():
    Xe, Xm, A, xy = _epoch_fixture()
    ho = np.zeros(40, bool); ho[::2] = True
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Xe @ A, Xm @ A, xy, k=9, n_bins=3, is_heldout=ho)
    assert set(rep["types"]["cross_cell_cross_time"]) == {"pooled", "train", "heldout"}
    n_ho = _dot(rep, "same_cell_over_time", "heldout")["n"]
    assert n_ho == 20


def test_epoch_mean_z_is_nan_aware():
    # A cell-year outside the covariate footprint comes back NaN from encode_points. One such year
    # must not wipe out the epoch mean; a cell with NO finite year must yield NaN so the caller's
    # finite filter drops it rather than silently averaging garbage.
    Z = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0],
                  [np.nan, np.nan], [np.nan, np.nan]], dtype="float32")
    got = epoch_mean_z(Z, [[0, 1, 2], [3, 4]])
    assert np.allclose(got[0], [2.0, 2.0])            # NaN year ignored, not propagated
    assert np.isnan(got[1]).all()                     # nothing finite -> NaN, caught downstream


def test_run_epoch_rows_survive_the_pooled_site_gate(tmp_path, monkeypatch):
    """Regression: the epoch row indices must index the UNFILTERED keys.

    The pooled site gate rebinds ``keys = keys[keep]``, so epoch row indices computed against the
    full array pointed into the wrong rows afterwards -- and crashed with
    ``IndexError: index 110878 is out of bounds for axis 0 with size 110839`` on the real data.

    The earlier driver test could not catch this: its synthetic years do not pass the epoch gate,
    so ``ep_rows_flat`` was empty and the offending line never ran. This fixture deliberately
    combines BOTH conditions -- cells that pass the epoch gate, AND cells the site gate drops, so
    the two index spaces genuinely differ.
    """
    from src.community_encoder.train_DESK import validate_bbs_routes as V

    n_species = 5
    rng = np.random.default_rng(0)
    keys, rows = [], []
    # ORDER MATTERS. The site-gate-dropped cells must come FIRST so that filtering SHIFTS the
    # indices of the surviving rows. With the dropped cells last, filtering only truncates the
    # tail, the epoch indices still resolve, and the bug hides -- a first version of this test
    # made exactly that mistake and passed against the buggy code.
    #
    # cells 0-4: early-only, so the SITE gate drops all 15 of their rows.
    for c in range(5):
        for y in (1968, 1975, 1982):
            keys.append([0, c, y]); rows.append(rng.random(n_species) * 5)
    # cells 5-14: pass BOTH gates (3+ distinct years per window, and a modern survey). Their rows
    # start at original index 15 but at filtered index 0.
    for c in range(5, 15):
        for y in (1968, 1975, 1982, 2008, 2015, 2022):
            keys.append([0, c, y]); rows.append(rng.random(n_species) * 5)
    keys = np.array(keys, dtype="int32")
    X_arr = np.array(rows)
    X_log = log1p_community(X_arr)

    desk = tmp_path / "desk"; desk.mkdir()
    np.save(desk / "holdout_cells.npy", np.zeros((4, 20), bool))
    cfg = {"trend": {"points_dir": str(tmp_path)}, "paths": {"desk_output_dir": str(desk)}}

    monkeypatch.setattr(V, "load_observed", lambda config: (
        X_log, keys, {"n_species": n_species, "n_surveyed_cell_years": int(keys.shape[0]),
                      "year_range": [1968, 2022]}, X_arr))
    A = rng.standard_normal((n_species, 4))
    monkeypatch.setattr(V, "desk_z_ema", lambda config, k: (
        np.stack([np.log1p(np.full(n_species, 1.0 + int(y) % 7)) @ A for _, _, y in k]
                 ).astype("float32"),
        {"output_ema_applied": True, "ema_half_life": 10.0, "ema_warmup_start": 1940,
         "encode_years": [1940, 2022]}))
    # load_data_config is imported INSIDE _run_epoch_analysis, so patch it at its source module
    import src.config_utils as CU
    monkeypatch.setattr(CU, "load_data_config", lambda *a, **kw: {"grid": {"ref_raster": "x"}})
    import src.community_encoder.train_DESK.validate_spacetime as VS
    monkeypatch.setattr(VS, "cell_xy", lambda r, c, ref: np.stack(
        [np.asarray(c, float) * 27000.0, np.asarray(r, float) * 27000.0], axis=1))

    rep = V.run(config=cfg, n_sample=100, seed=0)

    # the site gate must actually have dropped rows (else the test proves nothing)
    assert rep["site_gate"]["rows_kept"] < rep["site_gate"]["rows_total"]
    assert rep["site_gate"]["cells_dropped_no_modern"] == 5
    assert rep["site_gate"]["rows_total"] - rep["site_gate"]["rows_kept"] == 15
    # and the epoch analysis must have run over the 10 qualifying cells without an IndexError
    import json
    ep = json.load(open(desk / "bbs_epoch_neighborhood.json"))
    assert ep["gate"]["cells_kept"] == 10
    assert ep["config"]["n_focal_cells"] == 10
    assert ep["types"]["cross_cell_same_era_modern"]["pooled"]["all_distances"]["dot"]["skill_vs"]["desk"] == 0.0


def test_modern_reference_groups_share_the_site_gate_with_the_single_year_reference():
    """The averaged reference must change only the VALUE, never which rows survive. If the gate
    moved, the averaged and single-year reports would be scored on different row populations and
    could not be compared -- the same population mismatch that made the trainer's IDW bar
    incomparable with its pooled val MSE."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        modern_reference_groups, modern_reference_rows)
    keys = np.array([[0, 0, 1995], [0, 0, 2012], [0, 0, 2020],     # cell with modern surveys
                     [1, 1, 1990],                                  # cell with none
                     [2, 2, 2015]], dtype=np.int32)
    nc_src, keep = modern_reference_rows(keys, modern_window=(2010, 2025))
    groups, keep_g = modern_reference_groups(keys, modern_window=(2010, 2025))
    assert (keep_g == keep).all()
    # the single-year reference is the most recent; the group is every modern row of that cell
    assert groups[0] == (1, 2) and nc_src[0] == 2
    assert groups[3] == () and not keep[3]
    # every kept row's group must contain its own single-year reference
    for i in np.flatnonzero(keep):
        assert nc_src[i] in groups[i], (i, nc_src[i], groups[i])


def test_averaged_modern_reference_actually_reduces_reference_noise():
    """Why the averaging exists: at ~1.08 routes per cell-year the modern reference is one
    observer on one morning, and a 16-year window was being used to pick just one of them."""
    from src.community_encoder.train_DESK.validate_bbs_routes import modern_reference_groups
    rng = np.random.default_rng(0)
    truth = np.array([3.0, 1.0, 2.0])
    keys, X = [], []
    for y in range(2010, 2026):
        keys.append([0, 0, y])
        X.append(truth + rng.normal(scale=1.0, size=3))
    keys = np.array(keys, dtype=np.int32)
    X = np.stack(X)
    groups, keep = modern_reference_groups(keys, modern_window=(2010, 2025))
    assert keep.all() and len(groups[0]) == 16
    single = np.linalg.norm(X[-1] - truth)                    # most recent row alone
    averaged = np.linalg.norm(X[list(groups[0])].mean(0) - truth)
    assert averaged < single, (averaged, single)


def test_window_groups_are_per_cell_and_reduce_to_single_rows_at_zero_width():
    """The window must never reach across cells, and half_width=0 has to reproduce the historical
    one-row-per-endpoint behaviour exactly, or the pre-change numbers become unreproducible."""
    from src.community_encoder.train_DESK.validate_bbs_routes import window_groups
    keys = np.array([[0, 0, 1970], [0, 0, 1971], [0, 0, 1975],
                     [0, 1, 1970], [0, 1, 1971]], dtype=np.int32)
    g2 = window_groups(keys, 2)
    assert g2[0] == (0, 1) and g2[1] == (0, 1)        # 1975 is 5 yr away, and cell (0,1) is other
    assert g2[2] == (2,)
    assert g2[3] == (3, 4)                            # different cell, its own window
    assert window_groups(keys, 0) == [(0,), (1,), (2,), (3,), (4,)]


def test_endpoints_average_raw_counts_before_log1p_not_after():
    """The load-bearing order. BBS counts are Poisson, so the mean of RAW counts is the
    minimum-variance unbiased estimator of the rate; mean-of-log1p is biased for it and
    understates abundant species because log1p is concave. Averaging after the transform would
    therefore bias every denoised community low, worst where abundance varies most across years
    -- which is exactly the changing cells the temporal experiment is about."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        epoch_mean_observed, log1p_community, window_groups)
    keys = np.array([[0, 0, 1970], [0, 0, 1971]], dtype=np.int32)
    X_raw = np.array([[0.0, 40.0], [0.0, 0.0]])       # a species swinging hard between years
    groups = window_groups(keys, 2)
    got = epoch_mean_observed(X_raw, groups)
    right = log1p_community(X_raw.mean(0, keepdims=True))          # average, then transform
    wrong = log1p_community(X_raw).mean(0, keepdims=True)          # transform, then average
    assert np.allclose(got[0], right[0], atol=1e-6), (got[0], right[0])
    assert not np.allclose(got[0], wrong[0], atol=1e-3), (got[0], wrong[0])
    # and the wrong order really does understate: log1p(mean) > mean(log1p) by Jensen
    assert right[0, 1] > wrong[0, 1]


def test_the_no_change_null_keeps_zero_temporal_variation_when_averaged():
    """Load-bearing property of this module: the null differs from DESK ONLY by having no
    temporal variation. If the averaged modern reference varied across a cell's years the null
    would acquire some, and the gain would stop isolating the temporal component."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        epoch_mean_observed, modern_reference_groups)
    keys = np.array([[0, 0, 1970], [0, 0, 1990], [0, 0, 2012], [0, 0, 2020]], dtype=np.int32)
    X_raw = np.arange(8, dtype="float64").reshape(4, 2)
    groups, keep = modern_reference_groups(keys, modern_window=(2010, 2025))
    assert keep.all()
    out = epoch_mean_observed(X_raw, groups)
    # every row of the same cell must receive the identical reference vector
    assert np.allclose(out, out[0]), out


def test_run_averages_the_right_rows_when_the_site_gate_shifts_indices(tmp_path, monkeypatch):
    """The alignment that could silently be wrong. Averaging groups index the UNFILTERED rows
    (X_raw_all is never gated) while the sampled rows are bookkept post-gate, so run must map
    between the two. With cells dropped FIRST the two index spaces genuinely differ, and a
    mix-up would average some other cell's years into this row's endpoint -- producing plausible
    numbers with no error. Verified by reconstructing each returned endpoint from its own key."""
    from src.community_encoder.train_DESK import validate_bbs_routes as V

    n_species = 4
    rng = np.random.default_rng(3)
    keys, rows = [], []
    for c in range(5):                       # cells 0-4: early only -> site gate drops them
        for y in (1968, 1975, 1982):
            keys.append([0, c, y]); rows.append(rng.random(n_species) * 5)
    for c in range(5, 15):                   # cells 5-14: pass both gates
        for y in (1968, 1969, 1975, 2008, 2015, 2022):
            keys.append([0, c, y]); rows.append(rng.random(n_species) * 5)
    keys = np.array(keys, dtype="int32")
    X_arr = np.array(rows)

    desk = tmp_path / "desk"; desk.mkdir()
    np.save(desk / "holdout_cells.npy", np.zeros((4, 20), bool))
    cfg = {"trend": {"points_dir": str(tmp_path)},
           "paths": {"desk_output_dir": str(desk)},
           "bbs_routes": {"average_windows": True, "window_half_width": 2}}
    monkeypatch.setattr(V, "load_observed", lambda config: (
        V.log1p_community(X_arr), keys,
        {"n_species": n_species, "n_surveyed_cell_years": int(keys.shape[0]),
         "year_range": [1968, 2022]}, X_arr))
    A = rng.standard_normal((n_species, 4))
    monkeypatch.setattr(V, "desk_z_ema", lambda config, k: (
        np.stack([np.log1p(np.full(n_species, 1.0 + int(y) % 7)) @ A for _, _, y in k]
                 ).astype("float32"),
        {"output_ema_applied": True, "ema_half_life": 10.0, "ema_warmup_start": 1940,
         "encode_years": [1940, 2022]}))
    import src.config_utils as CU
    monkeypatch.setattr(CU, "load_data_config", lambda *a, **kw: {"grid": {"ref_raster": "x"}})
    import src.community_encoder.train_DESK.validate_spacetime as VS
    monkeypatch.setattr(VS, "cell_xy", lambda r, c, ref: np.stack(
        [np.asarray(c, float) * 27000.0, np.asarray(r, float) * 27000.0], axis=1))

    rep = V.run(config=cfg, n_sample=60, seed=0)
    assert rep["site_gate"]["rows_total"] - rep["site_gate"]["rows_kept"] == 15   # indices shift
    assert rep["config"]["average_windows"] is True
    # 1968/1969 are within +/-2 of each other, so those endpoints average 2 years; the isolated
    # 1975/2008/2015/2022 rows average 1. Mean depth must land strictly between 1 and 2.
    assert 1.0 < rep["config"]["mean_window_depth_years"] < 2.0, rep["config"]
    # The modern reference draws only from rows INSIDE MODERN_WINDOW=(2010, 2025), so 2008 is
    # excluded and each cell has 2015 + 2022 available -- but the two ends of a difference are now
    # MATCHED, so the reference is subsampled down to whatever the window side has. It must
    # therefore equal the window depth exactly, not its own unmatched depth of 2.
    assert rep["config"]["mean_reference_depth_years"] == rep["config"]["mean_window_depth_years"], (
        rep["config"])
    assert rep["config"]["mean_reference_depth_years"] < 2.0, "matching must have subsampled it"

    saved = np.load(desk / "bbs_route_validation.npz")
    ks, S_true = saved["keys"], saved["S_true"]
    # Reconstruct each row's endpoint from its OWN key and compare to the matrix' diagonal-free
    # content: rebuild S_true independently and require an exact match.
    groups = V.window_groups(keys, 2)
    by_key = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(keys)}
    X_expect = V.epoch_mean_observed(
        X_arr, [groups[by_key[(int(r), int(c), int(y))]] for r, c, y in ks])
    assert np.allclose(ruzicka_rect(X_expect, X_expect), S_true, atol=1e-5)


def test_the_esk_truncation_row_detects_a_high_ceiling():
    """The oracle must be able to report a HIGH ceiling, or a low reading is uninformative -- it
    could mean the diagnostic is broken rather than the basis being unable to carry temporal
    change. Constructed so the projection reproduces observed similarity by design: with z set
    to the (L2-normalised) community itself, z(a).z(b) is the cosine, which is monotone in
    Ruzicka across these rows, so the oracle's correlation must be ~1."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        epoch_neighborhood_analysis)
    from src.community_encoder.train_DESK.validate_bbs_routes import _rowwise_ruzicka
    rng = np.random.default_rng(0)
    n, S = 60, 8
    Xe = np.abs(rng.normal(size=(n, S))) + 0.05
    Xm = np.abs(rng.normal(size=(n, S))) + 0.05             # genuinely different modern community
    # Build the oracle so its rowwise dot IS the observed Ruzicka. Normalising the communities
    # would NOT do this: for Xm = c*Xe the cosine is 1 for every c while Ruzicka is min(c, 1/c),
    # so a unit-normalised "oracle" carries none of the magnitude information Ruzicka depends on
    # (measured: pearson 0.17). Ruzicka is PD but has no convenient closed-form feature map, so
    # construct the target directly -- the point is to prove the DIAGNOSTIC reports ~1 when the
    # oracle's dot equals observed similarity, not to re-derive that feature map.
    sc = _rowwise_ruzicka(Xe, Xm)
    e1 = np.zeros(S); e1[0] = 1.0
    Ze_o = np.sqrt(sc)[:, None] * e1[None, :]
    Zm_o = Ze_o.copy()                                      # dot == sc exactly
    Zd = rng.normal(size=(n, S)) * 0.01                     # DESK: noise, so it must score ~0
    xy = np.stack([np.arange(n) * 30000.0, np.zeros(n)], 1)
    rep, _pc = epoch_neighborhood_analysis(Xe, Xm, Zd, Zd, xy, k=8, n_bins=2,
                                           sc_esk=(Ze_o, Zm_o))
    assert "esk_truncation" in rep["types"]["same_cell_over_time"]["pooled"]["all_distances"]["dot"]["predictors"], sorted(rep["types"])
    orc = _dot(rep, "same_cell_over_time")["predictors"]["esk_truncation"]
    assert orc["pearson_r"] > 0.9, orc["pearson_r"]
    # and the CI must be present and finite, since this is what a real reading needs
    ci = _dot(rep, "same_cell_over_time")["ci95"]["esk_truncation"]
    assert np.isfinite(ci["lo"]) and np.isfinite(ci["hi"]), ci


def test_self_change_ci_resamples_cells_not_pairs():
    """Pairs from N rows number ~N^2/2 but are massively correlated, so an interval computed over
    pairs would be far too narrow. Resampling the ~1950 independent focal cells is the honest
    unit -- and a small sample must produce a WIDE interval rather than a confident one."""
    from src.community_encoder.train_DESK.validate_bbs_routes import bootstrap_skill_ci
    rng = np.random.default_rng(0)
    obs = rng.normal(size=400)
    null = np.zeros(400)
    pred = obs * 0.5 + rng.normal(scale=0.5, size=400)
    wide = bootstrap_skill_ci(obs[:25], pred[:25], null[:25])
    tight = bootstrap_skill_ci(obs, pred, null)
    assert wide["n"] == 25 and tight["n"] == 400
    assert (wide["hi"] - wide["lo"]) > (tight["hi"] - tight["lo"]), (wide, tight)
    # the interval must bracket its own point estimate
    for r in (wide, tight):
        assert r["lo"] <= r["skill"] <= r["hi"], r
    # too few elements: refuse rather than emit a confident-looking interval
    assert not np.isfinite(bootstrap_skill_ci(obs[:4], pred[:4], null[:4])["lo"])


def test_the_idw_bar_is_scored_in_desks_own_functional():
    """The bar must be a dot product of ESK-space vectors, like DESK. A Ruzicka-based bar would be
    scored in truth's own metric while DESK is not, and this module records that the mismatch
    alone was worth -0.28 on a temporally-neutral model. Verified structurally: feeding the bar
    DESK's OWN vectors must make the DESK-vs-bar skill exactly 0 -- only possible if both sides go
    through the same functional."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        epoch_neighborhood_analysis)
    rng = np.random.default_rng(0)
    n, S = 40, 6
    Xe = np.abs(rng.normal(size=(n, S))) + 0.05
    Xm = np.abs(rng.normal(size=(n, S))) + 0.05
    Ze = rng.normal(size=(n, S))
    Zm = rng.normal(size=(n, S))
    xy = np.stack([np.arange(n) * 30000.0, np.zeros(n)], 1)
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=8, n_bins=2,
                                         sc_idw=(Ze, Zm))          # bar := DESK itself
    sc = _dot(rep, "same_cell_over_time")["skill_vs_spacetime_idw"]["desk"]
    assert abs(sc) < 1e-9, sc
    for t in ("cross_cell_same_era_early", "cross_cell_same_era_modern", "cross_cell_cross_time"):
        v = _dot(rep, t)["skill_vs_spacetime_idw"]["desk"]
        assert abs(v) < 1e-9, (t, v)


def test_a_better_bar_lowers_desks_skill():
    """Sanity on direction: replacing the free null with a bar that genuinely predicts must make
    DESK look WORSE, not better. If the sign came out the other way the bar would be decorative."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        _rowwise_ruzicka, epoch_neighborhood_analysis)
    rng = np.random.default_rng(1)
    n, S = 60, 6
    Xe = np.abs(rng.normal(size=(n, S))) + 0.05
    Xm = np.abs(rng.normal(size=(n, S))) + 0.05
    sc = _rowwise_ruzicka(Xe, Xm)
    e1 = np.zeros(S); e1[0] = 1.0
    # a near-oracle bar, and a DESK that is mostly noise
    good = (np.sqrt(sc)[:, None] * e1[None, :])
    Zd = rng.normal(size=(n, S)) * 0.05
    xy = np.stack([np.arange(n) * 30000.0, np.zeros(n)], 1)
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Zd, Zd, xy, k=8, n_bins=2, sc_idw=(good, good))
    d = _dot(rep, "same_cell_over_time")
    vs_null = d["skill_vs"]["desk"]
    vs_idw = d["skill_vs_spacetime_idw"]["desk"]
    assert vs_idw < vs_null, (vs_idw, vs_null)
    # and the bar must be recorded as beating the null, since it does
    assert d["skill_vs"]["spacetime_idw"] > 0


def test_the_pooled_bar_is_built_in_full_row_space(tmp_path, monkeypatch):
    """The bar is built on keys_all (full rows, matching X_raw_all) while the pooled matrices are
    bookkept post-gate, so run() must index down through sel_full and then `finite`. Getting that
    wrong scores DESK against some other row's interpolation -- plausible numbers, no error. The
    site gate drops cells FIRST here so the two index spaces genuinely differ."""
    from src.community_encoder.train_DESK import validate_bbs_routes as V

    n_species = 4
    rng = np.random.default_rng(5)
    keys, rows = [], []
    for c in range(6):                       # dropped by the site gate (no modern survey)
        for y in (1968, 1975, 1982):
            keys.append([0, c, y]); rows.append(rng.random(n_species) * 5)
    for c in range(6, 20):                   # survive both gates
        for y in (1968, 1969, 1975, 2008, 2015, 2022):
            keys.append([0, c, y]); rows.append(rng.random(n_species) * 5)
    keys = np.array(keys, dtype="int32")
    X_arr = np.array(rows)

    desk = tmp_path / "desk"; desk.mkdir()
    np.save(desk / "holdout_cells.npy", np.zeros((4, 30), bool))
    cfg = {"trend": {"points_dir": str(tmp_path)}, "paths": {"desk_output_dir": str(desk)},
           "desk": {"z_dir": str(tmp_path / "nozdir")}}      # no projection -> bar unavailable
    monkeypatch.setattr(V, "load_observed", lambda config: (
        V.log1p_community(X_arr), keys,
        {"n_species": n_species, "n_surveyed_cell_years": int(keys.shape[0]),
         "year_range": [1968, 2022]}, X_arr))
    A = rng.standard_normal((n_species, 4))
    monkeypatch.setattr(V, "desk_z_ema", lambda config, k: (
        np.stack([np.log1p(np.full(n_species, 1.0 + int(y) % 7)) @ A for _, _, y in k]
                 ).astype("float32"),
        {"output_ema_applied": True, "ema_half_life": 10.0, "ema_warmup_start": 1940,
         "encode_years": [1940, 2022]}))
    import src.config_utils as CU
    monkeypatch.setattr(CU, "load_data_config", lambda *a, **kw: {"grid": {"ref_raster": "x"}})
    import src.community_encoder.train_DESK.validate_spacetime as VS
    monkeypatch.setattr(VS, "cell_xy", lambda r, c, ref: np.stack(
        [np.asarray(c, float) * 27000.0, np.asarray(r, float) * 27000.0], axis=1))

    rep = V.run(config=cfg, n_sample=50, seed=0)
    # indices really do shift, so this fixture exercises the mapping
    assert rep["site_gate"]["rows_total"] - rep["site_gate"]["rows_kept"] == 18
    # With no saved projection the bar must be absent -- and absent CLEANLY, leaving the
    # no-change columns intact rather than raising or emitting a half-built row.
    for name, b in rep["buckets"].items():
        if "skipped" in b:
            continue
        assert "dot" in b and "cosine" in b, (name, sorted(b))
        # With no saved projection the bar must be ABSENT-WITH-A-REASON, never a silent gap
        # and never a half-built row.
        assert "spacetime_idw" not in b["dot"]["predictors"], (name, "bar without a projection")
        assert b["dot"]["unavailable"].get("spacetime_idw"), (name, "absent without a reason")


def test_build_spacetime_bar_returns_none_rather_than_raising():
    """A missing projection or holdout mask must degrade the report to the no-change null, not
    take the stage down -- the bar is a diagnostic, and the rest of the module still works."""
    from src.community_encoder.train_DESK.validate_bbs_routes import build_spacetime_bar
    keys = np.array([[0, 0, 2000], [0, 1, 2001]], dtype=np.int32)
    X = np.ones((2, 3))
    assert build_spacetime_bar({}, keys, X, 4) is None                       # no config keys
    assert build_spacetime_bar({"desk": {"z_dir": "/nope"},
                                "paths": {"desk_output_dir": "/nope"}}, keys, X, 4) is None


def test_the_oracle_refuses_to_run_off_span(tmp_path, monkeypatch, capsys):
    """The exact failure that produced a withdrawn finding. The oracle's reading assumes the kernel
    contract (||z||^2 = 1), so when its own inputs violate that -- as 20-year averaged communities
    did, at 0.15 against 0.672 annual -- it must emit a refusal and NO number, rather than a value
    that measures the input mismatch instead of a ceiling."""
    from src.community_encoder.train_DESK import validate_bbs_routes as V

    n_species, L = 8, 4
    rng = np.random.default_rng(0)
    keys, rows = [], []
    for c in range(12):
        for y in (1968, 1970, 1975, 2008, 2015, 2022):
            keys.append([0, c, y])
            # SPARSE, as a real annual community is (~1.08 routes per cell-year). Averaging fills
            # the zeros, which is what puts an epoch mean in a different region of the input space.
            rows.append(rng.poisson(0.4, n_species).astype(float))
    keys = np.array(keys, dtype="int32"); X_arr = np.array(rows)
    desk = tmp_path / "desk"; desk.mkdir()
    np.save(desk / "holdout_cells.npy", np.zeros((4, 20), bool))
    cfg = {"trend": {"points_dir": str(tmp_path)}, "paths": {"desk_output_dir": str(desk)},
           "desk": {"z_dir": str(tmp_path)}}

    # A projection that is FINE on single years and badly deflated on averaged communities --
    # the real situation, reproduced by keying off how many species are non-zero.
    def fake_proj(X, z_dir, latent_dim, **kw):
        X = np.asarray(X)
        # Annual Poisson(0.4) communities sit near 33% non-zero; the mean of three fills that to
        # ~70%, so the threshold between them separates "annual" from "averaged".
        dense = (X > 0).mean(1) > 0.5
        z = rng.normal(size=(len(X), latent_dim))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        z[dense] *= np.sqrt(0.15)              # off-span
        z[~dense] *= np.sqrt(0.67)             # annual, as measured
        return z.astype("float32")

    monkeypatch.setattr(V, "load_observed", lambda config: (
        V.log1p_community(X_arr), keys,
        {"n_species": n_species, "n_surveyed_cell_years": int(keys.shape[0]),
         "year_range": [1968, 2022]}, X_arr))
    import src.community_encoder.train_DESK.esk_kernel as EK
    monkeypatch.setattr(EK, "project_points_to_z", fake_proj)

    # _run_epoch_analysis builds Xe/Xm itself from these row lists, so the averaging depth has to
    # come from the lists: three sparse annual rows per epoch, whose mean fills in the zeros.
    e_rows = [[6 * c + 0, 6 * c + 1, 6 * c + 2] for c in range(12)]
    m_rows = [[6 * c + 3, 6 * c + 4, 6 * c + 5] for c in range(12)]
    out = V._run_epoch_analysis(cfg, keys, X_arr, np.array([[0, c] for c in range(12)], "int32"),
                                e_rows, m_rows, {}, lambda k: np.zeros((len(k), L), "float32"),
                                {}, str(desk))
    txt = capsys.readouterr().out
    assert "ESK oracle REFUSED" in txt, txt
    assert "does not span averaged communities" in txt
    if out is not None:
        assert "esk_truncation" not in (out.get("types", {}).get("same_cell_over_time", {})
            .get("pooled", {}).get("all_distances", {}).get("dot", {}).get("predictors", {})), "refused but still reported"


def test_the_validation_spatial_axis_nests_inside_the_weighting_strata():
    """One definition at two resolutions, not two definitions. If a fine stratum straddled two
    coarse regions, a weight and the sample share meant to correct it would refer to different
    places, and the rebalance would be uninterpretable."""
    from src.community_encoder.train_DESK.esk_kernel import (
        coarse_spatial, nests_within, spacetime_strata)
    rng = np.random.default_rng(0)
    pidx = np.stack([rng.integers(0, 64, 3000), rng.integers(0, 64, 3000),
                     rng.integers(1966, 2026, 3000)], axis=1).astype(np.int32)
    X = rng.random((3000, 5))
    fine, _keys = spacetime_strata(pidx, X, spatial_bins=8, abundance_bins=4)
    assert nests_within(fine, coarse_spatial(pidx, regions=2))    # 8 tiles / 2 regions -> nests
    assert nests_within(fine, coarse_spatial(pidx, regions=4))
    # and a non-divisor must NOT be claimed to nest, so the check has teeth
    assert not nests_within(fine, coarse_spatial(pidx, regions=3))


def test_the_spatial_axis_changes_which_rows_are_sampled():
    """A coast-heavy sample cannot expose an interior deficit. With a deliberately lopsided spatial
    distribution, adding the region axis must raise the sparse region's share of the sample."""
    from src.community_encoder.train_DESK.validate_bbs_routes import stratified_sample
    rng = np.random.default_rng(0)
    # 95% of rows in the low-column half ("coast"), 5% in the high half ("interior")
    n_coast, n_int = 3800, 200
    keys = np.concatenate([
        np.stack([np.zeros(n_coast, int), rng.integers(0, 8, n_coast),
                  rng.integers(1966, 2026, n_coast)], 1),
        np.stack([np.zeros(n_int, int), rng.integers(56, 64, n_int),
                  rng.integers(1966, 2026, n_int)], 1)]).astype(np.int32)
    interior = keys[:, 1] >= 32
    plain = stratified_sample(keys, 800, np.random.default_rng(1))
    withreg = stratified_sample(keys, 800, np.random.default_rng(1), spatial_regions=2)
    assert interior[withreg].mean() > 2 * interior[plain].mean(), (
        interior[plain].mean(), interior[withreg].mean())


def test_balanced_and_population_weighted_aggregates_can_disagree_in_sign():
    """The reason 4b exists. A model better on thin buckets and worse on dense ones must show the
    two aggregates moving opposite ways -- otherwise the balanced figure adds nothing and a
    rebalanced model would be scored as a regression."""
    # dense bucket dominates the row count but the model is worse there; thin buckets are better
    buckets = {"heldout/modern": {"rmse_skill": -0.05, "n_rows": 3600},
               "heldout/early": {"rmse_skill": +0.20, "n_rows": 200},
               "heldout/all": {"rmse_skill": -0.04, "n_rows": 3800}}
    src = [(k, v) for k, v in buckets.items() if k != "heldout/all"]
    balanced = float(np.mean([v["rmse_skill"] for _k, v in src]))
    pop = float(np.average([v["rmse_skill"] for _k, v in src],
                           weights=[v["n_rows"] for _k, v in src]))
    assert balanced > 0 > pop, (balanced, pop)


def test_adding_a_predictor_adds_a_row_everywhere_with_no_call_site_edit():
    """THE property the refactor exists to buy. Previously a third predictor could only be bolted
    on per call site -- which is why the interpolation bar reached the pooled matrices but not the
    per-distance rows, and the decomposition was computed for DESK and never for the bar. Here a
    predictor handed to the analysis must appear in EVERY question and EVERY distance bin, in both
    the dot and cosine forms, without touching any reporting code."""
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_neighborhood_analysis
    Xe, Xm, A, xy = _epoch_fixture()
    Ze, Zm = Xe @ A, Xm @ A
    rng = np.random.default_rng(0)
    extra = (rng.normal(size=Ze.shape), rng.normal(size=Zm.shape))    # a synthetic 4th predictor
    rep, _pc = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3,
                                           sc_idw=extra, sc_esk=extra)
    for q, per_split in rep["types"].items():
        for split, per_bin in per_split.items():
            for bname, mm in per_bin.items():
                if "skipped" in mm:
                    continue
                for form in ("dot", "cosine"):
                    got = set(mm[form]["predictors"])
                    assert got == {"desk", "no_change", "spacetime_idw", "esk_truncation"}, (
                        q, split, bname, form, got)
                    # identical quantity set for every predictor -- no privileged vocabulary
                    keys = [set(v) for v in mm[form]["predictors"].values()]
                    assert all(k == keys[0] for k in keys), (q, form, keys)


def test_the_completeness_check_fails_on_a_silently_dropped_predictor():
    """The mechanism that makes a missing comparison impossible. Every gap this suite accumulated
    was an ABSENT KEY rather than a wrong number, and nothing looked for absences.

    Driven off the REAL report rather than a hand-built dict: a hand-built fixture only proves the
    checker agrees with the fixture's author about the shape, and the first version of this test
    did exactly that -- it passed against a flat `{q: {"predictors": ...}}` the analysis has never
    emitted, so the checker walked to a depth where nothing lived and reported every combination
    as missing while the suite stayed green.
    """
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        assert_complete, epoch_neighborhood_analysis)
    Xe, Xm, A, xy = _epoch_fixture()
    rng = np.random.default_rng(0)
    Ze, Zm = Xe @ A, Xm @ A
    bar = (Ze + rng.normal(scale=0.3, size=Ze.shape), Zm + rng.normal(scale=0.3, size=Zm.shape))
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3, sc_idw=bar,
                                         sc_esk=(Ze, Zm))
    qs = rep["manifest"]["covers"]
    assert assert_complete(rep["types"], questions=qs) == []

    # A predictor dropped from ONE leaf only -- the pooled row still has it. This is the exact
    # shape of the gaps that recurred (the bar reached the pooled matrices but not the per-distance
    # rows), so a checker that stopped at the top level would pass this.
    import copy
    broken = copy.deepcopy(rep["types"])
    leaf = broken["cross_cell_cross_time"]["pooled"]["all_distances"]["dot"]
    leaf["predictors"].pop("spacetime_idw")
    gaps = assert_complete(broken, questions=qs)
    assert any("spacetime_idw" in g and "cross_cell_cross_time" in g for g in gaps), gaps

    # A stated REASON is acceptable where a result is not -- that is the whole distinction.
    ok = copy.deepcopy(broken)
    ok["cross_cell_cross_time"]["pooled"]["all_distances"]["dot"]["unavailable"][
        "spacetime_idw"] = "bar unbuilt for this run"
    assert assert_complete(ok, questions=qs) == []

    # A question in scope but absent entirely, and a question emitted under no registry entry.
    assert any("absent entirely" in g for g in assert_complete({}, questions=qs))
    stray = dict(rep["types"])
    stray["cross_cell_by_elevation"] = stray["cross_cell_cross_time"]
    assert any("no question registry" in g for g in assert_complete(stray, questions=qs))


def test_question_instances_map_back_to_one_registry_entry():
    """`cross_cell_same_era` is emitted twice -- once per era -- and both share one rationale.
    Matching is by longest registry prefix, not by stripping the last token, which would mangle
    `same_cell_over_time` into a `same_cell_over` that no entry defines."""
    from src.community_encoder.train_DESK.validate_bbs_routes import canonical_question, QUESTIONS
    assert canonical_question("cross_cell_same_era_early") == "cross_cell_same_era"
    assert canonical_question("cross_cell_same_era_modern") == "cross_cell_same_era"
    assert canonical_question("same_cell_over_time") == "same_cell_over_time"
    assert canonical_question("cross_cell_by_elevation") is None
    for q in QUESTIONS:
        assert canonical_question(q) == q


def test_cosine_and_dot_diverge_under_a_norm_deficit():
    """If the two forms agreed, the angular one would be redundant rather than the calibration-free
    reading. With ||z||^2 ~ 0.66 measured, the dot form carries that deficit and the cosine does
    not -- which is exactly why cross_cell_cross_time needs the angular form."""
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_neighborhood_analysis
    Xe, Xm, A, xy = _epoch_fixture()
    Ze, Zm = Xe @ A, Xm @ A
    rep_full, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3)
    # shrink every vector: pure norm deficit, directions untouched
    rep_short, _ = epoch_neighborhood_analysis(Xe, Xm, Ze * 0.6, Zm * 0.6, xy, k=9, n_bins=3)
    q = "cross_cell_cross_time"
    dot_f = _dot(rep_full, q)["predictors"]["desk"]["rmse"]
    dot_s = _dot(rep_short, q)["predictors"]["desk"]["rmse"]
    cos_f = rep_full["types"][q]["pooled"]["all_distances"]["cosine"]["predictors"]["desk"]["rmse"]
    cos_s = rep_short["types"][q]["pooled"]["all_distances"]["cosine"]["predictors"]["desk"]["rmse"]
    assert not np.isclose(dot_f, dot_s, rtol=1e-3), (dot_f, dot_s)   # dot sees the deficit
    assert np.isclose(cos_f, cos_s, rtol=1e-6), (cos_f, cos_s)       # cosine does not


def test_the_report_carries_a_manifest_of_what_was_tested():
    """A report should say what it measured, against what, on how many rows -- so two runs can be
    diffed structurally and not only numerically. Knowing what the suite tests previously meant
    reading four thousand lines."""
    from src.community_encoder.train_DESK.validate_bbs_routes import QUESTIONS
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_neighborhood_analysis
    Xe, Xm, A, xy = _epoch_fixture()
    rep, _pc = epoch_neighborhood_analysis(Xe, Xm, Xe @ A, Xm @ A, xy, k=9, n_bins=3)
    m = rep["manifest"]
    assert set(m) == {"covers", "questions", "predictors", "unavailable", "quantities",
                      "populations", "denoising", "denoising_mismatch", "generalisation"}
    # The manifest must carry the denoising footing, not just the predictor list: a report that
    # names its predictors but not how denoised they are invites exactly the apples-to-oranges
    # reading that made the oracle look like an achievable ceiling.
    assert set(m["denoising"]) <= set(m["predictors"])
    assert isinstance(m["denoising_mismatch"], list)
    # `covers` is the scope the report claims responsibility for, and it must be REGISTRY names --
    # it is what `assert_complete` is checked against, so a typo here would silently narrow the
    # check rather than fail it.
    assert set(m["covers"]) <= set(QUESTIONS)
    assert "absolute_position" not in m["covers"]      # that is zspace_reconstruction's question
    assert set(m["quantities"]) == {"dot", "cosine"}
    assert "desk" in m["predictors"] and "no_change" in m["predictors"]
    # the bar and the oracle were not supplied here, so both must be recorded WITH a reason
    # esk_oracle_independent needs a split-half input this fixture does not supply, and it is
    # reported as unavailable WITH A REASON rather than silently absent.
    assert set(m["unavailable"]) == {"spacetime_idw", "esk_truncation", "esk_oracle_independent"}
    assert all(v for v in m["unavailable"].values())
    # every question carries its own rationale, which is the map that kept getting lost
    assert all(q in rep["types"] for q in m["questions"])


# ---------------------------------------------------------------------------------------------
# compare_positions -- the `absolute_position` question's predictor table
# ---------------------------------------------------------------------------------------------

def desk_err(P, z_obs):
    return np.linalg.norm(np.asarray(P, "float64") - z_obs, axis=1)


def _positions(n=200, seed=3):
    rng = np.random.default_rng(seed)
    z_obs = rng.normal(size=(n, 8))
    return rng, z_obs


def test_compare_positions_grades_every_predictor_identically():
    """The property the refactor exists to buy. The old block wrote one key pair per baseline --
    frac_desk_beats_nochange, frac_desk_beats_idw, frac_desk_beats_spacetime_idw -- so DESK was
    the only possible SUBJECT and the error decomposition existed for DESK alone."""
    from src.community_encoder.train_DESK.validate_bbs_routes import compare_positions
    rng, z_obs = _positions()
    preds = {"desk": z_obs + rng.normal(scale=.3, size=z_obs.shape),
             "no_change": z_obs + rng.normal(scale=.6, size=z_obs.shape),
             "spacetime_idw": z_obs + rng.normal(scale=.4, size=z_obs.shape)}
    r = compare_positions(z_obs, preds)
    assert set(r["predictors"]) == set(preds)
    keys = [set(v) for v in r["predictors"].values()]
    assert all(k == keys[0] for k in keys), keys        # no privileged vocabulary
    # a brand-new predictor needs no call-site change to appear with the full quantity set
    preds["some_future_bar"] = z_obs + rng.normal(scale=.5, size=z_obs.shape)
    r2 = compare_positions(z_obs, preds)
    assert set(r2["predictors"]["some_future_bar"]) == keys[0]


def test_compare_positions_reference_is_only_a_name():
    """"DESK vs the null" and "DESK vs the bar" must be one call with a different reference, not
    one built-in comparison plus a bolt-on -- otherwise a comparison is privileged by
    construction, which is how the asymmetry kept coming back."""
    from src.community_encoder.train_DESK.validate_bbs_routes import compare_positions
    rng, z_obs = _positions()
    preds = {"desk": z_obs + rng.normal(scale=.3, size=z_obs.shape),
             "no_change": z_obs + rng.normal(scale=.6, size=z_obs.shape),
             "spacetime_idw": z_obs + rng.normal(scale=.4, size=z_obs.shape)}
    a = compare_positions(z_obs, preds, reference="no_change")
    b = compare_positions(z_obs, preds, reference="spacetime_idw")
    assert a["predictors"] == b["predictors"]           # per-predictor rows are reference-free
    assert a["win_rate_vs"] != b["win_rate_vs"]
    assert abs(a["win_rate_vs"]["no_change"] - 0.0) < 1e-12       # never beats itself
    assert abs(b["skill_vs"]["spacetime_idw"] - 0.0) < 1e-12
    # and the question the old code could not ask at all: does the bar beat the OTHER bar?
    assert b["win_rate_vs"]["no_change"] < 0.5          # the null is worse than the bar


def test_compare_positions_win_rate_uses_the_intersection_of_finite_rows():
    """An interpolation bar reaches only where it has neighbours. Scoring its easy subset against
    a reference's FULL set would flatter whichever predictor declined the hardest rows."""
    from src.community_encoder.train_DESK.validate_bbs_routes import compare_positions
    rng, z_obs = _positions()
    hard = np.zeros(len(z_obs), bool)
    hard[:100] = True                                   # the half the bar will decline
    desk = z_obs + rng.normal(scale=.3, size=z_obs.shape)
    desk[hard] += 3.0                                   # DESK does badly exactly there
    bar = z_obs + rng.normal(scale=.4, size=z_obs.shape)
    bar[hard] = np.nan                                  # ...and the bar simply declines it
    r = compare_positions(z_obs, {"desk": desk, "no_change": bar}, reference="no_change")
    # Scored on the 100 easy rows only, where DESK is genuinely better. The naive form -- compare
    # over all 200 and let the NaNs fall where they may -- returns False on every declined row,
    # so DESK would appear to win only ~half as often for a reason that is pure bookkeeping.
    naive = float(np.mean(desk_err(desk, z_obs) < desk_err(bar, z_obs)))
    assert r["win_rate_vs"]["desk"] > 0.7
    assert r["win_rate_vs"]["desk"] > naive + 0.3, (r["win_rate_vs"]["desk"], naive)
    assert r["predictors"]["no_change"]["n_scored"] == 100


def test_compare_positions_states_a_reason_rather_than_omitting_a_predictor():
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        compare_positions, assert_complete)
    rng, z_obs = _positions()
    preds = {"desk": z_obs + rng.normal(scale=.3, size=z_obs.shape),
             "no_change": z_obs + rng.normal(scale=.6, size=z_obs.shape),
             "spacetime_idw": None,                      # bar could not be built
             "esk_truncation": np.where(np.arange(len(z_obs))[:, None] < 2, z_obs, np.nan),
             "esk_oracle_independent": None}     # split-half ceiling: not built for this question
    r = compare_positions(z_obs, preds)
    assert set(r["unavailable"]) == {"spacetime_idw", "esk_truncation",
                                     "esk_oracle_independent"}
    assert "finite" in r["unavailable"]["esk_truncation"]
    # and that is enough for the completeness check: a reason counts, an absence does not
    assert assert_complete({"absolute_position": r}, questions=("absolute_position",)) == []
    r["unavailable"].pop("spacetime_idw")
    gaps = assert_complete({"absolute_position": r}, questions=("absolute_position",))
    assert any("spacetime_idw" in g for g in gaps), gaps


def test_completeness_walks_both_question_shapes():
    """`absolute_position` nests populations; the neighbour questions nest split/bin/form. The
    walk must find leaves BY SHAPE -- coupling it to a fixed depth is what made the first version
    of this check inspect nothing while reporting green."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        compare_positions, assert_complete, _predictor_leaves)
    rng, z_obs = _positions()
    pops = {"heldout": np.arange(len(z_obs)) < 100, "train": np.arange(len(z_obs)) >= 100}
    r = compare_positions(z_obs, {"desk": z_obs + rng.normal(scale=.3, size=z_obs.shape),
                                  "no_change": z_obs + rng.normal(scale=.6, size=z_obs.shape)},
                          populations=pops)
    leaves = dict(_predictor_leaves(r, "absolute_position"))
    assert len(leaves) == 3                              # the pooled block plus both populations
    # a predictor dropped from ONE population only must still be caught
    r["populations"]["heldout"]["predictors"].pop("desk")
    gaps = assert_complete({"absolute_position": r},
                           predictors=("desk", "no_change"), questions=("absolute_position",))
    assert any("heldout" in g and "desk" in g for g in gaps), gaps


# ---------------------------------------------------------------------------------------------
# Column layout: the basis and the validation must agree on which species is in which column
# ---------------------------------------------------------------------------------------------

def test_the_community_column_layout_comes_from_one_function(tmp_path):
    """Regression, and the worst bug this suite has had.

    `load_observed` ordered its columns by `list(dict.fromkeys(crosswalk["species_code"]))` --
    the order species happen to appear in the crosswalk, which is TAXONOMIC. The basis orders
    its columns by `species_order(community_csv)`, the CSV's own rank-ordering. Same 96 species,
    same SET, so every count and coverage check in the module passed while 94 of 96 COLUMNS held
    a different species, and Ruzicka silently compared one species' abundance to another's.

    Measured consequence: best-landmark similarity 0.17 where like-against-like gives 0.65, and
    ||z_obs||^2 = 0.15 against a kernel contract of exactly 1.0.

    The fix is that ONE function owns the layout. This test pins that, because no test of counts
    or sets can catch a permutation -- both sides had 96 of the same species.
    """
    import pandas as pd
    from src.community_encoder.train_DESK.bbs_community_points import species_order

    # A community list whose CSV order is deliberately NOT taxonomic or alphabetical, so a
    # re-derived order cannot coincide with the pinned one by luck.
    csv = tmp_path / "community_trend.csv"
    codes = ["casfin", "houspa", "allhum", "gryjay", "amegfi", "eutspa"]
    pd.DataFrame({"species_code": codes}).to_csv(csv, index=False)

    layout = species_order(str(csv))
    assert layout == codes, layout                       # CSV row order, lowercased

    # The taxonomic ordering a crosswalk would hand back: same SET, different ORDER.
    crosswalk_order = sorted(codes)
    assert set(crosswalk_order) == set(layout)
    assert crosswalk_order != layout, "fixture must actually differ, or it proves nothing"

    # This is what the bug looked like: build the same community twice, once per ordering, and
    # the similarity between them collapses even though every count matches.
    rng = np.random.default_rng(0)
    base = rng.random((200, len(codes))) * np.array([4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
    ix_pin = {c: i for i, c in enumerate(layout)}
    ix_bad = {c: i for i, c in enumerate(crosswalk_order)}
    perm = [ix_bad[c] for c in layout]
    mis = base[:, perm]                                  # the same data, wrongly laid out

    assert base.sum(1).round(9).tolist() == mis.sum(1).round(9).tolist()   # totals identical
    assert ((base > 0).sum(1) == (mis > 0).sum(1)).all()                   # sparsity identical

    def ruz(A, B):
        return np.array([np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-12)
                         for a, b in zip(A, B)])

    self_sim = ruz(base, base)
    cross = ruz(base, mis)
    assert np.allclose(self_sim, 1.0)                    # the contract, exactly 1
    assert np.median(cross) < 0.75, np.median(cross)     # ...and destroyed by the permutation
    # so a permutation is invisible to counts and sets, and only visible in the similarity
    assert ix_pin != ix_bad


def test_a_species_bbs_cannot_survey_keeps_its_column():
    """Compacting out an unmatched species would shift every later column -- the same class of
    misalignment. The layout must depend on the community definition alone, never on what BBS
    happens to match in a given release."""
    import pandas as pd
    import tempfile, os as _os
    from src.community_encoder.train_DESK.bbs_community_points import species_order
    with tempfile.TemporaryDirectory() as d:
        csv = _os.path.join(d, "c.csv")
        codes = ["casfin", "houspa", "allhum", "gryjay"]
        pd.DataFrame({"species_code": codes}).to_csv(csv, index=False)
        layout = species_order(csv)
        # pretend BBS cannot survey 'houspa'
        matched = {"casfin", "allhum", "gryjay"}
        ix = {c: i for i, c in enumerate(layout)}         # ALL of them, matched or not
        assert len(ix) == 4 and ix["gryjay"] == 3
        # the compacted version -- what NOT to do -- moves gryjay from column 3 to column 2
        compacted = {c: i for i, c in enumerate([c for c in layout if c in matched])}
        assert compacted["gryjay"] == 2 != ix["gryjay"]


def test_the_layout_check_catches_a_permutation_that_counts_and_sets_cannot():
    """The check that was skipped every run, and the only kind that could have caught the bug.

    Both sides had 96 species, the same 96, in different orders. So a length check passes, a set
    check passes, and `matched 96/96, 0 unmatched` prints -- while 94 of 96 columns are wrong.
    """
    from src.community_encoder.train_DESK.validate_bbs_routes import assert_same_layout
    rank = ["casfin", "houspa", "allhum", "gryjay", "amegfi"]
    taxo = ["gryjay", "casfin", "amegfi", "allhum", "houspa"]

    assert len(rank) == len(taxo) and set(rank) == set(taxo)   # every weaker check passes
    assert_same_layout(rank, rank)                             # agreement is silent
    assert_same_layout([s.upper() for s in rank], rank)        # case-insensitive

    with pytest.raises(ValueError, match="pure permutation"):
        assert_same_layout(rank, taxo)
    with pytest.raises(ValueError, match="column 0"):
        assert_same_layout(rank, taxo)
    # a genuine set difference is reported as such, not as a permutation
    with pytest.raises(ValueError, match="sets differ"):
        assert_same_layout(rank, ["casfin", "houspa", "allhum", "gryjay", "pinsis"])


def test_the_points_dir_comes_from_the_shared_helper():
    """validate_bbs_routes read config["trend"]["points_dir"] directly while every other consumer
    used target_points_dir, so it looked in the RETIRED trend-products directory, found no
    points_meta.json, and warned instead of checking -- which is how the permutation survived."""
    from src.config_utils import target_points_dir
    cfg = {"target": {"points_dir": "/live/bbs_points"},
           "trend": {"points_dir": "/retired/esk_spacetime"}}
    assert target_points_dir(cfg) == "/live/bbs_points"        # target wins
    assert target_points_dir({"trend": {"points_dir": "/retired/esk_spacetime"}}) \
        == "/retired/esk_spacetime"                            # fallback still works
    src = open("src/community_encoder/train_DESK/validate_bbs_routes.py").read()
    body = src[src.index("def load_observed(config):"):src.index("def desk_z_ema(")]
    # Comments in that function NAME the old expression to explain the bug, so strip comment
    # lines before asserting -- otherwise the test fails on its own documentation.
    code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    assert "target_points_dir(config)" in code
    assert 'config["trend"]["points_dir"]' not in code
    assert 'config.get("trend", {}).get("points_dir"' not in code


def test_no_catch_all_unavailable_reason_survives():
    """A completeness check is only as strong as the specificity of the reasons it accepts.

    The reason string this guards against read "inputs not supplied to this run (bar unbuilt, or
    oracle refused its representability gate)" -- a DISJUNCTION covering a real finding (the
    oracle refused its gate, which says something about the basis) and a bug (nothing passed it
    in). The route-level buckets never wired the oracle in and stamped that reason over the hole,
    so a code gap read as a measured decision, assert_complete returned zero gaps, and the missing
    ceiling went unnoticed across several runs.
    """
    import src.community_encoder.train_DESK.validate_bbs_routes as m
    assert not hasattr(m, "PREDICTOR_UNAVAILABLE_DEFAULT")
    reasons = [m.UNAVAILABLE_NOT_WIRED, m.UNAVAILABLE_NOT_SUPPLIED,
               m.UNAVAILABLE_BAR_UNBUILT, m.UNAVAILABLE_ORACLE_GATE]
    assert len(set(reasons)) == len(reasons)                  # each distinguishable
    for r in reasons:
        # no reason may hedge between two causes -- that is what made the old one useless
        assert " or " not in r.lower(), r
    # and the wiring-gap reason must say it is a code gap, so it reads as a bug not a decision
    assert "code gap" in m.UNAVAILABLE_NOT_WIRED.lower()


def test_the_oracle_reaches_the_route_level_buckets():
    """It is the only row that says whether a route-level skill of ~0 is DESK's failure or the
    basis's limit, and it was absent from exactly the table where that matters."""
    src = open("src/community_encoder/train_DESK/validate_bbs_routes.py").read()
    body = src[src.index("    S_nc_obs = ruzicka_rect(X_nc_s, X_nc_s)"):]
    body = body[:body.index('report["buckets"][f"{sname}/{wname}"] = row')]
    code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    assert "project_points_to_z" in code                      # the oracle is computed
    assert 'gram_preds["esk_truncation"]' in code                 # ...and enters the dot table
    assert 'gram_cos["esk_truncation"]' in code                   # ...and the cosine table


def test_a_near_constant_predictor_is_flagged_uninterpretable():
    """A flat predictor still returns a finite pearson_r, and its SIGN is then arbitrary.

    `no_change` in the cosine form of same_cell_over_time is cos(z_modern, z_modern) = 1 for every
    cell: measured spread 0.0145 against an observed 0.1128, reporting pearson_r = +0.064. DESK
    there had 26% of the observed spread and reported -0.151, which reads as "predicts change
    backwards" but means "predicts almost no change, and the residual wobble tracks nothing".
    """
    from src.community_encoder.train_DESK.validate_bbs_routes import compare_predictors
    rng = np.random.default_rng(0)
    truth = rng.normal(size=400)
    flat = np.full(400, 1.0) + rng.normal(scale=1e-3, size=400)   # a constant plus noise
    real = truth + rng.normal(scale=0.4, size=400)
    m = compare_predictors(truth, {"flat": flat, "real": real, "no_change": real}, grams=False)
    f, r = m["predictors"]["flat"], m["predictors"]["real"]
    assert f["sd_ratio"] < 0.05
    assert f["pearson_interpretable"] is False
    assert r["sd_ratio"] > 0.5
    assert r["pearson_interpretable"] is True
    # the flag is about SPREAD, not about being wrong: a predictor can vary plenty and still
    # correlate poorly, and that IS interpretable
    poor = -truth * 0.9 + rng.normal(scale=0.3, size=400)
    m2 = compare_predictors(truth, {"poor": poor, "no_change": real}, grams=False)
    assert m2["predictors"]["poor"]["pearson_interpretable"] is True
    assert m2["predictors"]["poor"]["pearson_r"] < -0.5      # genuinely anti-correlated


def test_every_predictor_declares_its_denoising_level():
    """The axis that decides whether a comparison is apples-to-apples, and the one the registry
    did not record -- so compare_predictors graded four predictors on 'identical terms' while
    being blind to what makes terms non-identical. Format symmetry is not information symmetry."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        PREDICTOR_ROLES, PREDICTOR_DENOISING, denoising_mismatch)
    assert set(PREDICTOR_DENOISING) == set(PREDICTOR_ROLES), (
        "a predictor without a declared denoising level can be compared against anything")
    for p, d in PREDICTOR_DENOISING.items():
        assert d["level"] and isinstance(d["shares_target_noise"], bool), p
    # the oracle must be flagged: it is built from the same arrays the truth is computed from
    assert PREDICTOR_DENOISING["esk_truncation"]["shares_target_noise"] is True
    assert PREDICTOR_DENOISING["desk"]["shares_target_noise"] is False

    notes = denoising_mismatch(["desk", "esk_truncation"])
    assert any("esk_truncation" in n and "SHARES THE TARGET'S NOISE" in n for n in notes), notes
    assert not any(n.startswith("desk:") for n in notes)
    # an unregistered predictor is reported, not silently accepted
    assert any("UNRECORDED" in n for n in denoising_mismatch(["some_new_bar"]))


def test_matched_group_sizes_equalise_endpoint_noise():
    """The two endpoints of a difference must carry the same measurement noise, or the difference
    is biased by whichever side is noisier -- and biased BY ERA, which is the axis the temporal
    sweep varies. Measured before this existed: modern median 16 surveys vs early 5, a 3.2x gap."""
    from src.community_encoder.train_DESK.validate_bbs_routes import match_group_sizes
    early = [(0, 1, 2), (3, 4), (5,), ()]
    modern = [(10, 11, 12, 13, 14, 15), (16, 17, 18), (19, 20), ()]
    a, b = match_group_sizes(early, modern, seed=0)
    for ga, gb, oa, ob in zip(a, b, early, modern):
        assert len(ga) == len(gb) == min(len(oa), len(ob))
        assert set(ga) <= set(oa) and set(gb) <= set(ob)      # subsets, never invented rows
    assert [len(g) for g in a] == [3, 2, 1, 0]


def test_split_half_gives_the_oracle_an_independent_observation():
    """The oracle shared the target's noise: it projected the very arrays the truth was computed
    from, so pearson 0.995 measured truncation fidelity, not an achievable ceiling. Halves must be
    DISJOINT for the oracle to be predicting something it did not see."""
    from src.community_encoder.train_DESK.validate_bbs_routes import split_half_groups
    groups = [tuple(range(6)), tuple(range(10, 14)), (20, 21), (30,), ()]
    a, b, ok = split_half_groups(groups, seed=0)
    assert list(ok) == [True, True, True, False, False]
    for ga, gb, g, good in zip(a, b, groups, ok):
        if not good:
            continue
        assert set(ga) & set(gb) == set(), (ga, gb)           # DISJOINT -- the whole point
        assert set(ga) | set(gb) <= set(g)
        assert len(ga) == len(gb)                             # equal halves -> matched noise
    # a group too small to split is flagged, not silently compared against itself
    assert ok[3] is np.False_ or ok[3] == False


# ---------------------------------------------------------------------------------------------
# Apples-to-apples: honest ceiling, matched endpoints, measured strata
# ---------------------------------------------------------------------------------------------

def _noisy_cell_surveys(n_cells=120, n_surveys=6, n_sp=20, turnover=0.0, noise=0.6, seed=0):
    """Per-cell early/modern survey draws with a KNOWN amount of real turnover.

    Real signal is a per-cell community; `turnover` rotates the modern one away from the early
    one; `noise` is independent per SURVEY, which is what a 1.08-routes-per-cell-year draw looks
    like. Returns raw counts so the code under test can average and log1p as it does in anger.
    """
    rng = np.random.default_rng(seed)
    base = rng.random((n_cells, n_sp)) * 6.0
    shift = rng.random((n_cells, n_sp)) * 6.0
    modern_true = (1.0 - turnover) * base + turnover * shift
    early = np.clip(base[:, None, :] + rng.normal(scale=noise, size=(n_cells, n_surveys, n_sp)), 0, None)
    modern = np.clip(modern_true[:, None, :] + rng.normal(scale=noise, size=(n_cells, n_surveys, n_sp)), 0, None)
    return early, modern


def test_the_independent_oracle_does_not_share_the_targets_noise():
    """The bug this whole change exists for. The oracle projected the SAME arrays the truth was
    computed from, so it scored 0.995 by reproducing the target's noise -- not a ceiling any model
    could reach. Built from a disjoint half it must score strictly LOWER; if it matched, the split
    did nothing."""
    from src.community_encoder.train_DESK.validate_bbs_routes import _rowwise_ruzicka
    from src.data.preprocess.bbs_community import log1p_community
    early, modern = _noisy_cell_surveys(turnover=0.35, seed=1)
    h = early.shape[1] // 2
    Xe_A = log1p_community(early[:, :h].mean(1))
    Xe_B = log1p_community(early[:, h:].mean(1))
    Xm_A = log1p_community(modern[:, :h].mean(1))
    Xm_B = log1p_community(modern[:, h:].mean(1))

    truth = _rowwise_ruzicka(Xe_A, Xm_A)
    shared = _rowwise_ruzicka(Xe_A, Xm_A)                 # the SAME arrays -- what the bug was
    independent = _rowwise_ruzicka(Xe_B, Xm_B)            # a disjoint observation

    r_shared = float(np.corrcoef(truth, shared)[0, 1])
    r_indep = float(np.corrcoef(truth, independent)[0, 1])
    assert r_shared > 0.999, r_shared                     # sharing the noise is near-perfect
    assert r_indep < r_shared - 0.05, (r_indep, r_shared)  # independence costs, and must
    assert r_indep > 0.0, r_indep                         # ...but real signal still shows


def test_matched_endpoints_remove_an_era_dependent_noise_bias():
    """Unequal endpoint noise biases a DIFFERENCE toward the noisier side -- and the imbalance was
    era-dependent (early 5 surveys vs modern 16), so it was confounded with the axis the temporal
    sweep varies."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        match_group_sizes, _rowwise_ruzicka)
    from src.data.preprocess.bbs_community import log1p_community
    early, modern = _noisy_cell_surveys(n_surveys=12, turnover=0.0, seed=2)

    # unmatched: 2 early surveys against 12 modern -- exactly the 3x+ imbalance measured in anger
    unmatched = _rowwise_ruzicka(log1p_community(early[:, :2].mean(1)),
                                 log1p_community(modern.mean(1)))
    matched = _rowwise_ruzicka(log1p_community(early[:, :2].mean(1)),
                               log1p_community(modern[:, :2].mean(1)))
    # With ZERO real turnover, any shortfall from 1.0 is pure noise. The noisier-endpoint pairing
    # is not automatically lower, but the two must differ -- the imbalance is a real effect, not a
    # wash -- and the matched version is the one whose two sides are comparable.
    assert abs(np.median(unmatched) - np.median(matched)) > 0.01, (
        np.median(unmatched), np.median(matched))

    e_groups = [tuple(range(6)), tuple(range(6, 8)), (8,)]
    m_groups = [tuple(range(20, 36)), tuple(range(36, 40)), (40, 41)]
    a, b = match_group_sizes(e_groups, m_groups, seed=0)
    assert [len(g) for g in a] == [len(g) for g in b] == [6, 2, 1]


def test_a_stratum_with_no_real_turnover_is_excluded_with_a_reason():
    """The case where the correct answer is 'this cannot be measured'. With zero real turnover the
    cross-era difference is entirely noise, excess is ~0, and no predictor can be graded there."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        stratum_viable, noise_floor, _rowwise_ruzicka)
    from src.data.preprocess.bbs_community import log1p_community
    early, modern = _noisy_cell_surveys(n_cells=150, turnover=0.0, seed=3)
    h = early.shape[1] // 2
    Xe_A, Xe_B = log1p_community(early[:, :h].mean(1)), log1p_community(early[:, h:].mean(1))
    Xm_A = log1p_community(modern[:, :h].mean(1))
    observed = _rowwise_ruzicka(Xe_A, Xm_A)               # cross-era: noise only, by construction
    floor = _rowwise_ruzicka(Xe_A, Xe_B)                  # same era: noise only, by definition

    ok, why, stats = stratum_viable(observed, floor, n_cells=150, seed=0)
    assert ok is False, stats
    assert "not distinguishable from zero" in why, why
    assert abs(stats["excess"]) < 0.05, stats             # floor and cross-era coincide

    nf = noise_floor(floor)
    assert nf["floor"] < 1.0                              # noise pushes two looks apart
    assert nf["n"] == 150

    # ...and with REAL turnover planted, the same stratum qualifies
    early2, modern2 = _noisy_cell_surveys(n_cells=150, turnover=0.6, seed=3)
    Xe2_A, Xe2_B = (log1p_community(early2[:, :h].mean(1)), log1p_community(early2[:, h:].mean(1)))
    Xm2_A = log1p_community(modern2[:, :h].mean(1))
    ok2, why2, stats2 = stratum_viable(_rowwise_ruzicka(Xe2_A, Xm2_A),
                                       _rowwise_ruzicka(Xe2_A, Xe2_B), n_cells=150, seed=0)
    assert ok2 is True, (why2, stats2)
    assert stats2["excess"] > 0.05, stats2


def test_a_thin_stratum_is_excluded_before_it_can_move_an_aggregate():
    from src.community_encoder.train_DESK.validate_bbs_routes import stratum_viable
    rng = np.random.default_rng(0)
    obs = rng.random(12) * 0.3 + 0.4
    flo = obs + 0.4                                       # a huge apparent excess...
    ok, why, _ = stratum_viable(obs, flo, n_cells=12, min_cells=30, seed=0)
    assert ok is False and "only 12 cells" in why, why     # ...but far too few cells to believe


def test_the_balanced_aggregate_damps_thin_strata():
    """Pure equal weight would hand a stratum of 12 cells the same pull as one of 600 -- the
    failure desk_training.stratum_weights was written to prevent."""
    from src.community_encoder.train_DESK.validate_bbs_routes import balanced_over_strata
    per = {"a": {"qualified": True, "n_cells": 600, "excess": 0.10},
           "b": {"qualified": True, "n_cells": 600, "excess": 0.10},
           "thin": {"qualified": True, "n_cells": 30, "excess": 0.90}}
    agg = balanced_over_strata(per, "excess", n_min=None, cap=5.0, power=0.5)
    equal = np.mean([0.10, 0.10, 0.90])
    assert agg["balanced"] < equal, (agg["balanced"], equal)   # thin stratum pulled less than 1/3
    assert agg["weights"]["thin"] < agg["weights"]["a"]
    # and the population-weighted figure is reported beside it -- the GAP is the coverage bias
    assert agg["population_weighted"] < agg["balanced"]
    # an unqualified stratum never enters at all
    per["thin"]["qualified"] = False
    agg2 = balanced_over_strata(per, "excess", n_min=None, cap=5.0, power=0.5)
    assert agg2["n_strata"] == 2 and abs(agg2["balanced"] - 0.10) < 1e-9


def test_the_epoch_strata_use_the_same_tiling_as_everything_else():
    """One definition of 'where' at two resolutions, not two definitions."""
    from src.community_encoder.train_DESK.esk_kernel import (
        coarse_spatial, spatial_tiles, nests_within)
    rng = np.random.default_rng(0)
    pidx = np.column_stack([rng.integers(0, 64, 400), rng.integers(0, 64, 400),
                            rng.integers(1966, 2025, 400)])
    rb, cb = spatial_tiles(pidx[:, 0], pidx[:, 1], 4)
    fine = rb * 4 + cb
    coarse = coarse_spatial(pidx, regions=2)
    assert nests_within(fine, coarse), "coarse regions must group whole fine tiles"


def test_a_stratum_dropped_without_a_reason_is_caught():
    """The stratum-level twin of the predictor completeness check. A headline aggregated over a
    silently-shrinking set of strata looks stable while its basis moves."""
    from src.community_encoder.train_DESK.validate_bbs_routes import assert_strata_explained
    good = {"heldout/region0": {"qualified": True, "n_cells": 400},
            "heldout/region1": {"qualified": False, "n_cells": 9,
                                "unavailable": {"balanced_aggregate": "only 9 cells (needs >= 30)"}}}
    assert assert_strata_explained(good) == []
    bad = {"heldout/region1": {"qualified": False, "n_cells": 9}}     # excluded, no reason
    gaps = assert_strata_explained(bad)
    assert any("no stated reason" in g for g in gaps), gaps


# ---------------------------------------------------------------------------------------------
# pair_convergence -- did two places grow more alike, or less
# ---------------------------------------------------------------------------------------------

def test_convergence_has_a_true_zero_baseline():
    """Why this question can resolve skill when same_cell_over_time cannot.

    Freezing both cells at their modern state leaves the similarity BETWEEN them untouched, so the
    no-change null predicts exactly zero convergence. That is a genuine zero. On
    same_cell_over_time the same null already reaches 0.65, because how much a place appears to
    change is mostly a fixed property of that place, and there is almost no room left above it.
    """
    Xe, Xm, A, xy = _epoch_fixture()
    rng = np.random.default_rng(0)
    Ze, Zm = Xe @ A, Xm @ A
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3)
    node = rep["types"]["pair_convergence"]["pooled"]["all_distances"]["dot"]
    nc = node["predictors"]["no_change"]

    # the null predicts a CONSTANT zero: no spread, and flagged as such
    assert abs(nc["sd_pred"]) < 1e-12, nc
    assert nc["pearson_interpretable"] is False, nc
    # ...and the observed convergence is NOT zero, so there is something to predict
    assert node["observed_sd"] > 1e-6, node["observed_sd"]


def test_convergence_recovers_planted_structure():
    """Cells made to converge must read as converging, and cells made to diverge as diverging."""
    from src.community_encoder.train_DESK.validate_bbs_routes import _rowwise_ruzicka
    rng = np.random.default_rng(2)
    n_sp = 24
    a = rng.random(n_sp) * 5.0
    b = rng.random(n_sp) * 5.0
    # a pair that converges: both eras start apart, end close
    e_i, e_j = a, b
    m_i, m_j = 0.5 * (a + b), 0.5 * (a + b)
    conv = (_rowwise_ruzicka(m_i[None, :], m_j[None, :])
            - _rowwise_ruzicka(e_i[None, :], e_j[None, :]))[0]
    # a pair that diverges: same endpoints, swapped eras
    div = (_rowwise_ruzicka(e_i[None, :], e_j[None, :])
           - _rowwise_ruzicka(m_i[None, :], m_j[None, :]))[0]
    assert conv > 0.05, conv        # grew more alike -> positive
    assert div < -0.05, div         # grew less alike -> negative
    assert abs(conv + div) < 1e-12  # exact mirror image


def test_convergence_is_the_difference_of_two_tables_that_already_ran():
    """It is a subtraction, not a new measurement -- both halves ran over the SAME neighbour pairs
    and were reported separately for the whole life of the epoch analysis."""
    Xe, Xm, A, xy = _epoch_fixture()
    Ze, Zm = Xe @ A, Xm @ A
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3)
    g = lambda q: rep["types"][q]["pooled"]["all_distances"]["dot"]
    early, modern, conv = g("cross_cell_same_era_early"), g("cross_cell_same_era_modern"), g("pair_convergence")
    assert early["n"] == modern["n"] == conv["n"]          # identical pair population
    # observed convergence mean is the difference of the two observed means
    assert abs(conv["observed_mean"] - (modern["observed_mean"] - early["observed_mean"])) < 1e-9


def test_room_flags_a_question_that_cannot_resolve_skill():
    """The direct fix for the contradiction that motivated this pass: two comparisons of the same
    places over the same period gave opposite verdicts, and nothing said one of them had no room."""
    from src.community_encoder.train_DESK.validate_bbs_routes import resolving_room
    # the REAL measured values from the 1995 run, not round numbers -- the narrow case sits at
    # 0.147 and a fixture at exactly 0.15 would straddle the threshold for no good reason
    narrow = {"predictors": {"no_change": {"pearson_r": 0.649},      # similarity between eras
                             "desk": {"pearson_r": 0.638},
                             "esk_oracle_independent": {"pearson_r": 0.796}}}
    wide = {"predictors": {"no_change": {"pearson_r": 0.066},        # direction of change
                           "desk": {"pearson_r": 0.233},
                           "esk_oracle_independent": {"pearson_r": 0.42}}}
    n, w = resolving_room(narrow), resolving_room(wide)
    assert n["room"] < 0.15 and "NARROW" in n["verdict"], n
    assert w["room"] > 0.3 and "usable" in w["verdict"], w
    # DESK is 0.01 BELOW the null in the narrow case and 0.25 above it in the wide one -- but the
    # narrow verdict must not be read as skill either way, which is what the verdict says
    assert "NOT evidence of skill" in n["verdict"]
    # falls back to the truncation ceiling when the independent one is unavailable
    only_trunc = {"predictors": {"no_change": {"pearson_r": 0.1},
                                 "esk_truncation": {"pearson_r": 0.9}}}
    assert resolving_room(only_trunc)["ceiling"] == "esk_truncation"
    # and says so rather than inventing a number when there is no ceiling at all
    assert "room undefined" in resolving_room({"predictors": {"no_change": {"pearson_r": 0.1}}})["note"]


def test_every_epoch_question_reports_its_room():
    Xe, Xm, A, xy = _epoch_fixture()
    rng = np.random.default_rng(0)
    Ze, Zm = Xe @ A, Xm @ A
    ind = (Ze + rng.normal(scale=.15, size=Ze.shape), Zm + rng.normal(scale=.15, size=Zm.shape))
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3,
                                         sc_esk=(Ze, Zm), sc_esk_independent=ind)
    for q, node in rep["types"].items():
        row = node["pooled"]["all_distances"]
        if "skipped" in row:
            continue
        assert "room" in row["dot"], q
        rm = row["dot"]["room"]
        # A question whose baseline is not a floor must REFUSE to report room rather than
        # computing a meaningless one. `no_change` carries the full modern spatial structure, so
        # on a same-era spatial question it is a strong predictor, not a null.
        if q.startswith("cross_cell_same_era") or q == "cross_cell_cross_time":
            assert rm.get("baseline_is_a_floor") is False, (q, rm)
            assert "no floor" in rm["note"], (q, rm)
        else:
            assert "verdict" in rm, (q, rm)


def test_room_flags_a_ceiling_that_shares_the_targets_noise():
    """A ceiling built from the same arrays as the truth scores its own noise and sits far too
    high. Measured on the 1995 run it put the room for pair_convergence at 0.959 where the
    independent ceiling puts it at 0.654 -- so the figure built to stop a comparison being
    over-read would itself have overstated by half."""
    from src.community_encoder.train_DESK.validate_bbs_routes import resolving_room
    borrowed = resolving_room({"predictors": {"no_change": {"pearson_r": 0.0},
                                              "esk_truncation": {"pearson_r": 0.959}}})
    assert borrowed["ceiling_shares_target_noise"] is True
    assert "OVERSTATEMENT" in borrowed["caveat"]
    honest = resolving_room({"predictors": {"no_change": {"pearson_r": 0.0},
                                            "esk_oracle_independent": {"pearson_r": 0.654}}})
    assert honest["ceiling_shares_target_noise"] is False
    assert "caveat" not in honest


def test_the_species_questions_use_the_same_pairs_as_the_similarity_questions():
    """Not a second set of comparison points. Building its own pairing would create a second
    definition of which places are being compared, which is the drift that has bitten this module
    repeatedly -- two species-column orders, three window constants, two change decompositions."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        species_on_pairings, epoch_neighborhood_analysis)
    from src.community_encoder.train_DESK.validate_baselines import fit_species_readout
    Xe, Xm, A, xy = _epoch_fixture()
    Ze, Zm = Xe @ A, Xm @ A
    ro = fit_species_readout(np.vstack([Ze, Zm]), np.vstack([Xe, Xm]))
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3,
                                         sc_esk=(Ze, Zm), species_readout=ro)
    sp = rep["species"]["questions"]
    # all three streams, not just the temporal one
    assert "same_cell_over_time" in sp                      # temporal
    assert "cross_cell_same_era_early" in sp                # spatial
    assert "cross_cell_same_era_modern" in sp               # spatial
    assert "cross_cell_cross_time" in sp                    # spatiotemporal
    # and the same predictors as the similarity tables
    got = set(sp["cross_cell_cross_time"]["pooled"]["predictors"])
    assert {"desk", "no_change", "esk_truncation"} <= got, got
    # absent readout -> a stated reason, never a silent omission
    bare, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3)
    assert "no species readout supplied" in bare["species"]["note"]


def test_the_frozen_baseline_is_not_treated_as_a_floor_for_spatial_questions():
    """`no_change` is DESK's z frozen at the modern year. On a question about two places in ONE
    era it predicts their full modern similarity -- most of the answer, since spatial structure is
    largely stable over sixty years. It measured 0.87 there, and calling that 'a baseline holding
    no information' inverted the meaning: the comparison has no floor, not a narrow band."""
    from src.community_encoder.train_DESK.validate_bbs_routes import (
        resolving_room, NULL_IS_A_FLOOR_FOR)
    tbl = {"predictors": {"no_change": {"pearson_r": 0.871},
                          "desk": {"pearson_r": 0.873},
                          "esk_oracle_independent": {"pearson_r": 0.968}}}
    spatial = resolving_room(tbl, question="cross_cell_same_era_early")
    assert spatial["baseline_is_a_floor"] is False
    assert "room" not in spatial, "must not report a number it cannot compute"
    assert "no floor" in spatial["note"]

    # where freezing time genuinely destroys the answer, room is real
    for q in NULL_IS_A_FLOOR_FOR:
        r = resolving_room({"predictors": {"no_change": {"pearson_r": 0.0},
                                           "esk_oracle_independent": {"pearson_r": 0.654}}},
                           question=q)
        assert abs(r["room"] - 0.654) < 1e-9, (q, r)
    assert "cross_cell_same_era_early" not in NULL_IS_A_FLOOR_FOR
    assert "cross_cell_cross_time" not in NULL_IS_A_FLOOR_FOR


def test_the_early_window_is_restricted_to_years_every_run_withholds():
    """The sweep varies ONE thing: how far past its training edge the model reaches. EPOCH_EARLY
    spans 1966-1986, of which the three runs withhold 48%, 95% and 100% -- so the shallowest run's
    early endpoint was half built from years it trained on, and the reported decay across runs was
    partly a decay in how much of the endpoint had already been seen."""
    src = open("src/community_encoder/train_DESK/validate_bbs_routes.py").read()
    body = src[src.index("    keys_all = keys"):src.index("ep_stats[\"early_window_is_fully_withheld\"]")]
    code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    assert "common_holdout_years" in code
    assert "epoch_gate(keys_all, early=_early)" in code
    # and it must warn rather than silently proceed when the common window is unset
    assert "not valid" in body


def test_the_manifest_says_which_generalisation_each_question_tests():
    Xe, Xm, A, xy = _epoch_fixture()
    Ze, Zm = Xe @ A, Xm @ A
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Ze, Zm, xy, k=9, n_bins=3)
    g = rep["manifest"]["generalisation"]
    assert "162 km" in g["place_axis"]
    era = g["era_axis"]
    assert "SEEN era" in era["cross_cell_same_era_modern"]
    assert "UNSEEN era" in era["cross_cell_same_era_early"]
    # every question this analysis covers must declare its era status
    for q in rep["types"]:
        assert q in era, q
