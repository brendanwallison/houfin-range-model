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

from src.community_encoder.train_DESK.validate_bbs_routes import (
    EPOCH_EARLY, EPOCH_MODERN, bucket_metrics, cosine_gram, densify_community, dot_gram,
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
    m = bucket_metrics(ruzicka_rect(X, X), S_nc, S_nc)
    assert m["cka_gain"] == 0.0, m
    assert m["mantel_gain"] == 0.0, m
    assert m["rmse_skill"] == 0.0, m               # identical kernels -> no error reduction
    assert m["rmse_desk"] == m["rmse_nochange"], m


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

    m = bucket_metrics(S_true, dot_gram(X @ A), S_nc)
    assert m["cka_gain"] > 0.0, m
    assert m["cka_desk"] > m["cka_nochange"], m


def test_observed_space_null_is_reported_but_not_differenced():
    # cka_nochange_observed uses Ruzicka (truth's functional) and so is NOT comparable to
    # cka_desk; it must appear in the output without contaminating cka_gain.
    keys, X = _synthetic()
    nc_src, _ = modern_reference_rows(keys, modern_window=(2010, 2025))
    X_nc = X[nc_src]
    A = np.random.default_rng(4).standard_normal((X.shape[1], 6))
    S_true = ruzicka_rect(X, X)

    m = bucket_metrics(S_true, dot_gram(X @ A), dot_gram(X_nc @ A),
                       S_nc_obs=ruzicka_rect(X_nc, X_nc),
                       S_desk_cos=cosine_gram(X @ A), S_nc_cos=cosine_gram(X_nc @ A))
    assert "cka_nochange_observed" in m and "rmse_nochange_observed" in m
    assert m["cka_gain"] == m["cka_desk"] - m["cka_nochange"]      # unaffected by the diagnostic
    # the cosine variant is reported SEPARATELY and must not overwrite the dot-product headline
    assert "cka_gain_cosine" in m and m["cka_gain_cosine"] != m["cka_gain"]


def test_cka_of_truth_against_itself_is_one():
    _, X = _synthetic()
    S = ruzicka_rect(X, X)
    m = bucket_metrics(S, S, S)
    assert abs(m["cka_desk"] - 1.0) < 1e-8
    assert abs(m["cka_gain"]) < 1e-8                           # identical inputs -> no gain


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

    m = bucket_metrics(S, S + noise_d, S + noise_n)
    ratio = m["rmse_desk"] / m["rmse_nochange"]
    assert abs(m["error_variance_removed"] - (1 - ratio ** 2)) < 1e-12
    assert abs(m["rmse_skill"] - (1 - ratio)) < 1e-12
    # r2_gain == (rmse_nc^2 - rmse_desk^2) / sd^2
    expect = (m["rmse_nochange"] ** 2 - m["rmse_desk"] ** 2) / m["observed_sd"] ** 2
    assert abs(m["r2_gain"] - expect) < 1e-9


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
    m = bucket_metrics(S, S + 0.05, S + 0.09)
    assert abs(m["calibration_loss_desk"] - (m["pearson_desk"] ** 2 - m["r2_desk"])) < 1e-12
    assert m["calibration_loss_desk"] > 0                     # ranking better than accuracy


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
    cfg = {"bbs": {"z_dir": str(tmp_path)}, "paths": {"desk_output_dir": str(desk)}}

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
    assert rep["buckets"]["pooled/all"]["cka_gain"] > 0, rep["buckets"]["pooled/all"]
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


def test_spatial_modern_skill_is_exactly_zero_the_builtin_null_test():
    # spatial_modern grades Zm.Zm against a null that IS Zm.Zm. Its skill must be exactly 0.
    # This is the analysis's own null test: if it ever moves, the harness mis-pairs its inputs
    # and no other row in the table can be trusted.
    Xe, Xm, A, xy = _epoch_fixture()
    rep, per_cell = epoch_neighborhood_analysis(Xe, Xm, Xe @ A, Xm @ A, xy, k=9, n_bins=3)

    for split, per_bin in rep["types"]["spatial_modern"].items():
        for bname, mm in per_bin.items():
            if "skipped" in mm:
                continue
            assert mm["rmse_skill"] == 0.0, (split, bname, mm)
            assert mm["rmse_desk"] == mm["rmse_null"], (split, bname, mm)
            assert mm["r2_gain"] == 0.0, (split, bname, mm)
    assert np.allclose(per_cell["spatial_modern_skill"], 0.0)


def test_epoch_analysis_shape_and_that_a_tracking_desk_beats_the_null():
    Xe, Xm, A, xy = _epoch_fixture()
    rep, per_cell = epoch_neighborhood_analysis(Xe, Xm, Xe @ A, Xm @ A, xy, k=9, n_bins=3)

    assert set(rep["types"]) == {"spatial_early", "spatial_modern", "cross_time", "self_change"}
    assert rep["config"]["k"] == 9 and rep["config"]["n_focal_cells"] == 40
    # a DESK that tracks the truth must beat the frozen-modern null where time matters
    assert rep["types"]["spatial_early"]["pooled"]["all_distances"]["rmse_skill"] > 0
    assert rep["types"]["self_change"]["pooled"]["all_distances"]["n_pairs"] == 40
    # per-cell fields are present and one value per focal cell, for mapping
    for key in ("spatial_early_skill", "cross_time_skill", "self_change_obs", "neighbour_dist_m"):
        assert key in per_cell
    assert per_cell["spatial_early_skill"].shape == (40,)
    assert per_cell["neighbour_dist_m"].shape == (40, 9)


def test_epoch_analysis_splits_by_holdout_when_given_a_mask():
    Xe, Xm, A, xy = _epoch_fixture()
    ho = np.zeros(40, bool); ho[::2] = True
    rep, _ = epoch_neighborhood_analysis(Xe, Xm, Xe @ A, Xm @ A, xy, k=9, n_bins=3, is_heldout=ho)
    assert set(rep["types"]["cross_time"]) == {"pooled", "train", "heldout"}
    n_ho = rep["types"]["self_change"]["heldout"]["all_distances"]["n_pairs"]
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
