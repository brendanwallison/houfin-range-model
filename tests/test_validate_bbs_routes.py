"""Route-level BBS validation: densification, the no-change gate, and the metric's SIGN.

The sign tests are the important ones. This validation's whole claim is that
``cka_gain = CKA(true, desk) - CKA(true, nochange)`` isolates temporal skill, so a
constant-per-cell DESK must score ~0 and a DESK carrying real temporal information must score
> 0. If those two ever stop holding, the headline number means nothing.
"""
import numpy as np

from src.community_encoder.train_DESK.validate_bbs_routes import (
    bucket_metrics, cosine_gram, densify_community, log1p_community,
    modern_reference_rows, stratified_sample)
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
    S_nc = cosine_gram(Z_nc)
    m = bucket_metrics(ruzicka_rect(X, X), S_nc, S_nc)
    assert m["cka_gain"] == 0.0, m
    assert m["mantel_gain"] == 0.0, m


def test_cka_gain_is_positive_only_when_temporal_information_is_added():
    # Controlled contrast: ONE linear map A, so spatial fidelity is identical between the two
    # models and the only difference is whether z varies over time.
    #
    #   Z_nc   = X_nc @ A  -> the null: modern community, frozen
    #   Z_good = X    @ A  -> same map, but tracks the true year-by-year community
    #
    # Both are scored with cosine_gram, matching how run() builds them. Scoring the null with
    # Ruzicka instead (truth's own functional) made a temporally-neutral model read -0.28, which
    # is the confound this arrangement removes.
    keys, X = _synthetic()
    nc_src, keep = modern_reference_rows(keys, modern_window=(2010, 2025))
    assert keep.all()

    A = np.random.default_rng(2).standard_normal((X.shape[1], 6))
    S_true = ruzicka_rect(X, X)
    S_nc = cosine_gram(X[nc_src] @ A)

    m = bucket_metrics(S_true, cosine_gram(X @ A), S_nc)
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

    m = bucket_metrics(S_true, cosine_gram(X @ A), cosine_gram(X_nc @ A),
                       S_nc_obs=ruzicka_rect(X_nc, X_nc))
    assert "cka_nochange_observed" in m and "mantel_nochange_observed" in m
    assert m["cka_gain"] == m["cka_desk"] - m["cka_nochange"]      # unaffected by the diagnostic


def test_cka_of_truth_against_itself_is_one():
    _, X = _synthetic()
    S = ruzicka_rect(X, X)
    m = bucket_metrics(S, S, S)
    assert abs(m["cka_desk"] - 1.0) < 1e-8
    assert abs(m["cka_gain"]) < 1e-8                           # identical inputs -> no gain


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


def test_stratified_sample_returns_all_rows_when_under_budget():
    keys = np.array([[0, 0, 1970], [1, 1, 2015]], dtype="int32")
    sel = stratified_sample(keys, n_sample=100, rng=np.random.default_rng(0))
    assert sel.tolist() == [0, 1]


# ----------------------------- driver wiring -----------------------------

def test_run_end_to_end_gates_cells_and_buckets(tmp_path, monkeypatch):
    """Exercises run()'s gate → reindex → sample → bucket wiring on a synthetic npz.

    The pure functions above are unit-tested; this covers the parts only the driver does --
    notably remapping ``nc_src`` into the post-gate row indices, which is silent if wrong (it
    would pair rows with the wrong cell's modern vector and quietly deflate the gain).
    """
    from src.community_encoder.train_DESK import validate_bbs_routes as V

    n_species, years = 6, [1970, 1975, 2000, 2012, 2020]
    rng = np.random.default_rng(0)
    row, col, yr, si, mc, cr, cc, cy = [], [], [], [], [], [], [], []
    for r in range(20):
        # cell 19 is surveyed ONLY in the early window -> no modern reference -> fully gated out
        for y in ([1970, 1975] if r == 19 else years):
            cr.append(r); cc.append(r); cy.append(y)
            for s in range(n_species):
                v = float(rng.random() * 5 + (y - 1970) * 0.05 * (s + 1))
                if v > 0.4:
                    row.append(r); col.append(r); yr.append(y); si.append(s); mc.append(v)

    cm = tmp_path / "community_matrix.npz"
    np.savez_compressed(
        cm, row=np.array(row, np.int32), col=np.array(col, np.int32), year=np.array(yr, np.int32),
        species_index=np.array(si, np.int32), mean_count=np.array(mc, np.float32),
        cov_row=np.array(cr, np.int32), cov_col=np.array(cc, np.int32),
        cov_year=np.array(cy, np.int32), cov_n=np.ones(len(cr), np.int32),
        species_codes=np.array([f"sp{i}" for i in range(n_species)], dtype=object),
        dims=np.array([25, 25], np.int32))
    desk = tmp_path / "desk"
    desk.mkdir()
    ho = np.zeros((25, 25), bool)
    ho[::3] = True                                             # ~1/3 of cells held out
    np.save(desk / "holdout_cells.npy", ho)

    cfg = {"bbs": {"community_matrix": str(cm), "z_dir": str(tmp_path)},
           "paths": {"desk_output_dir": str(desk)}}

    # Stub DESK: z is a linear map of the TRUE community, so it genuinely tracks time and the
    # gain must come out positive. Keyed by (cell, year) so the stub cannot accidentally leak
    # information the real encoder would not have.
    X_log, keys, _ = V.load_observed(cfg)
    lut = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(keys)}
    A = np.random.default_rng(1).standard_normal((X_log.shape[1], 5))

    def fake_z(config, k):
        Z = np.stack([X_log[lut[(int(r), int(c), int(y))]] @ A for r, c, y in k])
        return Z.astype("float32"), {"output_ema_applied": True, "ema_half_life": 10.0,
                                     "ema_warmup_start": 1940, "encode_years": [1940, 2020]}
    monkeypatch.setattr(V, "desk_z_ema", fake_z)

    rep = V.run(config=cfg, n_sample=200, seed=0)

    assert rep["site_gate"]["cells_total"] == 20
    assert rep["site_gate"]["cells_dropped_no_modern"] == 1     # exactly cell 19
    assert rep["site_gate"]["cells_kept"] == 19
    # every split x window bucket present, and the truth-tracking stub beats the frozen null
    for name in ("pooled/all", "train/all", "heldout/all",
                 "pooled/modern", "pooled/early", "heldout/early"):
        assert name in rep["buckets"], name
    assert rep["buckets"]["pooled/all"]["cka_gain"] > 0, rep["buckets"]["pooled/all"]
    assert (desk / "bbs_route_validation.json").exists()
    assert (desk / "bbs_route_validation.npz").exists()
