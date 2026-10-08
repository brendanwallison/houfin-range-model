"""GP species validation: the low-rank GP algebra, the metrics' sign, and the species layout.

The algebra tests are what entitle ``gp_kernels`` to skip the n x n covariance: the low-rank REML
likelihood and predictive must agree with the dense GP to float precision. The planted tests are
what entitle the skill numbers to a sign: a species driven by z's temporal change must score > 0
on change, and one driven by nothing z knows must not.
"""
import numpy as np
import pytest

from src.community_encoder.train_DESK import gp_kernels as gpk
from src.community_encoder.train_DESK.validate_gp_species import (
    block_sums, gaussian_crps, interval_coverage, modern_reference_keys,
    no_change_z, pooled_skill, skill_from_sse, species_layout)
from src.community_encoder.train_DESK.validation_core import row_splits


# ----------------------------- algebra -----------------------------

def _toy(n=60, r=5, k=3, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, r)) + 0.3
    W = rng.normal(size=(r, k))
    Y = 1.5 + Z @ W + 0.4 * rng.normal(size=(n, k))
    return Z, Y


def test_lowrank_likelihood_matches_dense_up_to_a_constant():
    # The low-rank form drops (s2, n2)-independent constants, so the DIFFERENCE from the dense
    # REML must be the same at every hyperparameter setting. A wrong term would make it drift.
    Z, Y = _toy()
    b = gpk.feature_svd(Z)
    st = gpk.sufficient_stats(b, Z, Y[:, :1])
    diffs = []
    for s2, n2 in [(0.1, 0.2), (1.0, 0.05), (3.0, 1.0), (0.01, 2.0)]:
        lr = gpk.neg_log_marginal(np.log([s2]), np.log([n2]), b, st)[0][0]
        dn = gpk.dense_reference(Z, Y[:, 0], s2, n2, Z[:2])[0]
        diffs.append(dn - lr)
    assert np.ptp(diffs) < 1e-8


def test_lowrank_gradient_matches_finite_differences():
    Z, Y = _toy()
    b = gpk.feature_svd(Z)
    st = gpk.sufficient_stats(b, Z, Y)
    a, c = np.log([0.5, 1.2, 0.2]), np.log([0.3, 0.1, 0.9])
    _, gs, gn = gpk.neg_log_marginal(a, c, b, st)
    h = 1e-6
    f = lambda a_, c_: gpk.neg_log_marginal(a_, c_, b, st)[0]
    fd_s = (f(a + h, c) - f(a - h, c)) / (2 * h)
    fd_n = (f(a, c + h) - f(a, c - h)) / (2 * h)
    np.testing.assert_allclose(gs, fd_s, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(gn, fd_n, rtol=1e-5, atol=1e-6)


def test_lowrank_predictive_matches_dense():
    Z, Y = _toy()
    Zn = np.random.default_rng(1).normal(size=(7, Z.shape[1]))
    m = gpk.fit(Z, Y[:, :1])
    mean, var = gpk.predict(m, Zn)
    _, dmean, dvar = gpk.dense_reference(Z, Y[:, 0], m["s2"][0], m["n2"][0], Zn)
    np.testing.assert_allclose(mean[:, 0], dmean, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(var[:, 0], dvar, rtol=1e-8, atol=1e-8)


def test_fit_recovers_noise_and_is_per_species():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(4000, 8))
    Y = np.stack([Z @ rng.normal(size=8) + 0.1 * rng.normal(size=4000),
                  Z @ rng.normal(size=8) + 1.0 * rng.normal(size=4000)], 1)
    m = gpk.fit(Z, Y)
    np.testing.assert_allclose(np.sqrt(m["n2"]), [0.1, 1.0], rtol=0.05)


def test_species_never_detected_in_training_is_floored_not_fitted():
    Z, Y = _toy()
    Y = np.hstack([Y, np.zeros((len(Y), 1))])
    m = gpk.fit(Z, Y)
    assert not m["ok"][-1] and m["ok"][:-1].all()
    mean, var = gpk.predict(m, Z[:3])
    assert np.allclose(mean[:, -1], 0.0) and np.all(np.isfinite(var))


# ----------------------------- planted sign tests -----------------------------

def _planted(seed=0, n_cells=300, years=range(1970, 2021, 5), r=6, temporal=True):
    """Cells x years with z drifting over time; one species is a linear readout of z (what DESK
    can know), one is a cell-level random effect plus its own random temporal drift (what it
    cannot)."""
    rng = np.random.default_rng(seed)
    years = list(years)
    base = rng.normal(size=(n_cells, r))
    drift = rng.normal(size=(n_cells, r)) * (1.0 if temporal else 0.0)
    rows, Z = [], []
    for c in range(n_cells):
        for t, y in enumerate(years):
            frac = t / (len(years) - 1)
            rows.append((c // 20, c % 20, y))
            Z.append(base[c] + frac * drift[c])
    keys, Z = np.array(rows), np.array(Z)
    w = rng.normal(size=r)
    cell_eff = rng.normal(size=n_cells)
    cell_drift = rng.normal(size=n_cells)
    frac = np.tile(np.linspace(0, 1, len(years)), n_cells)
    y_z = Z @ w + 0.2 * rng.normal(size=len(Z))
    y_blind = (np.repeat(cell_eff, len(years)) + frac * np.repeat(cell_drift, len(years))
               + 0.2 * rng.normal(size=len(Z)))
    return keys, Z, np.stack([y_z, y_blind], 1)


def epoch_change(v, e_rows, m_rows):
    """Plain epoch means, for the GP-algebra sign tests (their synthetic y is not a count)."""
    v = np.asarray(v, "float64")
    return (np.stack([v[np.asarray(r)].mean(0) for r in e_rows]),
            np.stack([v[np.asarray(r)].mean(0) for r in m_rows]))


def _change_skill(keys, Z, Y, test):
    m = gpk.fit(Z[~test], Y[~test])
    tk = keys[test]
    early = tk[:, 2] <= 1985
    modern = tk[:, 2] >= 2005
    cells = np.unique(tk[:, :2], axis=0)
    e_rows = [np.where((tk[:, 0] == a) & (tk[:, 1] == b) & early)[0] for a, b in cells]
    m_rows = [np.where((tk[:, 0] == a) & (tk[:, 1] == b) & modern)[0] for a, b in cells]
    # no-change: the cell's modern-epoch mean z for every year
    ref = np.stack([Z[test][r].mean(0) for r in m_rows])
    cell_of = {tuple(c): i for i, c in enumerate(cells)}
    Z_nc = ref[[cell_of[(a, b)] for a, b, _ in tk]]
    p_desk, _ = gpk.predict(m, Z[test])
    p_nc, _ = gpk.predict(m, Z_nc)
    oe, om = epoch_change(Y[test], e_rows, m_rows)
    de, dm = epoch_change(p_desk, e_rows, m_rows)
    ne, nm = epoch_change(p_nc, e_rows, m_rows)
    d = om - oe
    return skill_from_sse(((dm - de - d) ** 2).sum(0), ((nm - ne - d) ** 2).sum(0)), nm - ne


def test_species_driven_by_z_change_scores_positive_and_blind_species_does_not():
    keys, Z, Y = _planted()
    test = keys[:, 0] >= 12                      # whole "blocks" of cells held out
    sk, nc_change = _change_skill(keys, Z, Y, test)
    assert np.allclose(nc_change, 0.0)           # no_change predicts zero change by construction
    assert sk[0] > 0.5
    assert sk[1] < 0.1


def test_no_temporal_signal_in_z_means_no_change_skill():
    # If z does not move over time, desk and no_change are the same predictor: skill exactly 0.
    keys, Z, Y = _planted(temporal=False)
    test = keys[:, 0] >= 12
    sk, _ = _change_skill(keys, Z, Y, test)
    np.testing.assert_allclose(sk, 0.0, atol=1e-9)


# ----------------------------- metrics -----------------------------

def test_skill_undefined_where_baseline_has_nothing_to_predict():
    sk = skill_from_sse(np.array([1.0, 0.0, 4.0]), np.array([4.0, 0.0, 1.0]))
    assert sk[0] == pytest.approx(0.5) and np.isnan(sk[1]) and sk[2] == pytest.approx(-1.0)


def test_pooled_skill_counts_undefined_species_and_brackets_point_estimate():
    rng = np.random.default_rng(0)
    base = rng.uniform(1, 2, size=(40, 30))
    model = base * 0.64                           # skill 0.2 everywhere
    model[:, 0] = base[:, 0] = 0.0                # one species with nothing to predict
    out, sk = pooled_skill(model, base, n_boot=200, seed=1)
    assert out["n_species_undefined"] == 1 and out["n_species_defined"] == 29
    assert out["median"] == pytest.approx(0.2)
    lo, hi = out["median_ci"]
    assert lo - 1e-9 <= 0.2 <= hi + 1e-9
    assert out["share_above_zero"] == 1.0


def test_crps_matches_numerical_integral():
    from scipy.stats import norm
    y, mu, var = 0.7, 0.2, 0.5
    xs = np.linspace(-10, 10, 200001)
    F = norm.cdf(xs, mu, np.sqrt(var))
    num = np.trapezoid((F - (xs >= y)) ** 2, xs)
    assert gaussian_crps(np.array(y), np.array(mu), np.array(var)) == pytest.approx(num, rel=1e-4)


def test_interval_coverage_is_calibrated_for_a_correct_predictive():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200000)
    assert interval_coverage(y, 0.0, 1.0, 0.9).mean() == pytest.approx(0.9, abs=0.005)


def test_block_sums_groups_rows():
    b, s = block_sums(np.array([[1.0], [2.0], [3.0]]), np.array([5, 2, 5]))
    assert b.tolist() == [2, 5] and s[:, 0].tolist() == [2.0, 4.0]


# ----------------------------- splits and references -----------------------------

def test_row_splits_follow_holdout_buffer_and_block_tiling():
    ho = np.zeros((12, 12), bool)
    ho[0:6, 6:12] = True
    bf = np.zeros_like(ho)
    bf[0:6, 5] = True
    keys = np.array([[1, 7, 2000], [1, 5, 2000], [8, 1, 2000], [2, 8, 1970]])
    tr, grp, bid = row_splits(keys, ho, bf, 6)
    assert grp.tolist() == [1, 0, 0, 1]                    # no withheld years: space only
    assert tr.tolist() == [False, False, True, False]      # buffer and held-out excluded
    assert bid[0] == bid[3] and bid[0] != bid[2]


def test_row_splits_withhold_years_from_training_everywhere():
    # The tempho leak: a withheld year must leave TRAINING in every cell, and be scored as
    # time (training cells) or space_time (held-out cells).
    ho = np.zeros((12, 12), bool)
    ho[0:6, 6:12] = True
    keys = np.array([[8, 1, 1970], [8, 1, 2000], [1, 7, 1970], [1, 7, 2000], [8, 1, 1980]])
    tr, grp, _ = row_splits(keys, ho, None, 6, withheld_years=range(1966, 1986),
                            common_years=range(1966, 1976))
    assert tr.tolist() == [False, True, False, False, False]
    assert grp.tolist() == [2, 0, 3, 1, 0]                 # 1980: withheld, outside common


def test_change_sets_pair_the_right_epochs():
    from src.community_encoder.train_DESK.validate_gp_species import change_sets
    tk = np.array([[0, 0, 1970], [0, 0, 2010], [1, 1, 1970], [1, 1, 2010]])
    g = np.array([3, 1, 2, 0])
    sets = change_sets(tk, g, withheld=[1970])
    assert sets["space_time"].tolist() == [0, 1] and sets["time"].tolist() == [2, 3]
    assert "space" in change_sets(tk, np.array([1, 1, 0, 0]), withheld=[])


def test_no_change_z_is_cell_mean_over_modern_epoch_and_skips_nan():
    keys = np.array([[0, 0, 1970], [0, 0, 1990], [1, 1, 1970]])
    ref = modern_reference_keys(keys, modern=(2010, 2011))
    Zr = np.zeros((len(ref), 2))
    for i, (r, c, y) in enumerate(ref):
        Zr[i] = [r + c, y - 2010] if not (r == 1 and y == 2011) else [np.nan, np.nan]
    out = no_change_z(keys, ref, Zr)
    np.testing.assert_allclose(out[0], [0.0, 0.5])
    np.testing.assert_allclose(out[1], out[0])
    np.testing.assert_allclose(out[2], [2.0, 0.0])


# ----------------------------- species layout -----------------------------

def test_layout_keeps_community_order_and_excludes_focal_and_community():
    comm, ev = species_layout(["casfin", "houspa", "allhum"],
                              ["zzzbrd", "houspa", "houfin", "amerob", "casfin"])
    assert comm == ["casfin", "houspa", "allhum"]           # given order, unmatched kept
    assert ev == ["amerob", "zzzbrd"]
    assert "houfin" not in comm + ev


def test_layout_refuses_focal_species_inside_the_community():
    with pytest.raises(ValueError, match="focal"):
        species_layout(["casfin", "houfin"], ["amerob"])


# ----------------------------- end to end, with I/O mocked -----------------------------

def _synthetic_run(tmp_path, monkeypatch, withheld=(), common=(), n_comm=2, species=None):
    """The orchestration in ``run`` -- splits, encode bookkeeping, every predictor, the report
    and the saved tables -- on a synthetic grid where half the species are readouts of z."""
    from src.community_encoder.train_DESK import validate_bbs_routes as vbr
    from src.community_encoder.train_DESK import validate_gp_species as vgs
    from src.community_encoder.train_DESK import esk_kernel

    rng = np.random.default_rng(0)
    H = W = 24
    L = 6
    base = rng.normal(size=(H, W, L))
    drift = rng.normal(size=(H, W, L))

    def z_of(keys):
        k = np.asarray(keys)
        frac = (k[:, 2] - 1966) / (2025 - 1966)
        return (base[k[:, 0], k[:, 1]] + frac[:, None] * drift[k[:, 0], k[:, 1]]).astype("float32")

    years = list(range(1966, 2026, 3))
    keys = np.array([(r, c, y) for r in range(H) for c in range(W) for y in years], "int32")
    Zk = z_of(keys)
    n_ev = 10
    Wsp = rng.normal(size=(L, n_ev))
    lin = Zk @ Wsp
    lin[:, n_ev // 2:] = rng.normal(size=(len(keys), n_ev - n_ev // 2))   # blind species
    X_ev = np.expm1(np.clip(lin + 1.0, 0, None))
    # plus one species seen ONLY in held-out cells: no training detections, must be dropped
    only_ho = np.zeros((len(keys), 1))
    only_ho[(keys[:, 0] < 12) & (keys[:, 1] >= 12), 0] = 2.0
    X_raw = np.hstack([rng.uniform(0, 3, size=(len(keys), n_comm)), X_ev,
                       only_ho]).astype("float32")
    layout = {"community": [f"c{i}" for i in range(n_comm)],
              "evaluation": [f"s{i:02d}" for i in range(n_ev)] + ["heldonly"],
              "n_community": n_comm, "n_evaluation": n_ev + 1, "focal_excluded": "houfin"}

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ho = np.zeros((H, W), bool)
    ho[0:12, 12:24] = True
    ho[12:18, 0:6] = True
    bf = np.zeros_like(ho)
    np.save(run_dir / "holdout_cells.npy", ho)
    np.save(run_dir / "buffer_cells.npy", bf)
    np.savez(run_dir / "desk_meta.npz", latent_dim=L, best_epoch=107)

    monkeypatch.setattr(vgs, "load_all_species", lambda cfg: (X_raw, keys, layout))
    # raw z: the drift lands abruptly; z_ema: the same drift, so the z-driven species follow it
    monkeypatch.setattr(vbr, "desk_z_ema",
                        lambda cfg, k, return_raw=False: (z_of(k), {"ema_half_life": 10.0},
                                                          z_of(k) + 0.05))
    P = rng.normal(size=(n_comm, L))
    monkeypatch.setattr(esk_kernel, "project_points_to_z",
                        lambda X, zd, l: (np.asarray(X, "float64") @ P).astype("float32"))
    # Covariates that know each cell but not its change: the covariate GP can place a species
    # but, like no_change, cannot see z's drift.
    def _cov(cfg, k, both=False):
        c = base[np.asarray(k)[:, 0], np.asarray(k)[:, 1], :3].astype("float32")
        return (c, c + 0.01) if both else c
    monkeypatch.setattr(vgs, "covariates_for_keys", _cov)
    cfg = {"paths": {"desk_output_dir": str(run_dir)},
           "desk": {"z_dir": "unused", "trend": {"block_cells": 6,
                                                 "holdout_years": list(withheld),
                                                 "common_holdout_years": list(common)}}}
    rep = vgs.run(cfg, out_dir=str(tmp_path / "out"), n_boot=50,
                  opts={"cell_km": 27.0, "n_fit": 400, "shape_iters": 15, "k_nn": (16, 4),
                        "k_max": 800, "thin": (0.3,),
                        **({"species": species} if species else {})})
    return rep, n_ev, ho, keys


def test_run_end_to_end_on_a_synthetic_grid(tmp_path, monkeypatch):
    rep, n_ev, ho, keys = _synthetic_run(tmp_path, monkeypatch)
    assert rep["dropped_zero_training_detections"]["species"] == ["heldonly"]
    lv = rep["level"]["space"]
    for name in ("desk_pooled", "spacetime_pooled", "covariate_pooled", "desk_raw_pooled"):
        assert f"{name}_vs_intercept" in lv
    assert "desk_pooled_vs_spacetime_pooled" in lv
    assert set(rep["level"]) == {"space"} and set(rep["change"]) == {"space"}

    assert rep["rows"]["space"] == int(ho[keys[:, 0], keys[:, 1]].sum())
    assert rep["primary"]["set"] == "space"
    assert rep["primary"]["n_species_defined"] == n_ev
    assert 0.3 <= rep["primary"]["share_above_zero"] <= 0.7    # half the species are blind
    assert rep["direction"]["space"]["no_change"]["n_species_scored"] == 0   # never commits
    assert rep["completeness_gaps"] == []
    assert ("esk_oracle_independent" in rep["predictors"]
            or "esk_oracle_independent" in rep["unavailable"])
    v = rep["change"]["space"]["desk_vs_no_change"]
    assert "balanced_median" in v and "regions" in v
    assert set(rep["level_by_window"]["space"]) <= {"early", "mid", "modern"}
    assert rep["decomposition"]["space"]["desk"]["median_cos"] is not None
    import pandas as pd
    tab = pd.read_csv(tmp_path / "out" / "per_species.csv")
    sk = tab["change_skill_space_desk_vs_no_change"].to_numpy()
    assert (sk[: n_ev // 2] > 0.3).all() and (sk[n_ev // 2:] < 0.1).all()
    saved = np.load(tmp_path / "out" / "heldout_predictions.npz", allow_pickle=True)
    n_pred = rep["rows"]["predicted"]
    assert saved["mean_desk"].shape == (n_pred, n_ev)
    for name in ("spacetime", "covariate", "spacetime_k4", "covariate_k4", "desk_raw",
                 "no_change_raw", "covariate_raw"):
        assert saved[f"mean_{name}"].shape == (n_pred, n_ev)
    assert np.isfinite(saved["cov_novelty"]).all() and (saved["dist_to_train_km"] > 0).all()
    # desk beats both baselines on change for the z-driven species: they cannot see the drift
    for b in ("spacetime", "covariate"):
        skb = tab[f"change_skill_space_desk_vs_{b}"].to_numpy()
        assert (skb[: n_ev // 2] > 0).all()
    assert "desk_raw_vs_no_change_raw" in lv and "desk_vs_desk_raw" in lv
    th = np.load(tmp_path / "out" / "thinning.npz")
    assert "level_sse_covariate_raw" not in th.files       # raw covariate arm: main run only
    assert th["level_sse_desk"].shape == (2, n_ev)
    assert (th["n_train_detections"][1] <= th["n_train_detections"][0]).all()


# ----------------------------- stationary baselines -----------------------------

def _st_data(n=150, S=3, seed=0):
    rng = np.random.default_rng(seed)
    F = np.column_stack([rng.uniform(0, 500, n), rng.uniform(0, 500, n),
                         rng.integers(1970, 2020, n)]).astype("float64")
    Y = np.column_stack([np.sin(F[:, 0] / 80) + 0.1 * rng.normal(size=n) for _ in range(S)])
    Y[:, 1] = np.cos(F[:, 1] / 60) + 0.3 * rng.normal(size=n)
    return F, Y


def _profiled_total_nll(kind, F, Y, theta):
    import torch
    K = gpk.KERNELS[kind](torch.as_tensor(F), torch.as_tensor(F),
                          torch.as_tensor(np.asarray(theta, "float64"))).numpy()
    _, _, _, hp = gpk._scales_given_shape(K, Y - Y.mean(0))
    return float(np.nansum(hp["nll"]))


def test_envelope_gradient_matches_finite_differences_of_profiled_nll():
    import torch
    F, Y = _st_data()
    th = np.log([90.0, 15.0])
    Yc = Y - Y.mean(0)
    kern = gpk.KERNELS["spacetime"]
    Ft = torch.as_tensor(F)
    theta = torch.tensor(th, requires_grad=True)
    K = kern(Ft, Ft, theta).detach().numpy()
    Q, lam, U, hp = gpk._scales_given_shape(K, Yc)
    s2, n2 = hp["s2"], hp["n2"]
    D = s2[None, :] * lam[:, None] + n2[None, :]
    A = U / D
    M = Q @ (np.diag((s2[None, :] / D).sum(1)) - (A * s2[None, :]) @ A.T) @ Q.T
    (0.5 * (torch.as_tensor(M) * kern(Ft, Ft, theta)).sum()).backward()
    g = theta.grad.numpy()
    h = 1e-4
    fd = [(_profiled_total_nll("spacetime", F, Y, th + h * e)
           - _profiled_total_nll("spacetime", F, Y, th - h * e)) / (2 * h) for e in np.eye(2)]
    np.testing.assert_allclose(g, fd, rtol=2e-3, atol=2e-3)


def test_fit_shared_shape_lowers_the_objective():
    F, Y = _st_data()
    fit = gpk.fit_shared_shape("spacetime", F, Y, np.log([500.0, 50.0]), n_iter=40,
                               verbose=False)
    assert fit["trace"][-1] < fit["trace"][0]


def test_local_prediction_with_full_support_equals_exact_gp():
    import torch
    F, Y = _st_data(n=120)
    rng = np.random.default_rng(5)
    Ft = np.column_stack([rng.uniform(0, 500, 9), rng.uniform(0, 500, 9),
                          rng.integers(1970, 2020, 9)]).astype("float64")
    shape = {"kind": "spacetime", "theta": np.log([80.0, 20.0]), "ybar": Y.mean(0),
             "s2": np.array([0.8, 1.1, 0.5]), "n2": np.array([0.05, 0.2, 0.1])}
    mean, var = gpk.predict_local(shape, F, Y, Ft, np.zeros(9, int), k=200, k_max=10_000)
    kern = gpk.KERNELS["spacetime"]
    th = torch.as_tensor(shape["theta"])
    K = kern(torch.as_tensor(F), torch.as_tensor(F), th).numpy()
    Ks = kern(torch.as_tensor(Ft), torch.as_tensor(F), th).numpy()
    for s in range(3):
        C = shape["s2"][s] * K + shape["n2"][s] * np.eye(len(F))
        Ci = np.linalg.inv(C)
        m = Y[:, s].mean() + shape["s2"][s] * Ks @ Ci @ (Y[:, s] - Y[:, s].mean())
        v = (shape["s2"][s] - shape["s2"][s] ** 2 * np.einsum("ij,jk,ik->i", Ks, Ci, Ks)
             + shape["n2"][s])
        np.testing.assert_allclose(mean[:, s], m, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(var[:, s], v, rtol=1e-7, atol=1e-7)


def test_knn_union_caps_to_the_nearest_rows():
    tr = np.arange(100, dtype="float64")[:, None]
    idx = gpk.knn_union(tr, np.array([[10.0], [11.0]]), k=5, k_max=3, device="cpu")
    assert sorted(idx.tolist()) == [10, 11, 12] or sorted(idx.tolist()) == [9, 10, 11]


def test_spacetime_baseline_wins_level_on_a_purely_spatial_species():
    # A species that is a smooth spatial field, blind to z: the spacetime GP must beat desk.
    rng = np.random.default_rng(2)
    n = 1500
    xy = rng.uniform(0, 1000, size=(n, 2))
    yr = rng.integers(1970, 2020, n).astype(float)
    F = np.column_stack([xy, yr])
    y = np.sin(xy[:, 0] / 120) * np.cos(xy[:, 1] / 150) + 0.1 * rng.normal(size=n)
    Z = rng.normal(size=(n, 6))
    test = xy[:, 0] > 850
    Y = y[:, None]
    shape = gpk.fit_shared_shape("spacetime", F[~test][:600], Y[~test][:600],
                                 np.log([200.0, 30.0]), n_iter=30, verbose=False)
    m_st, _ = gpk.predict_local(shape, F[~test], Y[~test], F[test],
                                (xy[test, 1] // 200).astype(int), k=32, k_max=1500)
    m_d, _ = gpk.predict(gpk.fit(Z[~test], Y[~test]), Z[test])
    e_st = np.mean((m_st[:, 0] - y[test]) ** 2)
    e_d = np.mean((m_d[:, 0] - y[test]) ** 2)
    assert e_st < 0.5 * e_d


# ----------------------------- helpers -----------------------------

def test_rows_from_stack_gathers_and_marks_missing():
    from src.community_encoder.train_DESK.validate_gp_species import rows_from_stack
    stack = np.arange(2 * 2 * 1, dtype="float32").reshape(2, 2, 1)     # (T=2, cells=2, C=1)
    out = rows_from_stack(stack, [2000, 2001], np.array([[0, 0], [3, 4]]),
                          np.array([[3, 4, 2001], [0, 0, 2000], [9, 9, 2000], [0, 0, 1999]]))
    assert out[0, 0] == 3.0 and out[1, 0] == 0.0
    assert np.isnan(out[2, 0]) and np.isnan(out[3, 0])


def test_cooccurrence_similarity_finds_the_tracking_community_member():
    from src.community_encoder.train_DESK.validate_gp_species import cooccurrence_similarity
    rng = np.random.default_rng(0)
    comm = rng.normal(size=(500, 4))
    ev = np.column_stack([comm[:, 2] + 0.01 * rng.normal(size=500), rng.normal(size=500),
                          np.zeros(500)])
    mx, top = cooccurrence_similarity(ev, comm, top=2)
    assert mx[0] > 0.99 and abs(mx[1]) < 0.2 and np.isnan(mx[2]) and np.isnan(top[2])


def test_distance_and_novelty():
    from src.community_encoder.train_DESK.validate_gp_species import (
        distance_to_training_km, mahalanobis_novelty)
    d = distance_to_training_km(np.array([[0, 3], [5, 5]]), np.array([[0, 0], [5, 6]]), 27.0)
    np.testing.assert_allclose(d, [81.0, 27.0])
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(5000, 3)) * [1.0, 10.0, 1.0]
    nov = mahalanobis_novelty(np.array([[0, 0, 0], [3, 0, 0], [0, 30, 0]]), Xtr)
    assert nov[0] < 0.1 and nov[1] == pytest.approx(3, rel=0.05) and nov[2] == pytest.approx(
        3, rel=0.05)


def test_thin_rows_is_a_subset_of_the_right_size():
    from src.community_encoder.train_DESK.validate_gp_species import thin_rows
    rows = np.arange(10, 1010)
    t = thin_rows(rows, 0.1, np.random.default_rng(0))
    assert len(t) == 100 and set(t) <= set(rows)
    assert len(thin_rows(rows, 1.0, np.random.default_rng(0))) == 1000


def test_comparisons_cover_nulls_and_desk_against_every_baseline():
    from src.community_encoder.train_DESK.validate_gp_species import comparisons
    pairs = comparisons(["desk", "no_change", "intercept", "spacetime", "covariate"])
    for p in [("desk", "no_change"), ("spacetime", "no_change"), ("desk", "intercept"),
              ("desk", "spacetime"), ("desk", "covariate")]:
        assert p in pairs
    assert ("no_change", "no_change") not in pairs and ("intercept", "no_change") not in pairs
    pairs = comparisons(["desk", "desk_raw", "no_change", "no_change_raw", "intercept"])
    assert ("desk_raw", "no_change_raw") in pairs and ("desk", "desk_raw") in pairs
    assert not any(a.startswith("no_change") for a, _ in pairs)


# ----------------------------- analysis -----------------------------

def test_twoway_cluster_recovers_beta_and_widens_se_under_shared_shocks():
    from src.community_encoder.train_DESK.gp_species_analysis import wls_twoway_cluster
    rng = np.random.default_rng(0)
    n_sp, n_bl, per = 60, 20, 10
    sp = np.repeat(np.arange(n_sp), n_bl * per)
    bl = np.tile(np.repeat(np.arange(n_bl), per), n_sp)
    xb = rng.normal(size=n_bl)[bl]                         # regressor constant within block
    y = 0.5 * xb + rng.normal(size=n_bl)[bl] + 0.3 * rng.normal(size=len(sp))
    X = np.column_stack([np.ones(len(y)), xb])
    res = wls_twoway_cluster(X, y, np.ones(len(y)), sp, bl)
    assert abs(res["beta"][1] - 0.5) < 3 * res["se"][1]
    iid = np.sqrt(np.linalg.inv(X.T @ X)[1, 1] * np.var(y - X @ res["beta"]))
    assert res["se"][1] > 3 * iid                          # block shocks must widen the SE
    assert res["n_clusters"] == [n_sp, n_bl]


def test_trait_and_patristic_distance_to_a_set():
    import dendropy
    from src.community_encoder.train_DESK.gp_species_analysis import (
        patristic_to_set, trait_distance_to_set)
    mn, me = trait_distance_to_set(np.array([[0.0, 0.0], [np.nan, 1.0]]),
                                   np.array([[3.0, 4.0], [0.0, 1.0]]))
    assert mn[0] == pytest.approx(1.0) and me[0] == pytest.approx(3.0) and np.isnan(mn[1])
    tree = dendropy.Tree.get(data="((A:1,B:1):1,(C:2,D:2):1);", schema="newick",
                             preserve_underscores=True)
    pmin, pmean = patristic_to_set(tree, ["A", "A", "Zz"], ["B", "C"])
    np.testing.assert_allclose(pmin[:2], [2.0, 2.0])         # duplicates stay positional
    np.testing.assert_allclose(pmean[:2], [(2.0 + 5.0) / 2] * 2)   # A-C = 1+1+1+2
    assert np.isnan(pmin[2])


def test_analysis_runs_on_the_synthetic_outputs(tmp_path, monkeypatch):
    import pandas as pd
    from src.community_encoder.train_DESK import gp_species_analysis as ga
    rep, n_ev, _, _ = _synthetic_run(tmp_path, monkeypatch)
    out = tmp_path / "out"
    rng = np.random.default_rng(1)
    sim = pd.DataFrame({"species_code": [f"s{i:02d}" for i in range(n_ev)],
                        "phylo_min": rng.uniform(5, 50, n_ev),
                        "phylo_mean": rng.uniform(50, 90, n_ev),
                        "trait_min": rng.uniform(0, 3, n_ev),
                        "trait_mean": rng.uniform(3, 6, n_ev),
                        "migration": rng.integers(1, 4, n_ev).astype(float)})
    monkeypatch.setattr(ga, "species_similarity", lambda e, c, d: sim)
    res = ga.run(str(out), datasets_root="unused")
    for base in ("no_change", "spacetime", "covariate"):
        m = res["level"][f"space: desk_vs_{base}"]["main"]
        assert "coefficients" in m or "note" in m
    m = res["level"]["space: desk_vs_spacetime"]["main"]
    assert "dist_to_train_km x cooc_max" in m["coefficients"]
    tc = res["thinning_curves"]
    assert set(tc["change"]) >= {"desk", "spacetime", "covariate"}
    assert (out / "per_species_with_similarity.csv").exists()


# ----------------------------- observation noise -----------------------------

def _noise_fixture(n_cells=400, S=3, years=8, sig=(1.0, 0.3, 0.0), noise=0.5, seed=0):
    """Cells with a true per-species log-abundance change plus per-year lognormal noise (mean-one,
    so the epoch-mean count is unbiased). Species 0 has large real change, 1 small, 2 none. A
    predictor that knows the true expected counts is stored as mean_perfect (var 0)."""
    rng = np.random.default_rng(seed)
    base = np.log(rng.uniform(2, 20, size=(n_cells, S)))
    change = rng.normal(size=(n_cells, S)) * np.asarray(sig)
    y, e_rows, m_rows, perfect = [], [], [], []
    r = 0
    for c in range(n_cells):
        e = list(range(r, r + years))
        r += years
        m = list(range(r, r + years))
        r += years
        e_rows.append(np.array(e))
        m_rows.append(np.array(m))
        for lev in (base[c], base[c] + change[c]):
            eps = rng.normal(size=(years, S)) * noise
            y.append(np.log1p(np.exp(lev) * np.exp(eps - noise ** 2 / 2)))
            perfect.append(np.tile(np.log1p(np.exp(lev)), (years, 1)))
    yy = np.concatenate(y)
    H = {"y": yy, "species": np.array(["a", "b", "c"]),
         "change_early_rows_s": np.array(e_rows, dtype=object),
         "change_modern_rows_s": np.array(m_rows, dtype=object),
         "mean_perfect": np.concatenate(perfect), "var_perfect": np.zeros_like(yy),
         "mean_no_change": np.zeros_like(yy), "var_no_change": np.zeros_like(yy)}

    class _NpzLike(dict):
        files = property(lambda self: list(self.keys()))
    return _NpzLike(H), change, noise, years


def test_noise_ceiling_recovers_the_planted_noise_and_signal():
    from src.community_encoder.train_DESK.gp_species_analysis import change_noise_ceiling
    H, change, noise, years = _noise_fixture()
    summ, tab = change_noise_ceiling(H, "s", n_boot=200)
    # The split-half noise must match the ACTUAL error of the observed change against the truth,
    # measured directly here on the abundance estimand.
    from src.community_encoder.train_DESK.validation_core import epoch_values
    er, mr = list(H["change_early_rows_s"]), list(H["change_modern_rows_s"])
    oe, om = epoch_values(np.expm1(H["y"]), er, mr)
    te, tm = epoch_values(np.expm1(H["mean_perfect"]), er, mr)
    actual = (((om - oe) - (tm - te)) ** 2).mean(0)
    np.testing.assert_allclose(tab["noise_change"], actual, rtol=0.3)
    # species 0: large real change, resolvable; species 2: none, not resolvable
    assert tab["resolvable"].tolist()[0] and not tab["resolvable"].tolist()[2]
    # A predictor that knows the true change captures ~all of the available signal ...
    assert tab["captured_perfect"].iloc[0] == pytest.approx(1.0, abs=0.1)
    # ... and no_change captures none of it, by construction.
    assert tab["captured_no_change"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(tab["captured_perfect"].iloc[2])       # unresolvable: not graded


def test_noise_ceiling_on_pure_noise_resolves_nothing():
    from src.community_encoder.train_DESK.gp_species_analysis import change_noise_ceiling
    H, _, _, _ = _noise_fixture(sig=(0.0, 0.0, 0.0), seed=3)
    summ, tab = change_noise_ceiling(H, "s", n_boot=200)
    assert summ["n_resolvable"] == 0
    assert abs(summ["pooled_signal_share"]) < 0.1


def test_abba_halves_cancel_a_linear_within_epoch_trend():
    from src.community_encoder.train_DESK.validation_core import abba_halves, split_half_change
    yrs = np.arange(1990, 1998)                       # 8 years: two full ABBA blocks
    a, b = abba_halves(np.arange(8), yrs)
    assert len(a) == len(b) == 4 and set(a).isdisjoint(b)
    assert yrs[a].mean() == yrs[b].mean()
    # A pure within-epoch trend, no noise, no between-epoch change: halves must agree exactly,
    # i.e. zero estimated noise. A random split would report the trend as noise.
    raw = np.concatenate([np.arange(8.0), np.arange(8.0)])[:, None] + 1.0  # counts, linear trend
    full, da, db = split_half_change(raw, [np.arange(8)], [np.arange(8, 16)],
                                     years=np.concatenate([yrs, yrs + 30]))
    assert da[0, 0] == pytest.approx(db[0, 0])
    # rows given out of year order are sorted before splitting
    a2, b2 = abba_halves(np.array([3, 0, 2, 1]), np.array([1993, 1990, 1992, 1991]))
    assert sorted(a2.tolist()) == [0, 3] and sorted(b2.tolist()) == [1, 2]


# ----------------------------- fixes after the first run -----------------------------

def test_direction_treats_float_rounding_as_abstention():
    from src.community_encoder.train_DESK.validate_gp_species import direction_by_species
    rng = np.random.default_rng(0)
    obs_e, obs_m = rng.uniform(0, 2, (50, 1)), rng.uniform(0, 2, (50, 1))
    pe = np.full((50, 1), 0.3)
    pm = pe + rng.choice([-1, 1], (50, 1)) * 1e-16      # what a constant predictor yields
    r = direction_by_species(obs_e, obs_m, pe, pm)[0]
    assert "direction_skill" not in r                     # abstained, not scored


def test_shape_fit_converges_and_says_so():
    # Temporal structure in the data, so the time lengthscale has a finite optimum.
    F, Y = _st_data()
    Y = Y + np.sin(F[:, 2] / 4.0)[:, None]
    fit = gpk.fit_shared_shape("spacetime", F, Y, np.log([90.0, 15.0]), n_iter=400,
                               verbose=False)
    assert fit["converged"] and fit["n_iterations"] < 400
    # at the optimum the profiled objective is no better at nearby lengthscales
    best = _profiled_total_nll("spacetime", F, Y, fit["theta"])
    for e in np.eye(2):
        for h in (-0.05, 0.05):
            assert _profiled_total_nll("spacetime", F, Y, fit["theta"] + h * e) >= best - 1e-3


def test_regression_scale_is_data_side_so_a_perfect_baseline_cannot_blow_it_up():
    from src.community_encoder.train_DESK.gp_species_analysis import level_frame
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 200
    y = np.zeros((n, 2))
    y[:5, 0] = 1.0                                        # rare species: five detections
    y[:, 1] = rng.normal(size=n)
    class _N(dict):
        files = property(lambda self: list(self.keys()))
    H = _N(y=y, mean_desk=y + 0.1, mean_base=y.copy(),          # baseline is PERFECT
           keys=np.column_stack([np.arange(n) % 20, np.arange(n) // 20, np.full(n, 2000)]),
           block_id=np.arange(n) % 4, dist_to_train_km=np.ones(n), row_group=np.ones(n, int),
           cov_novelty=np.ones(n), years_before_modern=np.ones(n))
    f = level_frame(H, "base", pd.DataFrame({"species_code": ["a", "b"]}))
    assert np.isfinite(f["response"]).all() and f["response"].abs().max() < 1.0


def test_thinning_captured_share_on_resolvable_species():
    import pandas as pd
    from src.community_encoder.train_DESK.gp_species_analysis import thinning_curves

    class _T(dict):
        files = property(lambda self: list(self.keys()))
    n_cells = 10
    z = np.array([10.0, 10.0, 0.0])                       # sum of squared observed change
    T = _T(n_train_detections=np.array([[50, 5, 0], [20, 2, 0]]),
           fractions=np.array([1.0, 0.3]),
           change_sse_zero=z,
           change_sse_desk=np.array([[6.0, 10.0, 0.0], [8.0, 10.0, 0.0]]),
           change_sse_no_change=np.tile(z, (2, 1)))
    noise = pd.DataFrame({"ms_obs_change": z / n_cells, "noise_change": [0.2, 0.2, 0.0],
                          "resolvable": [True, True, False]})
    out = thinning_curves(T, noise=noise)
    # available = 10 - 0.2*10 = 8 per resolvable species; desk gains 4 and 0 at f=1
    assert out["captured"]["desk"]["pooled"][0] == pytest.approx(4.0 / 16.0)
    assert out["captured"]["desk"]["pooled"][1] == pytest.approx(2.0 / 16.0)
    assert out["captured"]["no_change"]["pooled"] == [0.0, 0.0]


# ----------------------------- pooled scales -----------------------------

def _pool_data(seed=0, n=3000, r=6, S=60, share=0.5):
    """Species with the SAME true kernel share; half are rare (a handful of nonzero rows)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, r))
    Y = np.zeros((n, S))
    det = np.zeros(S)
    for s in range(S):
        w = rng.normal(size=r)
        f = Z @ w
        f = f / f.std() * np.sqrt(share)
        y = f + rng.normal(size=n) * np.sqrt(1 - share)
        if s % 2:                                    # rare: observe only ~8 rows, rest zero
            keep = rng.choice(n, 8, replace=False)
            m = np.zeros(n)
            m[keep] = y[keep]
            y = m
        Y[:, s] = y
        det[s] = (y != 0).sum()
    return Z, Y, det


def test_pooled_objective_gradient_matches_finite_differences():
    Z, Y, _ = _pool_data(S=4)
    b = gpk.feature_svd(Z)
    st = gpk.sufficient_stats(b, Z, Y)
    kap = gpk._kappa(b)
    lv, eta = np.log(Y.var(0)), np.array([0.3, -1.0, 2.0, 0.0])
    mu, tau2 = np.zeros(4), 1.5
    _, gl, ge = gpk._pooled_objective(lv, eta, b, st, kap, mu, tau2)
    h = 1e-6
    f = lambda a, e: gpk._pooled_objective(a, e, b, st, kap, mu, tau2)[0]
    np.testing.assert_allclose(gl, (f(lv + h, eta) - f(lv - h, eta)) / (2 * h), rtol=1e-4,
                               atol=1e-4)
    np.testing.assert_allclose(ge, (f(lv, eta + h) - f(lv, eta - h)) / (2 * h), rtol=1e-4,
                               atol=1e-4)


def test_pooling_shrinks_noisy_amplitude_estimates_toward_the_shared_truth():
    # Every species has the SAME true share; with few rows the per-species estimates scatter, and
    # the pooled ones must sit closer to the truth.
    rng = np.random.default_rng(0)
    n, r, S, share = 120, 6, 80, 0.3
    Z = rng.normal(size=(n, r))
    Y = np.empty((n, S))
    for s_ in range(S):
        f = Z @ rng.normal(size=r)
        f = f / f.std() * np.sqrt(share)
        Y[:, s_] = f + rng.normal(size=n) * np.sqrt(1 - share)
    b = gpk.feature_svd(Z)
    kap = gpk._kappa(b)
    sh = lambda m: m["s2"] * kap / (m["s2"] * kap + m["n2"])
    m0, mp = gpk.fit(Z, Y), gpk.fit(Z, Y, pool=True)
    logit = lambda p: np.log(p / (1 - p))
    err0 = np.mean((logit(np.clip(sh(m0), 1e-6, 1 - 1e-6)) - logit(share)) ** 2)
    errp = np.mean((logit(np.clip(sh(mp), 1e-6, 1 - 1e-6)) - logit(share)) ** 2)
    assert errp < err0
    assert mp["prior"]["converged"] and mp["prior"]["tau"] >= gpk.TAU_FLOOR


def test_pooling_does_not_manufacture_signal_for_zero_dominated_species():
    # Rarity as mostly zeros: the kernel's share of the OBSERVED variance is genuinely small, and
    # the pooled fit must not pretend otherwise.
    Z, Y, det = _pool_data()
    b = gpk.feature_svd(Z)
    kap = gpk._kappa(b)
    mp = gpk.fit(Z, Y, pool=True)
    sh = mp["s2"] * kap / (mp["s2"] * kap + mp["n2"])
    assert np.median(sh[det < 20]) < 0.05 and np.median(sh[det >= 20]) > 0.4


def test_fit_is_scale_equivariant_down_to_rare_species_units():
    # Rescaling a species' abundance must rescale both variances by c^2 -- including at the tiny
    # scale of a rare species, which an absolute floor broke.
    Z, Y = _toy(n=400, k=2)
    for c in (1.0, 1e-3):
        m = gpk.fit(Z, Y * c)
        if c == 1.0:
            ref = m
        else:
            np.testing.assert_allclose(m["s2"], ref["s2"] * c ** 2, rtol=1e-3)
            np.testing.assert_allclose(m["n2"], ref["n2"] * c ** 2, rtol=1e-3)


def test_run_end_to_end_with_withheld_early_decades(tmp_path, monkeypatch):
    """The tempho design: early years withheld from EVERY cell. All three groups must be scored,
    and no withheld year may reach any fit."""
    import json
    from src.community_encoder.train_DESK import gp_kernels as gk
    seen = []
    real_fit = gk.fit

    def spy_fit(Z, Y, pool=False):
        seen.append(len(Z))
        return real_fit(Z, Y, pool)
    monkeypatch.setattr(gk, "fit", spy_fit)
    rep, n_ev, ho, keys = _synthetic_run(tmp_path, monkeypatch,
                                         withheld=range(1966, 1986), common=range(1966, 1976))
    assert set(rep["level"]) == {"space", "time", "space_time"}
    assert set(rep["change"]) == {"space_time", "time"}
    assert rep["primary"]["set"] == "space_time"
    # every fit saw only training rows: held-out cells, buffer, and withheld years excluded
    trainable = (~ho[keys[:, 0], keys[:, 1]]) & ~np.isin(keys[:, 2], range(1966, 1986))
    assert max(seen) <= int(trainable.sum())
    saved = np.load(tmp_path / "out" / "heldout_predictions.npz", allow_pickle=True)
    g = saved["row_group"]
    yrs = saved["keys"][:, 2]
    assert (yrs[g == 2] < 1976).all() and (yrs[g == 3] < 1976).all()
    assert (saved["years_beyond_training_edge"][g == 2] >= 11).all()   # 1975 is 11 yr past 1986


def test_summary_and_noise_run_on_tempho_outputs(tmp_path, monkeypatch, capsys):
    from src.community_encoder.train_DESK import gp_species_analysis as ga
    _synthetic_run(tmp_path, monkeypatch, withheld=range(1966, 1986), common=range(1966, 1976))
    out = str(tmp_path / "out")
    noise = ga.run_noise(out)
    assert set(noise) == {"space_time", "time"}
    ga.summarize(out)
    printed = capsys.readouterr().out
    assert "PRIMARY" in printed and "CHANGE -- time" in printed and "NOISE -- space_time" in printed


# ----------------------------- audit fixes -----------------------------

def test_predicted_raw_is_the_lognormal_mean():
    from src.community_encoder.train_DESK.validate_gp_species import predicted_raw
    rng = np.random.default_rng(0)
    y = rng.normal(0.7, 0.5, 400000)
    assert predicted_raw(0.7, 0.25) == pytest.approx(np.expm1(y).mean(), rel=5e-3)


def test_independent_oracle_never_uses_the_rows_own_survey(monkeypatch):
    from src.community_encoder.train_DESK import esk_kernel
    from src.community_encoder.train_DESK.validate_gp_species import independent_oracle_z
    # projection = identity on a 1-species community, so z IS the community value it was given
    monkeypatch.setattr(esk_kernel, "project_points_to_z",
                        lambda X, zd, l: np.asarray(X, "float32"))
    years = [1970, 1971, 1972, 1973]
    keys = np.array([[0, 0, y] for y in years] + [[0, 0, 1990]])
    X = np.array([[1.0], [10.0], [100.0], [1000.0], [5.0]])
    Z, info = independent_oracle_z(keys, X, "zd", 1, norm_tol=0.0)
    # ABBA over 1970..73: A = {1970, 1973}, B = {1971, 1972}
    a_val = np.log1p((1.0 + 1000.0) / 2)
    b_val = np.log1p((10.0 + 100.0) / 2)
    np.testing.assert_allclose(Z[[0, 3], 0], b_val, rtol=1e-6)   # A rows see B's community
    np.testing.assert_allclose(Z[[1, 2], 0], a_val, rtol=1e-6)
    assert np.isnan(Z[4]).all()                                  # 1990: outside both epochs


def test_independent_oracle_refuses_off_span_projections(monkeypatch):
    from src.community_encoder.train_DESK import esk_kernel
    from src.community_encoder.train_DESK.validate_gp_species import independent_oracle_z
    # a projection whose norm collapses for averaged input: refused by the gate
    monkeypatch.setattr(esk_kernel, "project_points_to_z",
                        lambda X, zd, l: (np.asarray(X, "float32") - 2.0))
    keys = np.array([[0, 0, y] for y in (1970, 1971, 1972, 1973)])
    X = np.array([[0.0], [100.0], [0.0], [100.0]])
    Z, info = independent_oracle_z(keys, X, "zd", 1, norm_tol=0.99)
    assert Z is None and "representability gate" in info["reason"]


def test_decomposition_separates_overmoving_from_wrong_direction():
    from src.community_encoder.train_DESK.validate_gp_species import decomposition
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(200, 2))
    over = np.column_stack([2.0 * obs[:, 0], -obs[:, 1]])     # sp0: right way, 2x; sp1: reversed
    summ, per = decomposition(over, obs, np.array([True, True]))
    assert per["cos"][0] == pytest.approx(1.0) and per["overmove"][0] == pytest.approx(2.0)
    assert per["cos"][1] == pytest.approx(-1.0) and np.isnan(per["overmove"][1])


def test_expected_predictors_cover_every_configured_arm():
    from src.community_encoder.train_DESK.validate_gp_species import (
        GP_PREDICTOR_ROLES, expected_predictors, role_of)
    names = expected_predictors({"baselines": True, "pooled": True, "k_nn": (32, 8)})
    assert {"spacetime_k8", "covariate_raw_k8", "covariate_raw_pooled",
            "esk_oracle_independent"} <= set(names)
    assert all(not role_of(n).startswith("UNREGISTERED") for n in names)
    assert "esk_oracle" not in GP_PREDICTOR_ROLES        # the same-rows oracle is gone


def test_predictions_are_decompressed_once(tmp_path):
    from src.community_encoder.train_DESK.gp_species_analysis import Arrays
    np.savez_compressed(tmp_path / "heldout_predictions.npz", y=np.arange(5.0), z=np.ones(3))
    H = Arrays(str(tmp_path / "heldout_predictions.npz"))
    assert H["y"] is H["y"] and "z" in H and "q" not in H and set(H.files) == {"y", "z"}


def test_regional_skill_is_undefined_where_a_species_is_absent():
    # Species 1 is absent in region 1. A baseline predicting ~0 there has a tiny but nonzero
    # error, and a skill against it is noise of any size: it must be undefined, not -1858.
    from src.community_encoder.train_DESK.validate_gp_species import _pair_table
    rng = np.random.default_rng(0)
    n = 400
    region = np.repeat([0, 1], n // 2)
    # 7 species, 4 of them absent in region 1: without the rule the region's MEDIAN is one of
    # the exploding skills; with it, 3 defined species remain (the minimum for a median).
    S = 7
    truth = np.abs(rng.normal(size=(n, S))) + 0.5
    truth[region == 1, :4] = 0.0
    err = {"desk": (rng.normal(size=(n, S)) * 0.3) ** 2,
           "base": np.where(truth > 0, (rng.normal(size=(n, S)) * 0.5) ** 2, 1e-12)}
    avail = {k: np.ones(n, bool) for k in err}
    cells = np.column_stack([np.arange(n), np.zeros(n, int)])
    out, _ = _pair_table(err, avail, np.arange(n) % 8, region, cells, [("desk", "base")], 50, 0,
                         {0: (True, ""), 1: (True, "")}, {}, truth=truth)
    # Without the rule, species 1 in region 1 scores 1 - sqrt(SSE/1e-12-ish): about -1e5.
    for rg in ("0", "1"):
        assert abs(out["desk_vs_base"]["regions"][rg]["median"]) < 5
    assert np.isfinite(out["desk_vs_base"]["balanced_median"])


def test_community_mode_grades_the_community_and_never_pairs_a_species_with_itself(
        tmp_path, monkeypatch):
    rep, n_ev, ho, keys = _synthetic_run(tmp_path, monkeypatch, n_comm=6, species="community")
    assert rep["species_mode"] == "community"
    import json
    lay = json.load(open(tmp_path / "out" / "layout.json"))
    assert lay["evaluation"] == lay["community"][: len(lay["evaluation"])] or \
        set(lay["evaluation"]) <= set(lay["community"])
    import pandas as pd
    tab = pd.read_csv(tmp_path / "out" / "per_species.csv")
    assert (tab["cooc_max"].dropna() < 0.999).all()        # never its own column


def test_trait_distance_excludes_the_species_itself():
    from src.community_encoder.train_DESK.gp_species_analysis import trait_distance_to_set
    T = np.array([[0.0, 0.0], [3.0, 4.0]])
    mn, _ = trait_distance_to_set(T, T, self_index=[0, 1])
    np.testing.assert_allclose(mn, [5.0, 5.0])
