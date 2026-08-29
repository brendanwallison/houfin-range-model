"""Tests for the component-predictability diagnostic: it must SEPARATE the two readings.

The script exists to decide between "components 9-32 are not environmentally determined" and
"they are, and DESK is not fitting them". Those imply opposite programmes -- stop proposing
covariates versus stop acquiring them and fix the encoder -- so a test that only checks the
script runs would miss the entire point. The load-bearing assertions here are the pairs: a
predictable component and an unpredictable one in the SAME fit must come back separated, and a
component that is nonlinearly predictable must be missed by the linear rung and caught by the
capacity ladder. If either pair collapses, the diagnostic's verdict is unfounded.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _mod():
    from scripts.diagnostics import component_predictability as M
    return M


def _split(n, frac=0.3, seed=0):
    p = np.random.default_rng(seed).permutation(n)
    cut = int(n * (1 - frac))
    return p[:cut], p[cut:]


def test_r2_separates_a_predictable_component_from_an_unpredictable_one():
    """One fit, three targets: linear signal, pure noise, and signal+noise.

    This is the claim every branch of the verdict rests on. The noise column must NOT come back
    predictable -- with 60 features and an alpha grid there is ample room to fit it, and if the
    inner selection leaked into the held-out rows it would.
    """
    M = _mod()
    rng = np.random.default_rng(0)
    n, d = 4000, 60
    X = rng.normal(size=(n, d)).astype("float32")
    w = rng.normal(size=d)
    sig = (X @ w) / (X @ w).std()                    # unit variance, so the mix below is the R^2
    Y = np.stack([sig, rng.normal(size=n), 0.3 * sig + 0.95 * rng.normal(size=n)], axis=1)
    tr, va = _split(n)
    res, _ = M.fit_and_score(X, Y, tr, va, rng=np.random.default_rng(1))
    r2 = res["r2"]
    assert r2[0] > 0.95, f"a clean linear component must be recovered; got {r2[0]:.3f}"
    assert r2[1] < 0.05, f"pure noise must not come back predictable; got {r2[1]:.3f}"
    assert 0.02 < r2[2] < 0.30, f"a 0.3/0.95 signal-to-noise mix landed at {r2[2]:.3f}"
    assert r2[0] - r2[1] > 0.9, "the diagnostic cannot separate the two readings"


def test_a_component_that_beats_nothing_is_flagged_not_scored():
    """``beats_const`` must fire when the model loses to the training mean.

    R^2 is taken against the HELD-OUT variance, so a val-mean shift can leave it positive for a
    model that is worse than a constant. Every band mean in the script zeroes such a component,
    which only works if the flag is right.
    """
    M = _mod()
    y = np.array([[10.0], [11.0], [12.0], [13.0]])
    pred = np.array([[11.4], [11.5], [11.6], [11.7]])       # tracks y weakly, centred correctly
    _r2, beats = M.held_out_r2(y, pred, train_mean=np.array([11.5]))
    assert beats[0], "a model beating the constant must not be flagged"
    bad = np.full_like(y, 30.0)                              # far from both means
    r2b, beatsb = M.held_out_r2(y, bad, train_mean=np.array([11.5]))
    assert not beatsb[0] and r2b[0] < 0, "a model worse than the training mean must be flagged"


def test_the_capacity_ladder_catches_what_the_linear_rung_misses():
    """A saturating threshold plus a two-way interaction: linear misses it, the ladder gets it.

    Without this, reading (a) -- "the information is not in the environment" -- could be produced
    by nothing more than a linear model on a nonlinear relationship, which is a statement about the
    regressor and not about the covariates. The target is the two shapes species-environment
    responses actually take (a switch-on past a threshold, and one covariate's effect depending on
    another), embedded in uninformative dimensions the way a real channel is.
    """
    M = _mod()
    rng = np.random.default_rng(2)
    n, d = 4000, 8
    X = rng.normal(size=(n, d)).astype("float32")
    y = 2.0 * np.maximum(X[:, 0] - 0.5, 0.0) + np.tanh(2.0 * X[:, 1]) + 1.5 * X[:, 2] * X[:, 3]
    Y = (y[:, None] + 0.1 * rng.normal(size=(n, 1))).astype("float64")
    tr, va = _split(n, seed=3)
    rungs = M.capacity_ladder(X, tr, n_pairs=400, pca_dim=8, rff_width=1024,
                              rng=np.random.default_rng(9))
    got = {}
    for name, fmaps, _w in rungs:
        res, _ = M.fit_and_score(X, Y, tr, va, fmaps=fmaps, rng=np.random.default_rng(4))
        got[name] = float(res["r2"][0])
    lin = got["linear"]
    add = next(v for k, v in got.items() if k.startswith("additive("))
    inter = next(v for k, v in got.items() if "pairs" in k)
    assert lin < 0.65, f"this target is not linear; linear rung got {lin:.3f} ({got})"
    assert add > lin + 0.05, f"hinges must beat linear on a threshold response ({got})"
    assert inter > 0.9, f"the interaction rung must recover it; got {inter:.3f} ({got})"


def test_an_isotropic_rbf_alone_would_have_licensed_the_wrong_conclusion():
    """The reason the ladder is additive-and-interaction rather than an RBF width sweep.

    An isotropic RFF over the FULL input space -- the obvious "just add capacity" choice, and what
    this script did first -- is defeated by the curse of dimensionality: it must resolve a short
    length scale in the informative directions while staying flat in the rest. On the same target
    the additive+interaction rung recovers above 0.9, so an RBF-only ladder reading near-zero would
    have been reported as "the information is not in the environment". That is the script's most
    expensive conclusion, and it would have been an artifact of the regressor.
    """
    M = _mod()
    rng = np.random.default_rng(2)
    n, d = 4000, 8
    X = rng.normal(size=(n, d)).astype("float32")
    y = np.sin(2.0 * X[:, 0]) * np.cos(2.0 * X[:, 1])
    Y = (y[:, None] + 0.05 * rng.normal(size=(n, 1))).astype("float64")
    tr, va = _split(n, seed=3)
    g = M._median_bandwidth(X, tr, np.random.default_rng(5))
    best = max(
        float(M.fit_and_score(X, Y, tr, va, fmaps=(M._rff(d, 4096, g * m, seed=6),),
                              rng=np.random.default_rng(7))[0]["r2"][0])
        for m in M.BANDWIDTH_MULTS)
    assert best < 0.6, (
        f"an isotropic RBF over all {d} dims reached {best:.3f} on a 2-dim signal. If this now "
        f"passes comfortably the curse-of-dimensionality argument in the module docstring needs "
        f"revisiting -- but it must never be trusted as the sole capacity evidence.")


def test_masked_mean_does_not_drag_the_border_toward_the_fill_value():
    """``norm_grid`` zero-fills invalid cells and 0 is the post-norm mean, so a plain filter
    invents a gradient at every coast. The masked version must not."""
    M = _mod()
    pytest.importorskip("scipy")
    from scipy.ndimage import uniform_filter
    H, W, C = 12, 12, 2
    mask = np.zeros((H, W), bool)
    mask[:, 4:] = True                               # a straight "coast" at column 4
    grid = np.zeros((H, W, C), dtype="float32")
    grid[mask] = 5.0                                 # constant inside the valid region
    got = M._masked_mean(grid, mask, 3)
    naive = uniform_filter(grid, size=(3, 3, 1), mode="nearest")
    assert np.allclose(got[mask], 5.0, atol=1e-5), \
        f"a constant field must survive the masked mean; edge got {got[:, 4].min():.3f}"
    assert naive[:, 4, 0].min() < 4.0, "the naive filter is supposed to fail here"


def test_build_features_clamps_lagged_years_instead_of_dropping_early_points():
    """A point at the first available state year must still get its lag-15 columns.

    Dropping instead would delete exactly the early-era points the historical reconstruction
    exists for, and the script would silently measure predictability on the modern era only.
    """
    M = _mod()
    tmp = os.environ.get("PYTEST_TMP") or None
    import tempfile
    with tempfile.TemporaryDirectory(dir=tmp) as d:
        H, W = 6, 6
        schema = {"streams": [{"name": "a", "dim": 2, "start": 0, "end": 2}]}
        for y in (1940, 1955):
            np.savez(os.path.join(d, f"state_{y}.npz"),
                     a=np.full((H, W, 2), float(y), dtype="float32"))
        pidx = np.array([[1, 1, 1940], [2, 2, 1955]], dtype=int)
        mu = np.zeros(2, dtype="float32")
        sd = np.ones(2, dtype="float32")
        F, ok, cols = M.build_features(pidx, d, schema, mu, sd,
                                       blocks=((0, 1), (15, 1)), verbose=False)
        assert ok.all(), "no point may be dropped for an out-of-range lag"
        # 1940 is the earliest state, so its lag-15 must CLAMP to 1940, not vanish.
        assert F[0, cols[(0, 1)]][0] == pytest.approx(1940.0)
        assert F[0, cols[(15, 1)]][0] == pytest.approx(1940.0)
        # 1955's lag-15 lands exactly on 1940.
        assert F[1, cols[(0, 1)]][0] == pytest.approx(1955.0)
        assert F[1, cols[(15, 1)]][0] == pytest.approx(1940.0)


def test_verdict_reaches_opposite_conclusions_on_the_two_cases_it_exists_for(capsys):
    """Constructed R^2 curves must drive the two decisive branches, not a middling message."""
    M = _mod()
    L = 64
    share = np.full(L, 1.0 / L)

    def rung(r2):
        return {"r2": r2, "beats_const": np.ones(L, bool), "alpha": [1e-3] * L}

    # (a) collapses after 8 and NOTHING moves it: identical curve at every rung and capacity.
    dead = np.concatenate([np.full(8, 0.8), np.full(L - 8, 0.01)])
    M.verdict({k: rung(dead) for k in M.RUNGS}, {"rff4096": rung(dead)}, share)
    out = capsys.readouterr().out
    assert "NOT IN THE ENVIRONMENT" in out, out

    # (b) moderate out to 32 -> the encoder is the limit.
    alive = np.concatenate([np.full(32, 0.55), np.full(L - 32, 0.05)])
    M.verdict({k: rung(alive) for k in M.RUNGS}, {"rff4096": rung(alive)}, share)
    out = capsys.readouterr().out
    assert "THE ENCODER IS THE LIMIT" in out, out

    # A dead curve that RESPONDS to the ladders must NOT claim reading (a): that is the
    # false-negative the ladders exist to prevent.
    rungs = {"point": rung(dead), "+spatial": rung(dead),
             "+temporal": rung(np.concatenate([np.full(8, 0.8), np.full(L - 8, 0.20)]))}
    M.verdict(rungs, {"rff4096": rung(dead)}, share)
    out = capsys.readouterr().out
    assert "NOT IN THE ENVIRONMENT" not in out, out
    assert "the ladders DID move" in out, out


def test_main_runs_end_to_end_on_a_synthetic_project(tmp_path, monkeypatch, capsys):
    """Wiring smoke test: every path, config key and array alignment in ``main``, sections 1-9.

    The numerics are asserted above on synthetic targets; what this catches is the other half --
    a wrong config key, a column slice that does not line up with its block, the val-row index map
    the rank curve uses, a numpy array the JSON encoder cannot take, or one of the borrowed
    validation functions being handed the wrong shape. All of those fail only inside ``main``,
    which cannot be run locally against the real artifacts, so without this the first time the
    script is exercised is on TACC after an hour of state loading.

    The year span is deliberately wide enough (1970-2005) for ``ATTEN_GAP``'s 20-year pairs to
    exist, because sections 5 and 7 both silently no-op on a short record and a smoke test that
    skips two of the sections it exists to cover is not covering them.
    """
    M = _mod()
    # L=40 so the 9-16 / 17-32 bands the verdict thresholds read actually exist.
    H, W, C, S_ENV, S_NOISE, L = 24, 24, 4, 8, 6, 40
    S = S_ENV + S_NOISE
    rng = np.random.default_rng(0)

    states = tmp_path / "states" / "yearly_states"
    states.mkdir(parents=True)
    schema = {"streams": [{"name": "s", "dim": C, "start": 0, "end": C, "variables": list("abcd")}]}
    (states / "state_schema.json").write_text(__import__("json").dumps(schema))
    # States must reach 15 years before the earliest point year for the lag blocks.
    years = list(range(1970, 2006))
    grids = {}
    for y in range(1955, 2006):
        g = rng.normal(size=(H, W, C)).astype("float32") + 0.05 * (y - 1955)
        grids[y] = g
        np.savez(states / f"state_{y}.npz", s=g)

    pts = tmp_path / "points"
    pts.mkdir()
    pidx = np.array([(r, c, y) for y in years
                     for r in range(2, H - 2) for c in range(2, W - 2)], dtype=int)
    # Communities: the first S_ENV species are driven by the covariates at that cell-year, the
    # last S_NOISE are per-point noise. The stub below reads z off those two groups separately, so
    # components 0-3 are exactly predictable from the covariates and 4-39 are exactly not.
    base = np.stack([grids[int(y)][r, c] for r, c, y in pidx])
    A = rng.normal(size=(C, S_ENV))
    Xp = np.exp(np.concatenate([base @ A, rng.normal(size=(len(pidx), S_NOISE))], 1)) \
        .astype("float32")
    np.save(pts / "X_points.npy", Xp)
    np.save(pts / "point_index.npy", pidx)
    # recent_year is read from here, not from config -- the same source validate_spacetime uses.
    (pts / "points_meta.json").write_text(__import__("json").dumps(
        {"recent_year": 2005, "n_species": S}))

    zdir = tmp_path / "z"
    zdir.mkdir()
    (zdir / "meta.json").write_text(__import__("json").dumps({"latent_dim": L, "n_species": S}))
    # The stub must be a genuine FUNCTION OF ITS INPUT, not a precomputed array indexed by
    # position -- the same requirement tests/test_eigenbasis_diag.py states for the ESK rank
    # curve, and for the same reason. ``main`` subsamples BEFORE projecting, so a fixed array in
    # the original row order no longer lines up with the sample, and a positional stub reported
    # the constructed components as unpredictable. That is a bug in the test, indistinguishable
    # in the output from the finding the script exists to report.
    Benv = rng.normal(size=(S_ENV, 4))
    Bnoise = rng.normal(size=(S_NOISE, L - 4))

    def _proj(X, _zdir, ld):
        lg = np.log(np.maximum(np.asarray(X, dtype="float64"), 1e-12))
        z = np.concatenate([lg[:, :S_ENV] @ Benv, lg[:, S_ENV:] @ Bnoise], 1)
        return z[:, :ld].astype("float32")
    monkeypatch.setattr(M, "project_points_to_z", _proj)

    desk_out = tmp_path / "desk"
    desk_out.mkdir()
    cfg = {
        "paths": {"hist_dir": str(tmp_path / "states"), "desk_output_dir": str(desk_out)},
        "target": {"points_dir": str(pts)},
        "trend": {"points_dir": str(pts)},
        "desk": {"z_dir": str(zdir), "latent_dim": L, "label_year": 2005,
                 "spatial_conv": {"enabled": True, "kernel": 3},
                 "trend": {"block_cells": 4, "holdout_frac": 0.25, "seed": 0,
                           "buffer_floor": None, "holdout_years": []}},
    }
    monkeypatch.setattr(M, "load_config", lambda *_a, **_k: cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["component_predictability", "--max-rows", "8000", "--pairs", "200", "--pca-dim", "8",
         "--rff-width", "256", "--curve-points", "120"])
    M.main()
    out = capsys.readouterr().out
    res = __import__("json").loads((tmp_path / "component_predictability.json").read_text())

    assert "=== verdict ===" in out, out
    assert "THE ENCODER IS THE LIMIT" not in out, \
        "36 of 40 components are functions of noise only; that branch must not fire\n" + out
    assert len(res["var_share"]) == L
    assert set(res["context_ladder"]) == set(M.RUNGS)
    assert "esk_oracle" in res["rank_curves"]
    # The two constructed groups must come back separated: this is the alignment check, since a
    # block/column mismatch would blur them together.
    r2 = res["context_ladder"]["+temporal"]["r2"]
    assert min(r2[:4]) > 0.5, f"the four covariate-driven components were lost: {r2[:4]}"
    assert max(r2[4:]) < 0.2, f"noise-driven components came back predictable: {max(r2[4:]):.3f}"

    # --- every new section must have RUN, not silently no-opped ---------------------------
    d = res["decompositions"]
    for section in ("=== 5. NOISE CEILING", "=== 6. AGAINST THE REAL BARS",
                    "=== 7. LEVEL vs CHANGE", "=== 8. DIRECTION and MAGNITUDE",
                    "=== 9. WHERE the predictability lives"):
        assert section in out, f"{section} did not print\n{out}"
    assert "unavailable" not in out.split("=== 5.")[1].split("=== 6.")[0], \
        "section 5 no-opped; the fixture must span a 20-year gap\n" + out
    assert len(d["signal_noise"]["signal_var"]) == L
    assert len(d["achievable_r2_level"]) == L
    assert len(d["signal_share_level"]) == L and len(d["signal_share_change"]) == L
    assert len(d["r2_spatial_idw"]) == L and len(d["r2_spacetime_idw"]) == L
    assert d["change"]["n_pairs"] > 200, d["change"]
    # Section 7 must grade DESK's actual output quantity: the EMA'd trajectory, differenced.
    assert "EMA, differenced" in d["change"], \
        "section 7 skipped or did not run the EMA path\n" + out
    assert "r2_level_ema" in d["change"] and len(d["change"]["r2_level_ema"]) == L
    assert d["change"]["ema_half_life"] > 0
    assert set(d["banded_error_split"]) and all(
        v["n_dims"] >= 8 for v in d["banded_error_split"].values()), d["banded_error_split"]
    assert any(k.startswith("era ") for k in d["r2_by_group"]), d["r2_by_group"].keys()
    assert "baseline_panel" in d and "epoch_direction_panel" in d
    # The covariate-driven components must beat the interpolator and the noise ones must not --
    # the assertion that protects the headline this whole section exists to check.
    gain = np.array(d["r2_gain_over_best_bar"], dtype="float64")
    assert np.nanmean(gain[:4]) > 0.1, f"real signal must beat IDW: {gain[:4]}"


def test_the_rank_curve_is_not_fooled_by_shrinkage():
    """A shrunk copy of the SAME z must not beat the original.

    Ridge predictions are shrunk toward the mean, so their dot products are systematically small.
    Raw MSE rewards that whenever the truth sits closer to zero than the honest dot products do --
    a synthetic run had the weakest rung "beating" the ESK oracle 41.9 to 272.9. ``corr`` and the
    calibrated ``mse`` are what make the section readable; this asserts both, and asserts that the
    raw column still shows the artifact so the reason for the calibration stays visible.
    """
    M = _mod()
    rng = np.random.default_rng(0)
    # z must be a real function of X through a SHARED map, or the calibration has nothing to
    # transfer between the two samples and the fitted slope is pure noise (it came back negative,
    # which is not a case this diagnostic ever sees).
    Wm = rng.normal(size=(16, 8))
    X = np.abs(rng.gamma(2.0, 1.0, size=(240, 16)))
    Xc = np.abs(rng.gamma(2.0, 1.0, size=(240, 16)))
    z = (X / X.sum(1, keepdims=True)) @ Wm
    zc = (Xc / Xc.sum(1, keepdims=True)) @ Wm
    full = M.rank_curve(X, z, Xc, zc, ranks=(4, 8))
    shrunk = M.rank_curve(X, 0.1 * z, Xc, 0.1 * zc, ranks=(4, 8))
    assert shrunk[8]["corr"] == pytest.approx(full[8]["corr"], abs=1e-9), \
        "corr must be invariant to a pure rescale of z"
    assert shrunk[8]["mse"] == pytest.approx(full[8]["mse"], rel=1e-6), \
        "the calibration must undo a pure rescale"
    assert full[8]["scale"] > 0, f"a real z/Ružička relationship must calibrate positively; " \
                                 f"got {full[8]['scale']:.4g}"
    assert shrunk[8]["scale"] > 50 * full[8]["scale"], \
        ("the scale column is what makes shrinkage visible; a 0.1x rescale of z is a 0.01x "
         "rescale of its dot products, so the slope must rise ~100x")
    # The artifact, stated in the direction it actually occurs: two rescalings of the IDENTICAL
    # z have identical corr, yet raw MSE ranks them apart by pure magnitude. Whichever happens to
    # sit nearer the truth's scale "wins", which is how a shrunk regressor outscored the oracle.
    big = M.rank_curve(X, 20.0 * z, Xc, 20.0 * zc, ranks=(4, 8))
    small = M.rank_curve(X, 0.05 * z, Xc, 0.05 * zc, ranks=(4, 8))
    assert big[8]["corr"] == pytest.approx(small[8]["corr"], abs=1e-9)
    assert big[8]["mse"] == pytest.approx(small[8]["mse"], rel=1e-6)
    assert small[8]["mse_raw"] < 0.05 * big[8]["mse_raw"], (
        "raw MSE must be shown to rank two rescalings of the same z apart -- that is why the "
        "calibrated column and corr exist. If this fails the docstring's justification is stale.")


def test_a_flat_rank_curve_is_reported_as_flat_not_as_a_winner():
    """``best_rank`` must not invent a winner from a tie.

    A synthetic run printed ``bestR=48`` for a curve reading +0.183 at every rank. A real bestR of
    48 and a tie at 48 imply opposite things for ``latent_dim``, and this script's output feeds
    exactly that decision, so the tie has to survive to the report.
    """
    M = _mod()
    flat = {r: {"corr": 0.183, "mse": 0.5, "mse_raw": 0.5, "scale": 1.0} for r in (8, 16, 32, 64)}
    assert M.best_rank(flat, "corr")[0] is None
    assert M.best_rank(flat, "mse")[0] is None
    rising = {r: {"corr": 0.1 + 0.05 * i, "mse": 0.5 - 0.1 * i, "mse_raw": 0.5, "scale": 1.0}
              for i, r in enumerate((8, 16, 32, 64))}
    assert M.best_rank(rising, "corr") == (64, pytest.approx(0.15))
    assert M.best_rank(rising, "mse")[0] == 64
    # Noise below the floor must still read FLAT, in both directions.
    jitter = {r: {"corr": 0.183 + s, "mse": 0.5, "mse_raw": 0.5, "scale": 1.0}
              for r, s in zip((8, 16, 32, 64), (0.0, 0.004, -0.003, 0.002))}
    assert M.best_rank(jitter, "corr")[0] is None


def test_the_two_noise_ceilings_are_different_ratios_and_are_not_interchangeable():
    """A level ceiling and a change ceiling come from different quantities.

    The first version used ``signal_var/total_var`` -- which is the share of a fixed-gap DIFFERENCE
    -- to rescale a LEVEL R^2 and printed it as "R^2 vs ACHIEVABLE". A single observation carries
    sigma^2 while a difference of two carries 2*sigma^2, and they sit in different variances, so the
    two ratios cannot substitute for one another. Constructed with sigma^2 = 0.5 so both are known.
    """
    M = _mod()
    sn = {"noise_var": [1.0], "signal_var": [1.0], "total_var": [2.0]}
    assert M.change_signal_share(sn)[0] == pytest.approx(0.5)
    # sigma^2 = noise_var/2 = 0.5 against a level variance of 2.0 -> 75% real.
    assert M.level_signal_share(sn, np.array([2.0]))[0] == pytest.approx(0.75)
    # ... and against a level variance of 1.0 -> 50%. Same sn, different answer: the level ceiling
    # depends on the target's own variance, which the change ratio knows nothing about.
    assert M.level_signal_share(sn, np.array([1.0]))[0] == pytest.approx(0.5)
    assert M.change_signal_share({"note": "too few pairs"}) is None
    assert M.level_signal_share({"note": "too few pairs"}, np.array([1.0])) is None


def test_the_noise_ceiling_rescales_a_noisy_component_to_its_achievable_r2():
    """A component that is mostly survey noise must stop looking like a covariate failure.

    This is the correction that can overturn a raw-R^2 reading: no covariate can predict noise, so
    reporting a noise-dominated component's low R^2 as a missing-covariate gap is a category error.
    Constructed so the answer is known -- signal share 0.25 and a raw R^2 of 0.25 is a model that
    captured ALL of the achievable signal, and must be reported at ~1.0.
    """
    M = _mod()
    ach = M.achievable_r2(np.array([0.25, 0.45]), np.array([0.25, 0.9]))
    assert ach[0] == pytest.approx(1.0), f"a fully-captured noisy component must read 1.0: {ach[0]}"
    assert ach[1] == pytest.approx(0.5), f"0.45 of an 0.9 signal share is 0.5: {ach[1]}"
    # Capped at 1.0: a ratio above 1 means the noise estimate is imprecise, not that the model
    # beat the ceiling, and reporting 1.4 would invite exactly that misreading.
    assert M.achievable_r2(np.array([0.9]), np.array([0.5]))[0] == pytest.approx(1.0)
    assert M.achievable_r2(np.array([0.3]), None) is None
    # BOTH bounds, and the refusal. Dividing a negative R^2 by a tiny share is how a synthetic run
    # printed "R^2 vs ACHIEVABLE -215.144" from a raw R^2 of -0.0005.
    a_lo = M.achievable_r2(np.array([-0.0005, -0.4]), np.array([0.006, 0.9]))
    assert not np.isfinite(a_lo[0]), f"a 0.6% signal share must be refused, not scaled: {a_lo[0]}"
    assert a_lo[1] == pytest.approx(0.0), \
        f"a negative R^2 over a real share floors at 0, not below: {a_lo[1]}"
    ok = M.achievable_r2(np.array([0.03]), np.array([M.MIN_SIGNAL_SHARE]))[0]
    assert np.isfinite(ok), "exactly at MIN_SIGNAL_SHARE the rescale must still apply"


def test_a_spatially_smooth_component_is_beaten_by_the_idw_bar():
    """The assertion that protects the headline this decomposition exists to check.

    A component that is nothing but spatial smoothness is 'predictable' from any smooth covariate,
    and an R^2 against its own mean will say so. Only a comparison against inverse-distance
    interpolation separates that from environmental signal. Built two ways in one array: column 0
    is a pure function of position, column 1 a pure function of a covariate that is spatial noise.
    """
    M = _mod()
    rng = np.random.default_rng(3)
    rows, cols, years = [], [], []
    for y in (2000, 2001):
        for r in range(20):
            for c in range(20):
                rows.append(r); cols.append(c); years.append(y)
    pidx = np.stack([rows, cols, years], 1).astype(int)
    smooth = 0.05 * pidx[:, 0] + 0.03 * pidx[:, 1]              # a plane: IDW nails it
    rough = rng.normal(size=len(pidx))                          # white noise in space
    Y = np.stack([smooth, rough], 1).astype("float64")
    # Held-out cells SCATTERED through the interior, each ringed by training cells. The two
    # obvious alternatives both break the measurement rather than the method: an edge strip makes
    # IDW extrapolate a gradient it can only average (scored 0.82), and one interior block leaves
    # the val set spanning so little of the plane that R^2's own denominator collapses (0.53).
    # Scattering keeps every val cell interpolable AND keeps the full range of the plane in the
    # val variance.
    holdout = np.zeros((20, 20), bool)
    holdout[1:19:3, 1:19:3] = True
    va_mask = holdout[pidx[:, 0], pidx[:, 1]]
    _err, zi = M.zspace_idw_baseline(pidx, Y, holdout, va_mask, return_z=True)
    r2, n = M.per_component_r2_of(zi, Y[va_mask])
    assert n[0] > 30
    assert r2[0] > 0.9, f"IDW must recover a spatial plane; got {r2[0]:.3f}"
    assert r2[1] < 0.2, f"IDW must NOT recover spatial white noise; got {r2[1]:.3f}"


def test_level_and_change_are_measured_as_different_quantities():
    """``gap_pairs`` must find fixed-gap within-cell pairs, and only those.

    Section 7 rests entirely on this pairing. A per-cell-span pairing would rank cells by RECORD
    LENGTH instead of measuring change over a fixed interval, which is the mistake recorded on
    ``per_era_attenuation``, so the gap is asserted exactly.
    """
    M = _mod()
    pidx = np.array([[0, 0, 1970], [0, 0, 1990], [0, 0, 1991], [0, 0, 2010],
                     [1, 1, 1980], [1, 1, 1985],                 # 5 yr apart: no pair
                     [2, 2, 1970], [2, 2, 1988]], dtype=int)     # 18 yr: inside 20+/-2
    ea, la = M.gap_pairs(pidx, gap=20, tol=2)
    got = {(int(pidx[a, 2]), int(pidx[b, 2]), int(pidx[a, 0])) for a, b in zip(ea, la)}
    assert (1970, 1990, 0) in got, got
    assert (1970, 1988, 2) in got, got
    assert not any(r == 1 for _y0, _y1, r in got), f"a 5-year gap must not pair: {got}"
    for y0, y1, _r in got:
        assert 18 <= y1 - y0 <= 22, f"gap {y1 - y0} outside 20+/-2: {got}"
    # One pair per (cell, earlier year): 1990 must not also pair with 2010 AND appear twice.
    assert len(ea) == len(set(zip(ea.tolist(), la.tolist())))


def test_direction_is_reported_per_band_and_never_per_component():
    """An angle needs >=2 dimensions; a one-column 'direction' is sign agreement.

    ``banded_direction`` must therefore keep bands at their real width and the angular term must be
    a genuine fraction of the error. Asserted both ways: a pure scale error is all magnitude and no
    angle, and a rotation is all angle and no magnitude.
    """
    M = _mod()
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(500, 16))
    scaled = 0.5 * truth                                   # same direction, wrong length
    got = M.banded_direction(scaled, truth, bands=((0, 8), (8, 16)))
    assert set(got) == {"1-8", "9-16"}
    for v in got.values():
        assert v["n_dims"] == 8, "bands must keep their width; a width-1 band cannot hold an angle"
        assert v["mag_share"] > 0.99, f"a pure rescale is all magnitude: {v}"
        assert v["ang_share"] < 0.01, f"a pure rescale has no angular error: {v}"
        assert v["norm_ratio"] == pytest.approx(0.5, abs=1e-6)
    # A norm-preserving perturbation is the other extreme.
    other = rng.normal(size=(500, 16))
    rot = other / np.linalg.norm(other, axis=1, keepdims=True) \
        * np.linalg.norm(truth, axis=1, keepdims=True)
    got2 = M.banded_direction(rot, truth, bands=((0, 16),))["1-16"]
    assert got2["ang_share"] > 0.9, f"a norm-preserving error is all angular: {got2}"
    assert got2["norm_ratio"] == pytest.approx(1.0, abs=0.05)


def test_the_no_change_null_never_silently_borrows_row_zero():
    """A cell with no ``recent_year`` row must come back unflagged, not pointing at row 0.

    The null is inlined in three places in the validation suite and copied here; the ``-1``
    sentinel plus the ``has_rec`` gate is the whole reason the copy is safe, so it is pinned.
    """
    M = _mod()
    pidx = np.array([[0, 0, 1990], [0, 0, 2005],       # has a recent row
                     [1, 1, 1990]], dtype=int)         # does NOT
    to_rec, has_rec = M.nochange_rows(pidx, 2005)
    assert has_rec.tolist() == [True, True, False]
    assert to_rec[0] == 1 and to_rec[1] == 1
    assert to_rec[2] == -1, f"a cell with no recent year must be -1, not 0: {to_rec}"


def test_differencing_a_level_model_beats_refitting_on_differences():
    """The error-cancellation section 7 originally threw away, isolated.

    A level model with a STATIC per-cell bias predicts change perfectly once differenced, because
    the bias cancels. A model refit on differenced features cannot recover that -- it never sees
    the level. The first version of section 7 did the second and reported the result as a statement
    about covariate sufficiency for change, which is why its answer disagreed with DESK's own
    validated temporal skill.
    """
    M = _mod()
    rng = np.random.default_rng(11)
    n_cells, T = 60, 2
    cell = np.repeat(np.arange(n_cells), T)
    x = rng.normal(size=(len(cell), 3))
    bias = rng.normal(size=n_cells)[cell] * 3.0        # large, static, per cell
    truth = (x @ np.array([1.0, -0.5, 0.25]))[:, None]
    level_pred = truth + bias[:, None]                 # a good model with a static offset
    ea = np.arange(0, len(cell), 2)
    la = ea + 1
    dY = truth[la] - truth[ea]
    r2_diff_of_level = M.per_component_r2_of(level_pred[la] - level_pred[ea], dY)[0][0]
    r2_of_level = M.per_component_r2_of(level_pred, truth)[0][0]
    assert r2_diff_of_level > 0.99, \
        f"a static per-cell bias must cancel under differencing; got {r2_diff_of_level:.3f}"
    assert r2_of_level < 0.5, \
        f"the same model must look POOR on level, which is the whole point: {r2_of_level:.3f}"


def test_ema_trajectories_applies_the_learned_smoothing_over_contiguous_years():
    """The EMA must actually smooth, be causal, and be keyed so a cell-year can be found.

    ``apply_output_ema`` cannot run on isolated cell-years -- it needs an ascending contiguous
    series per cell -- and getting that wrong silently yields an unsmoothed prediction that looks
    like a covariate finding. Uses a constant-in-space, alternating-in-time signal so the smoothing
    is visible as a variance drop and the causal direction is checkable.
    """
    M = _mod()
    H, W, C, L = 6, 6, 2, 3
    schema = {"streams": [{"name": "s", "dim": C, "start": 0, "end": C, "variables": ["a", "b"]}]}
    import tempfile, json as _j
    with tempfile.TemporaryDirectory() as d:
        years = list(range(1990, 2011))
        for i, y in enumerate(years):                   # alternating +/-1 in time
            np.savez(os.path.join(d, f"state_{y}.npz"),
                     s=np.full((H, W, C), 1.0 if i % 2 == 0 else -1.0, dtype="float32"))
        (open(os.path.join(d, "state_schema.json"), "w")).write(_j.dumps(schema))
        cells = np.array([[2, 2], [3, 3]])
        nb = len(M.BLOCKS) * C
        model = {"L": L, "train_mean": np.zeros(L), "parts": [{
            "fmap": None, "coef": np.tile(np.eye(L, dtype="float64"), (nb // L + 1, 1))[:nb],
            "mx": np.zeros(nb), "my": np.zeros(L), "cols": np.arange(L)}]}
        mu, sd = np.zeros(C, "float32"), np.ones(C, "float32")
        z_ema, z_raw, tix = M.ema_trajectories(
            model, cells, years, d, schema, mu, sd,
            np.zeros(nb, "float32"), np.ones(nb, "float32"), half_life=8.0)
        assert z_ema.shape == (len(years), len(cells), L)
        assert (2, 2, 2000) in tix and tix[(2, 2, 1990)][0] == 0
        # Smoothing: the alternating raw signal must lose variance through the EMA.
        assert z_ema[:, 0, 0].var() < 0.5 * z_raw[:, 0, 0].var(), \
            f"the EMA did not smooth: raw var {z_raw[:, 0, 0].var():.4f} " \
            f"ema var {z_ema[:, 0, 0].var():.4f}"
        # Causal: the first entry is the raw value, never a blend of the future.
        assert z_ema[0, 0, 0] == pytest.approx(z_raw[0, 0, 0], abs=1e-5)
