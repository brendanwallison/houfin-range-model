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
    """Wiring smoke test: every path, config key and array alignment in ``main``.

    The numerics are asserted above on synthetic targets; what this catches is the other half --
    a wrong config key, a column slice that does not line up with its block, the val-row index
    map used by the rank curve, or a numpy scalar that will not serialize. All of those fail only
    inside ``main``, which cannot be run locally against the real artifacts, so without this the
    first time the script is exercised is on TACC after an hour of state loading.
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
    years = list(range(1990, 2001))
    grids = {}
    for y in years:
        g = rng.normal(size=(H, W, C)).astype("float32") + 0.05 * (y - 1990)
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
        "desk": {"z_dir": str(zdir), "latent_dim": L, "label_year": 2000,
                 "spatial_conv": {"enabled": True, "kernel": 3},
                 "trend": {"block_cells": 4, "holdout_frac": 0.25, "seed": 0,
                           "buffer_floor": None}},
    }
    monkeypatch.setattr(M, "load_config", lambda *_a, **_k: cfg)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["component_predictability", "--max-rows", "6000", "--pairs", "200", "--pca-dim", "8",
         "--rff-width", "256", "--curve-points", "120"])
    M.main()
    out = capsys.readouterr().out
    assert "=== verdict ===" in out, out
    assert "THE ENCODER IS THE LIMIT" not in out, \
        "36 of 40 components are functions of noise only; that branch must not fire\n" + out
    res = __import__("json").loads((tmp_path / "component_predictability.json").read_text())
    assert len(res["var_share"]) == L
    assert set(res["context_ladder"]) == set(M.RUNGS)
    assert "esk_oracle" in res["rank_curves"]
    # The two constructed components must come back predictable and the noise must not: this is
    # the alignment check, since a block/column mismatch would blur them together.
    r2 = res["context_ladder"]["+temporal"]["r2"]
    assert min(r2[:4]) > 0.5, f"the four covariate-driven components were lost: {r2[:4]}"
    assert max(r2[4:]) < 0.2, f"noise-driven components came back predictable: {max(r2[4:]):.3f}"


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
