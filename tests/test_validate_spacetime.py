"""Report plumbing for validate_spacetime."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.validate_spacetime import (
    RECON_ARRAY_KEYS, report_scalars)


def test_new_reconstruction_scalars_reach_the_report():
    """Regression. The key filter was an ALLOW-LIST, so the interpolation bar was computed,
    never listed, and could never print -- the printed summary reads this filtered dict, not the
    original. Any new scalar must survive by default."""
    recon = {"n": 10, "recent_basis_residual": 1e-6, "shrinkage_slope": -0.01,
             "some_future_bar": 1.23,
             # the predictor table is NESTED, and an allow-list of scalar names would drop it
             # whole -- which is the same failure at a larger scale than the one that motivated
             # this regression, since it is now where every baseline's numbers live.
             "absolute_position": {"n": 10, "predictors": {"desk": {"median_err": 0.4},
                                                           "spacetime_idw": {"median_err": 0.42}},
                                   "win_rate_vs": {"desk": 0.51}},
             **{k: np.zeros(10) for k in RECON_ARRAY_KEYS}}
    kept = report_scalars(recon)
    for k in ("shrinkage_slope", "some_future_bar", "n", "absolute_position"):
        assert k in kept, k
    assert kept["absolute_position"]["predictors"]["spacetime_idw"]["median_err"] == 0.42


def test_the_per_point_arrays_stay_out_of_the_json():
    """They are large and go to the .npz for viz; in the report they would bloat it and break
    json.dumps."""
    import json
    kept = report_scalars({"n": 3, **{k: np.zeros(3) for k in RECON_ARRAY_KEYS}})
    assert set(kept) == {"n"}
    json.dumps(kept)


def test_every_desk_z_ema_call_site_unpacks_two_values():
    """A static check, because this bug class cannot be caught dynamically here.

    desk_z_ema returns ``(Z, metadata)``. Two call sites -- both added this session, in modules
    that need the full data tree to run -- bound the tuple to a single name and died only on
    TACC, one of them after burning two minutes of GPU time. There is no local fixture that
    would exercise them, so the contract is checked in the source instead.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "desk_z_ema"):
                continue
            tgt = node.targets[0]
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, ("desk_z_ema returns (Z, metadata); these bind it to one name: "
                          + ", ".join(offenders))


def test_shrinkage_profile_separates_a_uniform_rescale_from_a_tilt():
    """The distinction the whole diagnostic exists for. A uniform norm deficit is harmless -- the
    downstream's fitted w_env absorbs a constant rescale -- while shrinkage concentrated in the
    low-eigenvalue directions tilts the kernel toward spatial similarity and distorts the GP
    prior. Aggregate ||z||^2 is identical in both cases, so only the per-dimension profile can
    tell them apart.

    Also pins the companion fact: a uniform rescale puts ALL the error in the magnitude term,
    because scaling a vector cannot change its direction. If a flat profile ever came with a
    non-zero angular share, the decomposition would be mis-wired.
    """
    import numpy as np

    from src.community_encoder.train_DESK.validate_baselines import error_decomposition

    rng = np.random.default_rng(0)
    n, L = 3000, 32
    z_obs = rng.normal(size=(n, L)) * np.linspace(1.0, 0.3, L)     # eigen-ordered variance

    def profile(scale):
        z_desk = z_obs * scale
        tot, mag, ang, _cos = error_decomposition(z_desk, z_obs)
        assert np.allclose(mag + ang, tot, atol=1e-9)              # identity must hold
        prof = np.var(z_desk, 0) / np.var(z_obs, 0)
        slope = float(np.polyfit(np.arange(L), prof, 1)[0])
        return slope, float(np.median(prof)), float(mag.mean() / tot.mean())

    flat_slope, flat_med, flat_mag_share = profile(np.full(L, 0.8))
    assert abs(flat_slope) < 1e-6, flat_slope
    assert abs(flat_med - 0.64) < 0.02, flat_med          # variance ratio, so c^2 not c
    assert flat_mag_share > 0.999, flat_mag_share         # a rescale is pure magnitude error

    tilt_slope, _tilt_med, tilt_mag_share = profile(np.linspace(1.0, 0.2, L))
    assert tilt_slope < -0.01, tilt_slope                 # clearly negative
    assert tilt_mag_share < 0.8, tilt_mag_share           # a tilt introduces angular error


def test_the_absolute_position_table_prints():
    """A broken f-string in the summary would otherwise surface only at the end of a multi-hour
    job, which is why the printer is a module-level function rather than inline in run_validate."""
    import io, contextlib
    from src.community_encoder.train_DESK.validate_spacetime import print_absolute_position
    from src.community_encoder.train_DESK.validate_bbs_routes import compare_positions
    rng = np.random.default_rng(1)
    z = rng.normal(size=(300, 8))
    preds = {"desk": z + rng.normal(scale=.3, size=z.shape),
             "no_change": z + rng.normal(scale=.6, size=z.shape),
             "spacetime_idw": z + rng.normal(scale=.4, size=z.shape),
             "zspace_idw": None}
    pops = {"heldout": np.arange(300) < 120, "train": np.arange(300) >= 120}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_absolute_position({"recent_basis_residual": 1.2e-6,
                                 "absolute_position": compare_positions(z, preds,
                                                                        populations=pops)})
    txt = buf.getvalue()
    for want in ("desk", "no_change", "spacetime_idw", "heldout", "train"):
        assert want in txt, (want, txt)
    assert "not scored" in txt                    # the unbuilt bar states its reason
    # an empty report must print nothing rather than raise
    with contextlib.redirect_stdout(io.StringIO()) as empty:
        print_absolute_position({})
    assert empty.getvalue() == ""


def test_parallel_movement_is_distinguishable_from_no_movement():
    """The whole reason the co-movement curve exists.

    Two places can both move a long way in the SAME direction and end up exactly as similar as
    they started. The convergence measure reads that as nothing happening; this must read it as
    strong agreement. If the two agreed, one of them would be redundant.
    """
    from src.community_encoder.train_DESK.validate_spacetime import _co_movement_by_distance
    rng = np.random.default_rng(0)
    n, n_anchor = 200, 40
    rows = rng.integers(0, 40, n)
    cols = rng.integers(0, 40, n)

    shared = rng.normal(size=(1, n_anchor))
    parallel = np.repeat(shared, n, axis=0) + rng.normal(scale=0.05, size=(n, n_anchor))
    scattered = rng.normal(size=(n, n_anchor))

    par = _co_movement_by_distance(parallel, parallel, rows, cols, rng)
    sca = _co_movement_by_distance(scattered, scattered, rows, cols, rng)
    assert par["bins"] and sca["bins"]
    assert np.mean([b["observed_co_movement"] for b in par["bins"]]) > 0.9   # all moving together
    assert abs(np.mean([b["observed_co_movement"] for b in sca["bins"]])) < 0.2  # unrelated


def test_a_model_that_oversmooths_reads_above_the_observed_curve():
    """The expected failure for a smooth function of smooth covariates: neighbouring places all
    given much the same predicted change while the real ones differ."""
    from src.community_encoder.train_DESK.validate_spacetime import _co_movement_by_distance
    rng = np.random.default_rng(1)
    n, n_anchor = 200, 40
    rows, cols = rng.integers(0, 40, n), rng.integers(0, 40, n)
    observed = rng.normal(size=(n, n_anchor))                 # every place moves its own way
    smooth = np.repeat(rng.normal(size=(1, n_anchor)), n, axis=0)   # model gives everyone the same
    r = _co_movement_by_distance(smooth, observed, rows, cols, rng)
    gaps = [b["gap"] for b in r["bins"]]
    assert all(g > 0.5 for g in gaps), gaps       # model curve well ABOVE observed at every range
    assert len(r["bins"]) >= 4


def test_the_co_movement_curve_needs_enough_sites():
    from src.community_encoder.train_DESK.validate_spacetime import _co_movement_by_distance
    rng = np.random.default_rng(0)
    r = _co_movement_by_distance(np.zeros((5, 3)), np.zeros((5, 3)),
                                 np.arange(5), np.arange(5), rng)
    assert "too few" in r["note"]
