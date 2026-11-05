"""The FULL model must initialize and take an optimizer step, NaN-free, under every overlay.

Unit tests exercise the pieces; this runs build_model_2d end to end on a small synthetic
input set shaped like load_data's output, with JAX_DEBUG_NANS on, exactly as age_run_map
starts (AutoDelta, svi.init at prior_scale 0.1, then one update). Several model changes
first failed on TACC because no test did this (e.g. fixed_coupling = 1 made a reporting
Cholesky NaN, which JAX_DEBUG_NANS then reported instead of the real failure).

Each overlay runs in a subprocess: age_priors reads its config at import.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OVERLAYS = [None, "map_run19_exchangeable.json", "map_run21_single_field.json",
            "map_run22_relative_allee.json", "map_run23a_weak_allee_juv_ratio.json",
            "map_run24c_pin_both.json", "map_run25_ideal_free.json",
            "map_run26_age_departure.json",
            "map_run27_crowding_departure.json",
            "map_run28a_fitness_settle.json", "map_run28b_uniform_settle.json",
            "map_run28c_capacity_fitness_settle.json",
            "map_run29_subcell_strata.json",
            "map_run30_single_release.json",
            "map_run31_disease_mortality.json",
            "map_run32_kernel_tail.json",
            "map_run33_year_effects.json",
            "map_run34_breeding_obs.json",
            "map_run35_candidate.json",
            "map_run36_disperser_settlement.json",
            "map_run37a_r0_settle.json", "map_run37b_pure_r0_settle.json",
            "map_run38_habitat_shrink.json",
            "map_run39_nb_density_conc.json",
            "map_run40_phase1.json",
            "map_run41_route_effects.json",
            "map_run50a_lit_dispersal.json", "map_run50b_lit_dispersal.json",
            "map_run51a_veit.json", "map_run51b_veit.json",
            "map_run52a_yearling.json", "map_run52b_yearling.json",
            "map_run53a_veit_repro.json", "map_run53b_veit_repro.json",
            "map_run55a_allee_pinned.json", "map_run55b_allee_pinned.json", "map_run55c_allee_pinned.json",
            "map_run55d_allee_pinned.json", "map_run55e_allee_pinned.json",
            "map_run56a_ktrend_loose.json", "map_run56b_ktrend_loose.json",
            "map_run58a_subcell.json", "map_run59a_route_pinned.json", "map_run60a_k_amp.json",
            "map_run60b_coupling.json", "map_run61a_fitness_settle.json",
            "map_run57a_drift_p0.json", "map_run57n_drift_null.json", "map_run57r_drift_recovery.json",
            "map_run62a_numerics.json", "map_run62b_numerics.json",
            "map_run63a_occupancy.json", "map_run63b_occupancy.json", "map_run63c_occupancy.json", "map_run63f_occupancy.json",
            "map_run59b_route_ktrend.json",
            "map_run57p_popd_cap.json", "map_run57q_popd_surv.json", "map_run57b_builtup_cap.json",
            "map_run57g_popd_cap_cosine.json", "map_run57z_builtup_cap_plummer.json", "map_run57x_panel.json",
            "map_run57xn_panel_null.json", "map_run57xr_panel_recovery.json",
            "map_run58b_subcell_surv.json", "map_run58c_subcell_surv.json",
            "map_run64a_subcell_smooth.json", "map_run64h_survival_harsh.json",
            "map_run64j_compensation_route.json", "map_run64k_route_control.json",
            "map_run64l_route_ktrend_control.json",
            "map_run65d_survival_nosort.json", "map_run65h_survival_harsh_nosort.json",
            "map_run65j_compensation_route_nosort.json",
            "map_run66a_readout.json", "map_run66b_readout.json", "map_run66c_readout.json", "map_run66d_readout.json", "map_run66e_readout.json", "map_run66f_readout.json", "map_run66g_readout.json", "map_run66h_readout.json", "map_run66i_readout.json", "map_run66j_readout.json", "map_run66k_readout.json", "map_run66l_readout.json",
            "map_run67a_st_correction.json", "map_run67c_st_correction.json", "map_run67n_st_correction.json",
            "map_run67r_st_correction.json", "map_run66m_readout.json", "map_run69a_b1_nofloor.json", "map_run69b_b1_smoothfloor.json", "map_run68b_plausible_truth_nofloor.json", "map_run69c_b3_nofloor.json", "map_run70z_holdout_base.json", "map_run70b_process_noise.json", "map_run70d_process_noise.json", "map_run71b_allee_sensitivity.json", "map_run71c_allee_sensitivity.json", "map_run74a_settlement.json", "map_run74b_settlement.json", "map_run74c_settlement.json", "map_run74e_settlement.json", "map_run75a_retest.json", "map_run75b_retest.json", "map_run77a_native_departure.json", "map_run80b_temporal.json", "map_run81b_desk_cbc1902.json", "map_run80g_temporal.json", "map_run80h_temporal.json", "map_run78b_combo.json", "map_run78o_combo.json", "map_run78f_combo.json", "map_run72a_route_occupancy.json", "map_run73sh_lambda_t.json", "map_run73h_lambda_t.json",
            "map_run67a2_st_correction.json", "map_run67c2_st_correction.json", "map_run67n2_st_correction.json", "map_run67r2_st_correction.json",
            "map_obsroute_effect.json"]


def _DRIFT_C():
    """Columns of the synthetic drift panel the active overlay asks for (1 for single-covariate arms)."""
    from src.config_utils import load_age_model_config
    spec = load_age_model_config().get("population_model", {}).get("covariate_drift") or {}
    return max(1, len(spec.get("covariates", [spec.get("covariate", "builtup")])))


def load_age_model_config_pn():
    from src.config_utils import load_age_model_config
    return dict(load_age_model_config().get("population_model", {}).get("process_noise") or {"n_freq": 2})


def synthetic_inputs(T=40, Ny=12, Nx=16, M=24, z_sd=2.0, seed=0):
    import jax.numpy as jnp
    from src.model.build_kernels import build_simulation_struct
    from src.model.data_loading import occupied_habitat_center
    rng = np.random.default_rng(seed)
    land = np.ones((Ny, Nx)); land[:, :2] = 0.0; land[:2, -4:] = 0.0
    ss = build_simulation_struct(jnp.asarray(land), 27.0, 100.0, 330.0, 0.859, 0.468,
                                 [0.0, 132.6, 330.7, 1e9])
    nk = ss["juvenile_fft_kernel_stack"].shape[0]
    rows, cols = np.nonzero(land); n = rows.size
    Z = np.abs(rng.normal(0.5, z_sd, size=(T, n, M))).astype(np.float32)
    nobs = 300
    t_idx = rng.integers(5, T, nobs); oc = rng.integers(0, n, nobs)
    initpop = np.zeros((Ny, Nx)); initpop[2:8, 2:7] = 0.5; initpop[3:7, 2:6] = 1.0
    meta = dict(
        Ny=Ny, Nx=Nx, time=T, M=M, K=nk, N_land=n, land_mask=land, land_rows=rows, land_cols=cols,
        Z_gathered=Z, Z_disp_gathered=np.repeat(Z[:, :, None, :], nk, axis=2) * 0.95,
        N_sev_basis=24, N_lag_basis=24,
        disease_sev_basis=rng.normal(0, 0.1, (24, n)), disease_lag_basis=rng.normal(0, 0.1, (24, n)),
        disease_onset=np.full(n, 30.0) + rng.normal(0, 3, n), disease_onset_decades=rng.uniform(0, 2, n),
        disease_timestep=28,
        k_trend_basis=np.stack([np.cos(np.pi * k * (np.arange(T) + 0.5) / T) for k in (1, 2, 3)]),
        initpop_latent=initpop, inv_locations=np.array([[Ny // 2, Nx - 4], [Ny // 2 + 1, Nx - 4]]),
        inv_window=10, inv_timestep=12, pop_scalar=20.0, dispersal_target_fraction=0.8,
        adult_fft_kernel=np.asarray(ss["adult_fft_kernel"]),
        juvenile_fft_kernel_stack=np.asarray(ss["juvenile_fft_kernel_stack"]),
        adult_edge_correction=np.asarray(ss["adult_edge_correction"]),
        juvenile_edge_correction_stack=np.asarray(ss["juvenile_edge_correction_stack"]),
        obs_time_indices=t_idx, obs_rows=rows[oc], obs_cols=cols[oc],
        observed_results=rng.poisson(1.0, nobs) * (rng.random(nobs) < 0.5),
        obs_quality=(rng.random(nobs) < 0.1).astype(np.int64),
        z_kernel_contract={"kernel": "ruzicka", "centered": False, "feature_prior": "isotropic",
                           "latent_dim": M, "source_latent_dim": 64, "truncation": "top_eigenfeatures"},
        age_structure_prior={"alpha": 1.01, "beta": 1.01, "effective_sites": 100.0},
        # within-cell urbanisation slices (only read when population_model.subcell_strata is on);
        # the last 3 cells uncovered (u = 0 in every slice), as outside BUI's CONUS footprint
        strata_u=np.concatenate([np.sort(rng.normal(0, 1, (T, n - 3, 6)), axis=-1),
                                 np.zeros((T, 3, 6))], axis=1).astype(np.float32),
        strata_a=np.array([0.15, 0.225, 0.25, 0.2, 0.12, 0.055]))
    meta["obs_route"] = np.where(rng.random(nobs) < 0.2, -1, rng.integers(0, 40, nobs))  # route_effects
    meta["n_routes"] = 40
    # observation.observer_effect (when enabled): observer x route ids and first-year flags; the
    # route-less pseudo-zeros carry -1 / 0, as bbs.py writes them
    meta["obs_obsroute_idx"] = np.where(meta["obs_route"] < 0, -1, rng.integers(0, 60, nobs))
    meta["obs_first_year"] = np.where(meta["obs_obsroute_idx"] < 0, 0, rng.integers(0, 2, nobs))
    meta["n_obsroute"] = int(meta["obs_obsroute_idx"].max()) + 1
    meta["Z_mean"] = meta["Z_gathered"].mean(axis=0)              # habitat_change (when enabled)
    meta["Z_disp_mean"] = meta["Z_disp_gathered"].mean(axis=0)
    meta["n_obs_quality_tiers"] = int(np.unique(meta["obs_quality"]).size)
    meta["Z_ref"] = meta["Z_gathered"].mean(axis=0)                 # readout.split (when enabled)
    meta["Z_disp_ref"] = meta["Z_disp_gathered"].mean(axis=0)
    from src.model.data_loading import covariate_drift_basis           # covariate_drift (when enabled)
    meta["years"] = np.arange(2024 - T, 2024)
    meta["drift_basis"] = covariate_drift_basis(meta["years"], t_idx, 3, 1966)
    from src.model.data_loading import covariate_drift_ramp
    meta["subcell_drift_g"] = covariate_drift_ramp(meta["years"], t_idx, 1966, float(meta["years"][-1]), 1966)[0]
    meta["journey_water_W"] = rng.uniform(0, 1, (n, nk)).astype(np.float32)
    # path_compounding (when enabled): 4 segments scattered about the path mean; kernel mean distances
    _seg_noise = rng.normal(0, 0.5, (T, n, nk, 4, M)).astype(np.float32)
    meta["Z_disp_seg_gathered"] = (meta["Z_disp_gathered"][:, :, :, None, :] + _seg_noise
                                   - _seg_noise.mean(axis=3, keepdims=True))
    meta["kernel_mean_km"] = np.tile(np.array([80.0, 286.0, 867.0]), nk // 3)   # journey.water_hazard (when enabled)
    _nb = 8                                                          # st_correction (when enabled)
    meta["st_S"] = rng.normal(0, 1, (n, _nb)).astype(np.float32)
    meta["st_S_disp"] = rng.normal(0, 1, (n, nk, _nb)).astype(np.float32)
    meta["st_g"] = covariate_drift_ramp(meta["years"], t_idx, 1966, float(meta["years"][-1]), 1966)
    from src.model.data_loading import process_noise_basis            # process_noise (when enabled)
    _pn = load_age_model_config_pn()
    meta["pn_S"], meta["pn_active"] = process_noise_basis(dict(meta, Ny=Ny, Nx=Nx), _pn)
    if _pn.get("detrend_degree") is not None:
        from src.model.data_loading import process_noise_detrend
        meta["pn_P"] = process_noise_detrend(meta["years"], meta["pn_active"], int(_pn["detrend_degree"]))
    from src.model.data_loading import process_noise_groups as _png
    meta["pn_groups"], meta["pn_n_groups"] = _png(dict(meta, Ny=Ny, Nx=Nx), _pn)
    meta["pn_penalty_years"] = np.ones(len(meta["years"]))
    meta["warp_S"], meta["warp_active"] = process_noise_basis(dict(meta, Ny=Ny, Nx=Nx),
                                                              {"n_freq": 3, "window_start": int(meta["years"][0])})
    meta["skel_penalty_years"] = np.ones(len(meta["years"]))
    from src.model.age_priors import _hat_basis as _hb
    meta["band_time_B"] = _hb(np.asarray(meta["years"]), [int(meta["years"][0]), int(meta["years"][-1])])
    meta["anchor_t"] = 2
    meta["anchor_A"] = rng.random(n).astype(np.float32) * 0.05
    meta["anchor_s"] = (rng.random(n) > 0.5).astype(np.float32)
    from src.model.data_loading import process_noise_detrend as _pnd   # obs_region_effects (when enabled)
    meta["ore_S"], meta["ore_active"] = process_noise_basis(dict(meta, Ny=Ny, Nx=Nx),
                                                            {"n_freq": 3, "window_start": int(meta["years"][0])})
    meta["ore_P"] = _pnd(meta["years"], meta["ore_active"], 1)
    _lut = -np.ones((Ny, Nx), np.int64)
    _lut[np.asarray(meta["land_rows"]), np.asarray(meta["land_cols"])] = np.arange(len(meta["land_rows"]))
    meta["obs_land_idx"] = _lut[np.asarray(meta["obs_rows"]), np.asarray(meta["obs_cols"])].astype(np.int32)
    assert (meta["obs_land_idx"] >= 0).all()
    # P2 covariates (T, N_land, C): column 0 = built-up (from the strata); the panel arms use 3 columns
    _bu = (meta["strata_u"] * meta["strata_a"][None, None, :]).sum(-1)
    meta["drift_x"] = np.stack([_bu] + [rng.normal(0, 1, _bu.shape) for _ in range(_DRIFT_C() - 1)], -1)
    from src.config_utils import load_age_model_config
    _sc = load_age_model_config().get("population_model", {}).get("subcell_strata") or {}
    if _sc.get("capacity_moderator"):
        x = np.maximum(-_bu, 0.0)
        meta["subcell_capacity_moderator"] = x / np.sqrt(np.mean(x * x))
    if _sc.get("survival_moderator"):
        x = np.maximum(rng.normal(0, 1, _bu.shape), 0.0)
        meta["subcell_survival_moderator"] = x / np.sqrt(np.mean(x * x))
    meta["habitat_center_occupied"] = occupied_habitat_center(meta)
    # population_model.nest_likelihood (when enabled): 150 attempts at 20 distinct land cells; exposure rows, count rows
    _nn, _nc = 150, 20
    _cells = rng.choice(n, _nc, replace=False)
    meta["nest_rows"], meta["nest_cols"] = rows[_cells], cols[_cells]
    meta["nest_ci"] = rng.integers(0, _nc, _nn); meta["nest_t"] = rng.integers(T // 2, T, _nn)
    meta["nest_region"] = rng.integers(0, 3, _nn); meta["nest_X"] = (rng.random((_nn, 3)) < 0.3).astype(np.float32)
    meta["nest_outcome"] = (rng.random(_nn) < 0.65).astype(np.int64)
    meta["nest_fledged"] = np.where(meta["nest_outcome"] == 1, rng.integers(1, 6, _nn), 0)
    meta["nest_e_pre"] = rng.uniform(0, 25, _nn).astype(np.float32); meta["nest_d_last"] = rng.uniform(1, 6, _nn).astype(np.float32)
    meta["nest_w_exp"] = (rng.random(_nn) < 0.6).astype(np.float32)
    meta["nest_w_cnt"] = (meta["nest_outcome"] == 1).astype(np.float32) * (rng.random(_nn) < 0.7)
    meta["nest_suburban"] = (rng.random(_nn) < 0.3).astype(np.float32)
    return {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in meta.items()}


def _start_like_age_run_map():
    import jax
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    from src.model.age_priors import build_model_2d
    data = synthetic_inputs()
    svi = SVI(build_model_2d, AutoDelta(build_model_2d), numpyro.optim.Adam(1e-3), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(41), data=data, prior_scale=0.1)
    state, loss = svi.update(state, data=data, prior_scale=0.1)
    assert np.isfinite(float(loss))
    # Every pinned site must be absent from the guide, wherever the model samples it.
    from src.model.age_priors import PINNED
    _free = sorted(k for k in PINNED if f"{k}_auto_loc" in svi.get_params(state))
    assert not _free, f"pinned_sites still fitted: {_free}"
    # refine path: age_refine_map splits data into device arrays (TRACED under jit) and static metadata; any numpy
    # call on a data array inside the model breaks only there (step14_dispera, 2026-10-30). Trace + grad it here.
    from src.model import hmc_common as _hc
    import jax.numpy as jnp
    _arrays, _static = _hc.split_data(data)
    _wrapped = _hc.array_model(build_model_2d, _static, prior_scale=0.1)
    _lat = svi.guide.median(svi.get_params(state))
    def _lp(lat, arrays):
        from numpyro.infer.util import log_density
        return log_density(_wrapped, (), {"arrays": arrays}, lat)[0]
    _v, _g = jax.jit(jax.value_and_grad(_lp))(_lat, _arrays)
    assert np.isfinite(float(_v)), _v
    _bad = [k for k, g in _g.items() if not np.all(np.isfinite(np.asarray(g)))]
    assert not _bad, f"non-finite refine-path gradients: {_bad}"
    if os.environ.get("ASSERT_PINNED_ROUTE_SD") == "1":
        params = svi.get_params(state)
        assert "obs_route_sd_auto_loc" not in params, sorted(params)
    if os.environ.get("ASSERT_PINNED_W_SCALE"):
        from numpyro.infer import Predictive
        params = svi.get_params(state)
        assert "w_scale_raw_auto_loc" not in params, sorted(params)
        assert "manifold_idio_auto_loc" not in params, sorted(params)
        raw = np.array([float(x) for x in os.environ["ASSERT_PINNED_W_SCALE"].split(",")])
        lat = {k[:-len("_auto_loc")]: v[None] for k, v in params.items() if k.endswith("_auto_loc")}
        out = Predictive(build_model_2d, posterior_samples=lat, return_sites=["w_scale", "w_env"])(
            jax.random.PRNGKey(0), data=data, prior_scale=0.1)
        assert np.allclose(np.asarray(out["w_scale"][0]), np.logaddexp(0.0, raw), rtol=1e-6)
        w_env = np.asarray(out["w_env"][0])
        assert np.allclose(w_env / w_env[:, :1], np.logaddexp(0.0, raw) / np.logaddexp(0.0, raw[0]), rtol=1e-5)
    print("SMOKE_OK", float(loss))


@pytest.mark.parametrize("overlay", OVERLAYS)
def test_full_model_starts_nan_free(overlay):
    env = dict(os.environ, JAX_DEBUG_NANS="True")
    env.pop("AGE_MODEL_CONFIG", None)
    if overlay:
        env["AGE_MODEL_CONFIG"] = os.path.join(REPO, "config", "overlays", overlay)
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert "SMOKE_OK" in r.stdout, (r.stdout[-2000:] + r.stderr[-4000:])


def test_pinned_route_sd_is_not_fitted():
    """Likelihood-level pins must be excluded from AutoDelta just like prior-level pins."""
    env = dict(os.environ,
               AGE_MODEL_CONFIG=os.path.join(REPO, "config", "overlays",
                                             "map_run64b_subcell_route.json"),
               ASSERT_PINNED_ROUTE_SD="1")
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert "SMOKE_OK" in r.stdout, (r.stdout[-2000:] + r.stderr[-4000:])


def test_pinned_w_scale_raw_vector(tmp_path):
    """A vector pin (w_scale_raw, single habitat field) is conditioned, not fitted, and sets w_scale exactly."""
    import json
    with open(os.path.join(REPO, "config", "overlays", "map_obsroute_effect.json")) as f:
        cfg = json.load(f)
    raw = [0.3, 1.2, -0.4, 0.8]
    pm = cfg.setdefault("population_model", {})
    pm.setdefault("manifold_prior", {}).update({"form": "exchangeable", "fixed_coupling": 1.0})
    pm.setdefault("pinned_sites", {})["w_scale_raw"] = raw
    p = tmp_path / "pinned_w.json"; p.write_text(json.dumps(cfg))
    env = dict(os.environ, AGE_MODEL_CONFIG=str(p), ASSERT_PINNED_W_SCALE=",".join(map(str, raw)))
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert "SMOKE_OK" in r.stdout, (r.stdout[-2000:] + r.stderr[-4000:])


def test_pinned_log_inv_total(tmp_path):
    """The single-release founder total (sampled outside sample_priors) honours pinned_sites."""
    import json
    with open(os.path.join(REPO, "config", "overlays", "map_obsroute_effect.json")) as f:
        cfg = json.load(f)
    pm = cfg.setdefault("population_model", {})
    pm.setdefault("invasion_pulse_prior", {}).update({"form": "single"})
    pm.setdefault("pinned_sites", {})["log_inv_total"] = -1.35
    p = tmp_path / "pinned_inv.json"; p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=REPO,
                       env=dict(os.environ, AGE_MODEL_CONFIG=str(p)), capture_output=True, text=True)
    assert "SMOKE_OK" in r.stdout, (r.stdout[-2000:] + r.stderr[-4000:])


def _run_variant(tmp_path, edit):
    import json
    with open(os.path.join(REPO, "config", "overlays", "map_obsroute_effect.json")) as f:
        cfg = json.load(f)
    edit(cfg.setdefault("population_model", {}))
    p = tmp_path / "variant.json"; p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=REPO,
                       env=dict(os.environ, AGE_MODEL_CONFIG=str(p)), capture_output=True, text=True)
    assert "SMOKE_OK" in r.stdout, (r.stdout[-2000:] + r.stderr[-4000:])


def test_conserving_reset_with_skeleton_penalty(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(process_noise={
        "enabled": True, "sigma": 0.5, "n_freq": 3, "conserve": "regional", "conserve_block": 4,
        "skeleton_tau": 0.3}))


def test_state_anchor_with_penalty(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(state_anchor={"enabled": True, "skeleton_tau": 0.3}))


def test_state_warp_with_penalty(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(state_warp={
        "enabled": True, "sigma_cells": 2.0, "n_freq": 3, "skeleton_tau": 0.3}))


def test_conserving_total(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(process_noise={
        "enabled": True, "sigma": 0.5, "n_freq": 3, "conserve": "total"}))


def test_journey_distance_hazard(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(journey={"distance_hazard": True, "band_mean_km": [80.0, 286.0, 867.0]}))


def test_journey_water_hazard(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(journey={"water_hazard": True}))


def test_journey_water_and_distance_hazard(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(journey={"water_hazard": True, "distance_hazard": True,
                                                         "band_mean_km": [80.0, 286.0, 867.0]}))


def test_settlement_attraction(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(settlement_attraction={"enabled": True}))


def test_settlement_attraction_juveniles_only_with_journey(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(settlement_attraction={"enabled": True, "adults": False},
                                                journey={"water_hazard": True}))


def test_directional_choice(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(directional_choice={"enabled": True}))


def test_directional_choice_with_water_hazard(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(directional_choice={"enabled": True}, journey={"water_hazard": True}))


def test_path_compounding_endpoint(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(path_compounding={"enabled": True}))


def test_path_compounding_mean_with_water(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(path_compounding={"enabled": True, "reference": "mean"},
                                                journey={"water_hazard": True}))


_NEST_ON = {"enabled": True, "path": "unused-in-smoke", "covariates": ["building", "nestbox", "card"]}


def test_nest_likelihood_yearling_pinned(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(nest_likelihood=dict(_NEST_ON),
                                                reproduction={"form": "standard", "yearling_breeding": True}))


def test_nest_likelihood_broods_lognormal_with_distance_hazard(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(
        nest_likelihood=dict(_NEST_ON, broods={"form": "lognormal", "values": [2.5, 2.5, 2.5], "log_sd": 0.3}),
        reproduction={"form": "standard", "yearling_breeding": True},
        journey={"distance_hazard": True, "band_mean_km": [80.0, 286.0, 867.0]}))


def test_nest_likelihood_delayed_breeding_no_covariates(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(nest_likelihood=dict(_NEST_ON, covariates=[])))


def test_journey_decouple_and_band_cost(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(journey={"decouple": True, "band_cost": True}))


def test_low_density_noise(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(process_noise={
        "enabled": True, "sigma": 0.2, "n_freq": 3, "detrend_degree": 1, "target": "low_density"}))


def test_obs_region_effects(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.update(obs_region_effects={"enabled": True, "sigma": 0.2, "n_freq": 3}))


def test_time_varying_band_weights(tmp_path):
    _run_variant(tmp_path, lambda pm: pm.setdefault("dispersal_kernel", {}).update(
        band_weights=True, time_knots=[1940, 1960, 1980, 2000, 2020], time_scale=0.5))


def test_process_noise_fecundity_target(tmp_path):
    """process_noise.target = fecundity (transient growth-rate deviation) builds and steps NaN-free."""
    import json
    with open(os.path.join(REPO, "config", "overlays", "map_obsroute_effect.json")) as f:
        cfg = json.load(f)
    cfg.setdefault("population_model", {})["process_noise"] = {
        "enabled": True, "sigma": 0.1, "n_freq": 3, "detrend_degree": 1, "target": "fecundity"}
    p = tmp_path / "pn_fec.json"; p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=REPO,
                       env=dict(os.environ, AGE_MODEL_CONFIG=str(p)), capture_output=True, text=True)
    assert "SMOKE_OK" in r.stdout, (r.stdout[-2000:] + r.stderr[-4000:])


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _start_like_age_run_map()
