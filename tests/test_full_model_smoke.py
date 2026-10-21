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
            "map_run67r_st_correction.json", "map_run66m_readout.json", "map_run69a_b1_nofloor.json", "map_run69b_b1_smoothfloor.json", "map_run68b_plausible_truth_nofloor.json", "map_run69c_b3_nofloor.json", "map_run70z_holdout_base.json", "map_run70b_process_noise.json", "map_run70d_process_noise.json", "map_run71b_allee_sensitivity.json", "map_run71c_allee_sensitivity.json", "map_run74a_settlement.json", "map_run74b_settlement.json", "map_run74c_settlement.json", "map_run74e_settlement.json", "map_run75a_retest.json", "map_run75b_retest.json", "map_run77a_native_departure.json", "map_run72a_route_occupancy.json", "map_run73sh_lambda_t.json", "map_run73h_lambda_t.json",
            "map_run67a2_st_correction.json", "map_run67c2_st_correction.json", "map_run67n2_st_correction.json", "map_run67r2_st_correction.json"]


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
    ss = build_simulation_struct(jnp.asarray(land), 27.0, 100.0, 330.0, 0.468, 0.468,
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
    if os.environ.get("ASSERT_PINNED_ROUTE_SD") == "1":
        params = svi.get_params(state)
        assert "obs_route_sd_auto_loc" not in params, sorted(params)
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


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _start_like_age_run_map()
