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
            "map_run22_relative_allee.json"]


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
        age_structure_prior={"alpha": 1.01, "beta": 1.01, "effective_sites": 100.0})
    meta["n_obs_quality_tiers"] = int(np.unique(meta["obs_quality"]).size)
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


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    _start_like_age_run_map()
