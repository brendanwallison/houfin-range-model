"""SDM baseline plan, step 0 (D1): export the shared BASELINE DESIGN TABLE from the age model's own data loader,
plus the GRID54 step-3 refined point's per-observation predictive (for the scorer parity gate).

Run from a frozen step-3 code tree with that run's environment (see research/experiments/SDM_BASELINE_B0_2026-10-30.md):
    JAX_PLATFORMS=cpu AGE_MODEL_CONFIG=<code>/config/overlays/grid54_step3.json DATA_CONFIG=<code>/config/overlays/grid54_data.json \
      python sdm_b0_design.py --run-dir <refined run dir> --out <dir>

Writes <out>/design.npz (one row per observation; pseudo-zeros flagged, never scored), <out>/z_cells.npy
(time, land, M) float32 Z the age model reads, <out>/step3_pred.npz (mu, mu before observer effect, concentration,
gate, sigma_u, eta, eps_part) and <out>/design_manifest.json (hashes, config, counts). CPU only.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-z", action="store_true", help="skip z_cells.npy (rescoring needs only the predictive)")
    ap.add_argument("--pred-name", default="step3_pred.npz")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("HOUFIN_MODEL_PRECISION", "float64")
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.config_utils import load_age_model_config, load_data_config
    from src.model.checkpoints import load_map_params
    from src.model.data_loading import _observation_mask_spec, holdout_observation_mask, load_data
    import rasterio
    from rasterio.warp import transform as crs_transform

    cfg = load_age_model_config()
    dev = jax.devices("cpu")[0]
    data = load_data(cfg["input_dir"], target_device=dev, precision="float64")
    print("data keys:", sorted(k for k in data))
    years = np.asarray(data["years"]).astype(int)
    t = np.asarray(data["obs_time_indices"]).astype(int)
    y = np.asarray(data["observed_results"], dtype=np.float64)
    rows, cols = np.asarray(data["obs_rows"]).astype(int), np.asarray(data["obs_cols"]).astype(int)
    idx = np.asarray(data["obs_obsroute_idx"]).astype(int)
    spec = _observation_mask_spec() or {}
    meta = {k: np.asarray(data[k]) for k in ("years", "obs_time_indices", "obs_rows", "obs_cols")}
    temporal = holdout_observation_mask(meta, {"holdout_years": spec.get("holdout_years")})
    held = np.asarray(data["obs_holdout"]).astype(bool)
    spatial = held & ~temporal                      # as map_diagnostics._holdout_metrics (authoritative stored mask)

    # route ids come with the loaded observations; pseudo-zeros have no route (and fall before BBS began)
    route = np.asarray(data["obs_route_idx"]).astype(int)
    pseudo = route < 0
    yrs_obs = years[t]
    align = (f"route ids from load_data; pseudo = route < 0 (n {int(pseudo.sum())}); pseudo years "
             f"{int(yrs_obs[pseudo].min()) if pseudo.any() else None}-{int(yrs_obs[pseudo].max()) if pseudo.any() else None}; "
             f"real rows before 1966: {int(((yrs_obs < 1966) & ~pseudo).sum())}; "
             f"real rows with obsroute < 0: {int(((idx < 0) & ~pseudo).sum())}")
    print("alignment:", align)

    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as src:
        xx, yy = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        lon, lat = map(np.asarray, crs_transform(src.crs, "EPSG:4326", list(xx), list(yy)))
        lrows, lcols = np.asarray(data["land_rows"]), np.asarray(data["land_cols"])
        lx, ly = rasterio.transform.xy(src.transform, lrows, lcols, offset="center")
        llon, llat = map(np.asarray, crs_transform(src.crs, "EPSG:4326", list(lx), list(ly)))
        px, py = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        crs = str(src.crs)

    design = dict(y=y, year=years[t], t=t, row=rows, col=cols, lon=lon, lat=lat,
                  x_m=np.asarray(px, float), y_m=np.asarray(py, float),
                  obsroute=idx, route=route, first_year=np.asarray(data["obs_first_year"]).astype(int),
                  quality=np.asarray(data["obs_quality"]).astype(int) if data.get("obs_quality") is not None else np.zeros(y.shape, int),
                  weight=np.asarray(data.get("obs_weight", np.ones_like(y)), float),
                  pseudo=pseudo, temporal=temporal, spatial=spatial, holdout=held,
                  land_rows=lrows, land_cols=lcols, land_lon=llon, land_lat=llat, years=years)
    np.savez_compressed(os.path.join(a.out, "design.npz"), **design)
    Z = np.asarray(data["Z_gathered"], dtype=np.float32)
    if not a.no_z:
        np.save(os.path.join(a.out, "z_cells.npy"), Z)

    params, _ = load_map_params(a.run_dir)
    # This frozen tree's reconstruct_map does not request expected_obs_before_observer (added later in 30238e0, the
    # code step 3's integrated scores were computed with), so request the needed sites directly.
    import jax.numpy as jnp
    from numpyro.infer import Predictive
    from src.model.age_priors import PINNED, build_model_2d, observer_effect_spec
    from src.model.checkpoints import auto_delta_params_to_latents
    latents = auto_delta_params_to_latents(params)
    post = {k: jnp.expand_dims(v, 0) for k, v in latents.items()}
    sites = ["expected_obs", "expected_obs_before_observer", "obs_concentration", "obs_zero_gate",
             "route_occupancy_log_m", "obs_year_effect", "simulated_density", "observer_route_sigma",
             "observer_first_year_eta"]
    r = Predictive(build_model_2d, posterior_samples=post, return_sites=sites)(
        jax.random.PRNGKey(104), data=data, prior_scale=1.0)
    sim = {k: np.asarray(v[0]) for k, v in r.items()}
    print("returned sites:", sorted(sim))
    lat = {k: np.asarray(v) for k, v in latents.items()}
    for k, v in PINNED.items():
        lat.setdefault(k, np.asarray(v))
    for k in ("observer_route_sigma", "observer_first_year_eta"):
        if k in sim:
            lat.setdefault(k, sim[k])
    sigma = float(np.asarray(lat.get("observer_route_sigma", observer_effect_spec()["sigma_u"])))
    eta = float(np.asarray(lat.get("observer_first_year_eta", 0.0)))
    mu = np.asarray(sim["expected_obs"], np.float64)
    if "expected_obs_before_observer" in sim:
        mu0 = np.asarray(sim["expected_obs_before_observer"], np.float64)
    else:                                   # no observer effect in this model: marginal = plug-in
        mu0, sigma = mu.copy(), 0.0
    log_off = np.log(np.maximum(mu, 1e-300)) - np.log(np.maximum(mu0, 1e-300))
    if "obs_year_effect" in sim:
        log_off = log_off - np.asarray(sim["obs_year_effect"])[t]
    eps_part = log_off - eta * np.where(idx >= 0, design["first_year"], 0)
    pred = dict(mu=mu, mu0=mu0, phi=np.broadcast_to(np.asarray(sim["obs_concentration"], np.float64), mu.shape).copy(),
                sigma_u=sigma, eta=eta, eps_part=eps_part,
                has_gate="obs_zero_gate" in sim)
    if "obs_zero_gate" in sim:
        pred["gate"] = np.asarray(sim["obs_zero_gate"], np.float64)
        pred["route_occupancy_m"] = float(np.exp(np.asarray(sim["route_occupancy_log_m"])))
    if "simulated_density" in sim:
        dens = np.asarray(sim["simulated_density"]).reshape(len(years), -1)
        if dens.shape[1] != len(lrows):
            W = np.asarray(data["land_mask"]).shape[1]
            dens = dens[:, lrows * W + lcols]
        np.save(os.path.join(a.out, a.pred_name.replace("_pred.npz", "_density.npy")), dens.astype(np.float32))
    np.savez_compressed(os.path.join(a.out, a.pred_name), **pred)

    h = hashlib.sha256()
    for k in ("y", "year", "row", "col", "obsroute", "temporal", "spatial"):
        h.update(np.ascontiguousarray(design[k]).tobytes())
    man = {"run_dir": a.run_dir, "input_dir": cfg["input_dir"], "bbs_npz": cfg["bbs_npz"], "crs": crs,
           "observation_mask": spec, "alignment": align, "design_sha256": h.hexdigest(),
           "n_obs": int(y.size), "n_pseudo": int(pseudo.sum()), "n_temporal": int(temporal.sum()),
           "n_spatial": int(spatial.sum()), "n_spatial_real": int((spatial & ~pseudo).sum()),
           "n_temporal_pseudo": int((temporal & pseudo).sum()), "z_shape": list(Z.shape),
           "n_land": int(len(lrows)), "sigma_u": sigma, "eta": eta, "has_gate": pred["has_gate"]}
    json.dump(man, open(os.path.join(a.out, "design_manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
