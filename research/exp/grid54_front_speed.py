"""Front-speed diagnostic: re-simulate a refined fit with habitat (Z, Z_disp) frozen at its 1966-2025 mean and compare
each eastern cell's first-occupied year with the first BBS-positive year. Diagnostic only (static habitat is not a model).
Usage: front_speed.py RUN_DIR OUT_JSON"""
import json, sys
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "scripts/viz")
from map_diagnostics import reconstruct_map
from src.config_utils import load_age_model_config
from src.model.checkpoints import load_map_params
from src.model.data_loading import load_data

cfg = load_age_model_config()
data = load_data(cfg["input_dir"], target_device=jax.devices()[0], precision="float64")
params, _ = load_map_params(sys.argv[1])
years = np.asarray(data["years"]); rows, cols = np.asarray(data["land_rows"]), np.asarray(data["land_cols"])
print("data keys with scal/pop:", [k for k in data if "scal" in k.lower()][:10])
win = (years >= 1966) & (years <= 2025)

def freeze(d):
    d = dict(d)
    for k in ("Z_gathered", "Z_disp_gathered"):
        z = jnp.asarray(d[k]); m = z[jnp.asarray(win)].mean(0, keepdims=True)
        d[k] = jnp.broadcast_to(m, z.shape).astype(z.dtype)
    return d

# observed first positive BBS year per land cell (real observations only: pseudo-zeros have no route)
lookup = -np.ones((int(rows.max()) + 1, int(cols.max()) + 1), int); lookup[rows, cols] = np.arange(len(rows))
orow, ocol = np.asarray(data["obs_rows"]), np.asarray(data["obs_cols"])
ot = np.asarray(data["obs_time_indices"]); y = np.asarray(data["observed_results"])
real = np.asarray(data["obs_obsroute_idx"]) >= 0 if "obs_obsroute_idx" in data else np.ones(len(y), bool)
cell = lookup[orow, ocol]
first_obs = np.full(len(rows), np.nan); surveyed_from = np.full(len(rows), np.nan)
for i in np.flatnonzero(real & (cell >= 0)):
    c, yr = cell[i], years[ot[i]]
    surveyed_from[c] = yr if np.isnan(surveyed_from[c]) else min(surveyed_from[c], yr)
    if y[i] > 0:
        first_obs[c] = yr if np.isnan(first_obs[c]) else min(first_obs[c], yr)

import rasterio
from rasterio.warp import transform as crs_transform
from src.config_utils import load_data_config
with rasterio.open(load_data_config()["grid"]["ref_raster"]) as src:
    xx, yy = rasterio.transform.xy(src.transform, rows, cols, offset="center")
    lon, lat = map(np.asarray, crs_transform(src.crs, "EPSG:4326", list(xx), list(yy)))
east = (lon >= -95) & (lat >= 25) & (lat < 49)
# cells that were surveyed before birds arrived (left-censoring removed): first positive > first survey year + 2
ok = east & np.isfinite(first_obs) & (first_obs >= surveyed_from + 3)

out = {}
thresholds = {}
def no_native(d):
    d = dict(d); d["initpop_latent"] = jnp.zeros_like(jnp.asarray(d["initpop_latent"])); return d

arms = (("fitted", data), ("frozen_1966_2025_mean", freeze(data)),
        ("fitted_release_only", no_native(data)), ("frozen_release_only", no_native(freeze(data))))
if cfg['population_model'].get('process_noise',{}).get('enabled'):
    arms += tuple((name+'_zero_innovations',d) for name,d in arms)
for name, d in arms:
    if name.endswith('_zero_innovations'):
        import inspect
        import src.model.age_priors as ap
        original=ap.forward_sim_age_structured
        def without_innovations(*aa,**kw):
            bound=inspect.signature(original).bind(*aa,**kw);bound.apply_defaults()
            bound.arguments['process_noise']=None
            return original(**bound.arguments)
        ap.forward_sim_age_structured=without_innovations
        try:sim=reconstruct_map(d,params)
        finally:ap.forward_sim_age_structured=original
    else:
        sim = reconstruct_map(d, params)
    eo = np.asarray(sim["expected_obs"]) if False else None
    dens = np.asarray(sim["simulated_density"])          # (time, land) model density
    dens = dens.reshape(len(years), -1)
    if dens.shape[1] != len(rows):                       # full grid -> land cells
        W = np.asarray(data["land_mask"]).shape[1]
        dens = dens[:, rows * W + cols]
    res = {}
    for thr_name, thr in (("1pct_of_median_occupied", 0.01), ("10pct", 0.10)):
        reference=name.removesuffix('_zero_innovations')
        if name==reference:thresholds[(reference,thr_name)]=float(thr*np.nanmedian(dens[(years>=2010)][:,ok]))
        threshold=thresholds[(reference,thr_name)]
        occ = dens >= threshold
        first_mod = np.where(occ.any(0), years[np.argmax(occ, 0)], np.nan)
        lag = first_mod[ok] - first_obs[ok]
        res[thr_name] = {"threshold_internal":threshold,"threshold_reference_arm":reference,"n_cells": int(ok.sum()), "median_model_minus_observed_yr": float(np.nanmedian(lag)),
                         "p25": float(np.nanpercentile(lag, 25)), "p75": float(np.nanpercentile(lag, 75)),
                         "share_model_earlier_by_5y_plus": float(np.mean(lag <= -5)),
                         "never_occupied": float(np.mean(np.isnan(first_mod[ok])))}
        bins = [1966, 1975, 1985, 1995, 2010]
        res[thr_name]["by_observed_arrival"] = {f"{a}-{b-1}": float(np.nanmedian(lag[(first_obs[ok] >= a) & (first_obs[ok] < b)]))
                                               for a, b in zip(bins[:-1], bins[1:])}
    tot = dens[:, east].sum(1)
    res["east_total_density_by_year"] = {int(yy_): float(tot[i]) for i, yy_ in enumerate(years) if yy_ in (1950, 1960, 1966, 1975, 1985, 1995, 2010, 2025)}
    out[name] = res
    print(name, json.dumps(res, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
