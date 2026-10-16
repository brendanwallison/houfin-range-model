"""Who leaves, and from how crowded a cell? Departure readout at a MAP-format point.

    AGE_MODEL_CONFIG=<overlay> HOUFIN_MODEL_PRECISION=float64 \\
      python research/age_model/exp/departure_readout.py --run-dir <dir> --out <file.json>

Dispersal in year t acts on the end-of-year state of t-1 (Na_grid + Nj_grid) against K_flat[t]
(the invasion pulse added in between is ignored). With x = N/K and the fitted departure curve
p(x) = sigmoid(int + slope*(x - target)) [+ adult offset], reports:
  - the abundance-weighted distribution of x, by region x era -- is N/K > 1 ever common? (it decides
    whether a crowding arm of departure would be identifiable);
  - where departing birds come from, by x band, adults and juveniles separately;
  - the departure probability the fit implies at each band.
Writes <out> (json) and <out>.npz (per land cell, abundance-weighted time-mean x and departures).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)

REGIONS = [("west", -180, -105), ("plains", -105, -95), ("midwest", -95, -85), ("east", -85, 0)]
ERAS = [("1902-1965", 1902, 1965), ("1966-1979", 1966, 1979), ("1980-1993", 1980, 1993),
        ("1994-2007", 1994, 2007), ("2008-2025", 2008, 2025)]
BANDS = [0.0, 0.1, 0.5, 0.9, 1.1, 2.0, np.inf]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("HOUFIN_MODEL_PRECISION", "float64")
    from src.model import hmc_common  # noqa: F401  (sets x64 from the precision)
    import jax
    import rasterio
    from numpyro.infer import Predictive
    from rasterio.warp import transform as crs_transform
    from src.config_utils import load_age_model_config, load_data_config
    from src.model.age_priors import build_model_2d
    from src.model.checkpoints import auto_delta_params_to_latents, load_map_params
    from src.model.data_loading import load_data
    from src.model.runtime_diagnostics import require_gpu

    device = require_gpu("departure readout")
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=os.environ["HOUFIN_MODEL_PRECISION"], verbose=False)
    params, _ = load_map_params(a.run_dir)
    lat = {k: np.asarray(v) for k, v in auto_delta_params_to_latents(params).items()}
    post = {k: jax.numpy.asarray(v)[None] for k, v in lat.items()}
    sites = ["K_flat", "Na_grid", "Nj_grid", "dispersal_logit_intercept", "dispersal_logit_slope"]
    out = Predictive(build_model_2d, posterior_samples=post, return_sites=sites)(
        jax.random.PRNGKey(0), data=data, prior_scale=1.0)
    rows, cols = np.asarray(data["land_rows"]), np.asarray(data["land_cols"])
    K = np.asarray(out["K_flat"][0], np.float64)                      # (T, n_land)
    Na = np.asarray(out["Na_grid"][0], np.float64)[:, rows, cols]     # (T, n_land), end of year
    Nj = np.asarray(out["Nj_grid"][0], np.float64)[:, rows, cols]
    b0 = float(np.asarray(out["dispersal_logit_intercept"][0]))
    b1 = float(np.asarray(out["dispersal_logit_slope"][0]))
    off = float(lat.get("dispersal_logit_adult_offset", 0.0))
    bh = float(lat["dispersal_crowding_beta"]) if "dispersal_crowding_beta" in lat else 0.0
    from src.model.age_priors import CROWDING_KAPPA as kap
    tf = float(data["dispersal_target_fraction"])
    years = np.asarray(data["years"])

    # dispersal in year t (t >= 1) sees the state left at the end of t-1
    Na_d, Nj_d, K_d, yr = Na[:-1], Nj[:-1], K[1:], years[1:]
    N = Na_d + Nj_d
    ok = K_d > 1e-8
    x = np.where(ok, N / np.where(ok, K_d, 1.0), 0.0)
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))
    zf = lambda v: b0 + b1 * (v - tf) + bh * np.logaddexp(0.0, kap * (v - 1.0)) / kap
    pj = sig(zf(x))
    pa = sig(np.clip(zf(x), -10, 10) + off)
    dep_a, dep_j = Na_d * pa, Nj_d * pj

    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as src:
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        lon, _ = crs_transform(src.crs, "EPSG:4326", list(xs), list(ys))
    lon = np.asarray(lon)

    def wq(v, w, qs=(0.1, 0.5, 0.9, 0.99)):
        if w.sum() <= 0:
            return [None] * len(qs)
        o = np.argsort(v); c = np.cumsum(w[o]) / w.sum()
        return [float(v[o][min(np.searchsorted(c, q), v.size - 1)]) for q in qs]

    res = {"run_dir": a.run_dir, "intercept": b0, "slope": b1, "adult_offset": off, "target": tf,
           "crowding_beta": bh,
           "p_at_x": {f"{v:g}": {"juv": float(sig(zf(v))),
                                 "adult": float(sig(np.clip(zf(v), -10, 10) + off))}
                      for v in (0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 4.0)},
           "x_quantiles_abundance_weighted": {}, "share_above_K": {}, "departures_by_band": {}}
    for rn, lo, hi in REGIONS:
        rm = (lon >= lo) & (lon < hi)
        for en, e0, e1 in ERAS:
            tm = (yr >= e0) & (yr <= e1)
            v, w = x[tm][:, rm].ravel(), N[tm][:, rm].ravel()
            key = f"{rn}|{en}"
            res["x_quantiles_abundance_weighted"][key] = wq(v, w)
            res["share_above_K"][key] = float(w[v > 1.1].sum() / max(w.sum(), 1e-300))
    for age, dep in (("adult", dep_a), ("juv", dep_j)):
        tot = dep.sum()
        res["departures_by_band"][age] = {
            f"[{lo:g},{hi:g})": float(dep[(x >= lo) & (x < hi)].sum() / max(tot, 1e-300))
            for lo, hi in zip(BANDS[:-1], BANDS[1:])}
        res["departures_by_band"][age]["fraction_of_pool_departing"] = float(
            tot / max((Na_d if age == "adult" else Nj_d).sum(), 1e-300))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    wN = np.maximum(N.sum(0), 1e-300)
    np.savez_compressed(a.out + ".npz", rows=rows, cols=cols, lon=lon,
                        x_mean=(x * N).sum(0) / wN, dep_adult=dep_a.sum(0), dep_juv=dep_j.sum(0),
                        N_mean=N.mean(0))
    print(json.dumps({k: res[k] for k in ("intercept", "slope", "adult_offset", "crowding_beta", "p_at_x",
                                          "share_above_K",
                                          "departures_by_band")}, indent=1))


if __name__ == "__main__":
    main()
