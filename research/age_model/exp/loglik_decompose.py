"""Where does a fit's likelihood live? Per-observation NB log-likelihood at a MAP-format point,
aggregated by region x era x count type -- and, with --compare, the difference between two points.

    AGE_MODEL_CONFIG=<overlay> HOUFIN_MODEL_PRECISION=float64 \\
      python research/age_model/exp/loglik_decompose.py --run-dir <A> [--compare <B> [--compare-config <overlay B>]] --out <dir>

Each point is evaluated under ITS OWN config (the model reads config at import), so --compare runs
a subprocess for B. Writes <out>/loglik.json (group sums) and <out>/per_obs.npz (per-observation
ll, plus lon/lat/year/count/quality) so later analyses need no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)

REGIONS = [("west", -180, -105), ("plains", -105, -95), ("midwest", -95, -85), ("east", -85, 0)]
ERAS = [("1966-1979", 1966, 1979), ("1980-1993", 1980, 1993), ("1994-2007", 1994, 2007), ("2008-2025", 2008, 2025)]


def per_observation(run_dir: str) -> dict:
    os.environ.setdefault("HOUFIN_MODEL_PRECISION", "float64")
    from src.model import hmc_common  # noqa: F401  (sets x64 from the precision)
    import jax
    import numpyro.distributions as dist
    import rasterio
    from numpyro.infer import Predictive
    from rasterio.warp import transform as crs_transform
    from src.config_utils import load_age_model_config, load_data_config
    from src.model.age_priors import PINNED, build_model_2d
    from src.model.checkpoints import auto_delta_params_to_latents, load_map_params
    from src.model.data_loading import load_data
    from src.model.runtime_diagnostics import require_gpu

    device = require_gpu("loglik decomposition")
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=os.environ["HOUFIN_MODEL_PRECISION"], verbose=False)
    params, _ = load_map_params(run_dir)
    lat = {k: np.asarray(v) for k, v in auto_delta_params_to_latents(params).items()}
    post = {k: jax.numpy.asarray(v)[None] for k, v in lat.items()}
    out = Predictive(build_model_2d, posterior_samples=post,
                     return_sites=["expected_obs", "concentration", "quality_conc_mult", "obs_conc_kappa"])(
        jax.random.PRNGKey(0), data=data, prior_scale=1.0)
    mu = np.asarray(out["expected_obs"][0], dtype=np.float64)
    conc = float(np.asarray(out["concentration"][0]))
    q = float(np.asarray(out["quality_conc_mult"][0])) if "quality_conc_mult" in out else 1.0
    y = np.asarray(data["observed_results"], dtype=np.float64)
    quality = np.asarray(data["obs_quality"]) if data.get("obs_quality") is not None else np.zeros_like(y)
    conc_obs = conc * q ** quality
    if "obs_conc_kappa" in out:     # observation.conc_density_power: phi(mu) = phi0 * mu^kappa
        conc_obs = conc_obs * np.maximum(mu, 1e-3) ** float(np.asarray(out["obs_conc_kappa"][0]))
    ll = np.asarray(dist.NegativeBinomial2(mean=mu, concentration=conc_obs).log_prob(y), dtype=np.float64)
    rows, cols = np.asarray(data["obs_rows"]), np.asarray(data["obs_cols"])
    years = np.asarray(data["years"])[np.asarray(data["obs_time_indices"])]
    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as src:
        x, yy = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        lon, latd = crs_transform(src.crs, "EPSG:4326", list(x), list(yy))
    return {"ll": ll, "mu": mu, "y": y, "quality": quality, "year": years,
            "lon": np.asarray(lon), "lat": np.asarray(latd), "rows": rows, "cols": cols,
            "pinned": {k: v for k, v in PINNED.items()}}


def groups(d: dict) -> dict:
    res = {"total": float(d["ll"].sum()), "n": int(d["ll"].size), "by": {}}
    pos = d["y"] > 0
    for rname, lo, hi in REGIONS:
        rmask = (d["lon"] >= lo) & (d["lon"] < hi) & (d["quality"] == 0)
        for ename, e0, e1 in ERAS:
            m = rmask & (d["year"] >= e0) & (d["year"] <= e1)
            for cname, cm in (("zero", ~pos), ("pos", pos)):
                mm = m & cm
                res["by"][f"{rname}|{ename}|{cname}"] = {"ll": float(d["ll"][mm].sum()), "n": int(mm.sum())}
    mx = d["quality"] > 0
    res["by"]["mexico|all|all"] = {"ll": float(d["ll"][mx].sum()), "n": int(mx.sum())}
    return res


def _evaluate(run_dir, config, out):
    """Evaluate one point in its OWN process: the model reads its config at import, and JAX
    pre-allocates ~75% of GPU memory per process -- so a parent that evaluated A itself left too
    little for B's child (TACC E001 f8). Pre-allocation is off here; the parent does no GPU work."""
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    if config:
        env["AGE_MODEL_CONFIG"] = config
    subprocess.run([sys.executable, __file__, "--single", "--run-dir", run_dir, "--out", out],
                   env=env, check=True)
    return json.load(open(os.path.join(out, "loglik.json")))["A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--compare-config", default=None, help="AGE_MODEL_CONFIG for --compare (default: same)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.single:
        d = per_observation(args.run_dir)
        np.savez(os.path.join(args.out, "per_obs_A.npz"), **{k: v for k, v in d.items() if k != "pinned"})
        json.dump({"A": {"run_dir": args.run_dir, "config": os.environ.get("AGE_MODEL_CONFIG"), **groups(d)}},
                  open(os.path.join(args.out, "loglik.json"), "w"), indent=1)
        return
    a = _evaluate(args.run_dir, None, args.out)
    report = {"A": a}
    if args.compare:
        b = _evaluate(args.compare, args.compare_config, os.path.join(args.out, "B"))
        report["B"] = b
        delta = {k: {"dll": b["by"][k]["ll"] - a["by"][k]["ll"], "n": b["by"][k]["n"]} for k in a["by"]}
        report["delta_B_minus_A"] = {"total": b["total"] - a["total"],
                                     "by": dict(sorted(delta.items(), key=lambda kv: kv[1]["dll"]))}
    json.dump(report, open(os.path.join(args.out, "loglik.json"), "w"), indent=1)
    print(json.dumps({"A_total": report["A"]["total"],
                      "delta_total": report.get("delta_B_minus_A", {}).get("total")}))
    if "delta_B_minus_A" in report:
        items = list(report["delta_B_minus_A"]["by"].items())
        print("most negative (B fits worse):", [(k, round(v["dll"], 1), v["n"]) for k, v in items[:6]])
        print("most positive (B fits better):", [(k, round(v["dll"], 1), v["n"]) for k, v in items[-6:]])


if __name__ == "__main__":
    main()
