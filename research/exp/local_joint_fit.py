"""Joint constrained-site MAP comparison with explicit coordinates and provenance.

No forward patches or output-change penalties. Every unpinned continuous site
is optimized except the verified inactive Veit logistic departure coordinates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.infer.util import log_density
from scipy.optimize import minimize
import scipy

from harness import Sandbox, ap
from src.config_utils import load_age_model_config
from src.model import hmc_common as hc
from local_map_coordinates import MapCoordinates
from local_profile_metrics import likelihood_gauge, late_growth_metric, constraint_cost


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_params(path, params):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **{k: np.asarray(v) for k, v in params.items()})
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--jitter", type=float, default=0.)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--maxcor", type=int, default=20, help="L-BFGS curvature-history size; model/objective unchanged")
    parser.add_argument("--objective", choices=["posterior","conditional_data"], default="posterior")
    parser.add_argument("--common-nuisance", type=Path)
    parser.add_argument("--profile-target",type=float)
    parser.add_argument("--profile-rho",type=float,default=10000.)
    parser.add_argument("--profile-multiplier",type=float,default=0.)
    parser.add_argument("--method",choices=["lbfgs","full_bfgs"],default="lbfgs")
    args = parser.parse_args()
    if args.profile_rho <= 0 or (args.profile_target is not None and args.profile_target < 0):
        raise ValueError("Profile penalty scale must be positive and RMS target nonnegative")
    if not ap.YEARLING or ap.VEIT_REPRO:
        raise ValueError("Registered growth metric assumes standard yearling reproduction")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write_json(args.out / "status.json", {"state": "loading", "pid": os.getpid(), "started_unix": started})
    cfg = load_age_model_config()
    if cfg["population_model"].get("cbc_likelihood", {}).get("enabled", False):
        raise ValueError("This comparison's data accounting is registered for BBS and NestWatch")
    sb = Sandbox(os.environ["SBX_REFINED"])
    lat = {k: jnp.asarray(v) for k, v in sb.lat.items()}
    if args.init:
        with np.load(args.init) as saved:
            lat.update({k: jnp.asarray(saved[k]) for k in saved.files})
    spec = cfg["population_model"]["manifold_prior"]
    if spec["form"] == "capacity_demography" and "capacity_factor" not in lat:
        r = float(spec["fixed_capacity_correlation"])
        if r < 1.:
            lat["capacity_factor"] = lat["manifold_factor"] * ((1-r) / np.sqrt(1-r*r))
    nuisance, gauge_fixed, pre_gauge_data = [], [], None
    if args.objective == "conditional_data":
        if args.common_nuisance is None:
            raise ValueError("Conditional data profiles require the same fixed observer nuisance checkpoint")
        nuisance = ["observer_route_eps","observer_first_year_eta"]
        with np.load(args.common_nuisance) as common:
            for name in nuisance:
                lat[name] = jnp.asarray(common[name])
        if spec["form"] == "exchangeable" and spec.get("fixed_coupling") != 1.:
            raise ValueError("Conditional gauge requires the registered single-field F0")
        if spec["form"] == "capacity_demography" and spec.get("fixed_capacity_correlation") != 0.:
            raise ValueError("Conditional gauge requires the registered independent F2")
        _, original_trace = log_density(ap.build_model_2d,(),{"data":sb.data,"prior_scale":1.},lat)
        pre_gauge_data = [float(jnp.sum(original_trace[name]["fn"].log_prob(original_trace[name]["value"]) *
                                (original_trace[name]["scale"] if original_trace[name].get("scale") is not None else 1.)))
                          for name in ("obs","nest_loglik")]
        del original_trace
        lat, gauge_fixed = likelihood_gauge(lat,spec["form"])
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), args.seed)).get_trace(
        data=sb.data, prior_scale=1.)
    excluded = ["dispersal_logit_intercept", "dispersal_logit_slope"] if ap.VEIT_DEPARTURE else []
    coords = MapCoordinates(tr, exclude=excluded+nuisance)
    del tr
    arrays, static = hc.split_data(sb.data)
    array_model = hc.array_model(ap.build_model_2d, static)
    def model(**kwargs):
        with handlers.block(hide_fn=lambda site: site["type"] == "deterministic" and site["name"] not in {"Sa_flat","Sj_flat","Fmax_flat"}):
            return array_model(**kwargs)
    cells = jnp.asarray(np.flatnonzero(sb.region["east"] & np.isfinite(sb.first_obs)))
    i0, i1 = [int(np.flatnonzero(sb.years == year)[0]) for year in (1993,2010)]
    fixed_positions, offset = [], 0
    for name in coords.names:
        if name == "w_scale_raw":
            fixed_positions = [offset+i for i in gauge_fixed]
        offset += int(coords.initial[name].size)
    active = jnp.asarray([i for i in range(int(coords.x0.size)) if i not in fixed_positions])
    def constrain(vector):
        return coords.constrain(coords.x0.at[active].set(vector))

    def score(vector, data_arrays):
        lp, trace = log_density(model, (), {"arrays": data_arrays}, constrain(vector))
        parts = []
        for name in ("obs", "nest_loglik"):
            site = trace[name]
            value = site["fn"].log_prob(site["value"])
            parts.append(jnp.sum(value * (site["scale"] if site.get("scale") is not None else 1.)))
        data_nll = -jnp.stack(parts)
        metric, delta = late_growth_metric(*[trace[k]["value"] for k in ("Sa_flat","Sj_flat","Fmax_flat")],i0,i1,cells)
        penalty = constraint_cost(metric,args.profile_target,args.profile_rho,args.profile_multiplier)
        base = -lp if args.objective == "posterior" else data_nll.sum()
        return base+penalty, (data_nll,-lp,metric,penalty,delta)

    value_grad = jax.jit(jax.value_and_grad(score, has_aux=True))
    x0 = np.asarray(coords.x0[active], dtype=np.float64)
    if args.method == "full_bfgs" and x0.size > 1000:
        raise ValueError("Dense BFGS is reserved for the small conditional profile coordinate set")
    if args.jitter:
        x0 = x0 + np.random.default_rng(args.seed).normal(0., args.jitter, x0.shape)
    save_params(args.out / "initial_params.npz", constrain(x0))
    sources = {name: {"path": mod.__file__, "sha256": digest(mod.__file__)}
               for name, mod in list(sys.modules.items())
               if name.startswith("src.") and getattr(mod, "__file__", "").endswith(".py")}
    for path in (Path(__file__), Path(__file__).with_name("local_map_coordinates.py"),Path(__file__).with_name("local_profile_metrics.py")):
        sources[path.name] = {"path": str(path), "sha256": digest(path)}
    root = Path(cfg["input_dir"])
    with (root / "metadata.pkl").open("rb") as f:
        meta = pickle.load(f)
    inputs = []
    for rel in ("metadata.pkl", meta["z_gathered_path"], meta["z_disp_gathered_path"]):
        path = root / rel
        inputs.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)})
    nest = cfg["population_model"].get("nest_likelihood", {})
    if nest.get("enabled"):
        path = Path(nest["path"])
        inputs.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {"code_commit": args.code_commit, "sources": sources, "inputs": inputs,
                "config": cfg, "coordinates": coords.describe(), "fixed_inactive": excluded,
                "objective_measure": ("negative constrained-site log posterior; no coordinate Jacobian" if args.objective == "posterior" else "conditional data NLL, common observer effects fixed; coefficient gauge removes exact likelihood-scale redundancy"),
                "data_terms": ["obs", "nest_loglik"], "seed": args.seed, "jitter": args.jitter,
                "initial_checkpoint": str(args.init or os.environ["SBX_REFINED"]),
                "initial_override_sha256": digest(args.init) if args.init else None,
                "initial_params_sha256": digest(args.out / "initial_params.npz"),
                "command": sys.argv, "devices": [str(d) for d in jax.devices()],
                "versions": {"python": sys.version, "jax": jax.__version__, "numpy": np.__version__,
                             "numpyro": numpyro.__version__, "scipy": scipy.__version__},
                "n_coordinates": int(x0.size), "started_unix": started}
    manifest["optimizer"] = {"method":args.method, "maxcor":args.maxcor,"ftol":1e-12,"gtol":1e-4}
    manifest["profile"] = {"target":args.profile_target,"rho":args.profile_rho,"multiplier":args.profile_multiplier,
                            "metric":"RMS fixed occupied-East per-cell log potential lambda change 1993-2010",
                            "common_nuisance":str(args.common_nuisance) if args.common_nuisance else None,
                            "fixed_nuisance_sites":nuisance,"gauge_fixed_amplitude_indices":gauge_fixed}
    manifest["pre_gauge_data_log_density"] = pre_gauge_data
    write_json(args.out / "manifest.json", manifest)
    trace_file = (args.out / "trace.jsonl").open("x")
    best = {"objective": float("inf")}
    state = {"evaluations": 0, "iterations": 0}

    def evaluate(x):
        (value, auxiliary), gradient = value_grad(jnp.asarray(x), arrays)
        data, posterior, metric, penalty, delta = jax.device_get(auxiliary)
        value, data, gradient = float(value), np.asarray(data), np.asarray(gradient, np.float64)
        if (not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(data).all()
                or not np.isfinite(np.asarray([posterior,metric,penalty])).all() or not np.isfinite(delta).all()):
            raise FloatingPointError("Nonfinite objective/gradient; last finite checkpoint preserved")
        state["evaluations"] += 1
        if state["evaluations"] == 1 and pre_gauge_data is not None and not args.jitter:
            if not np.allclose(-data,pre_gauge_data,rtol=0.,atol=1e-6):
                raise ValueError("Actual-data likelihood changed under supposedly equivalent coefficient gauge")
        row = {**state, "objective": value, "data_nll": dict(zip(manifest["data_terms"], data.tolist())),
               "raw_posterior_cost":float(posterior),"constraint_cost":float(penalty),
               "profile_metric":float(metric),"profile_residual":float(metric)-args.profile_target if args.profile_target is not None else None,
               "profile_delta_q10_50_90":np.quantile(delta,[.1,.5,.9]).tolist(),
               "prior_and_regularizer_cost": float(posterior) - float(data.sum()),
               "gradient_l2": float(np.linalg.norm(gradient)), "gradient_max": float(np.abs(gradient).max()),
               "seconds": time.time() - started}
        trace_file.write(json.dumps(row) + "\n")
        trace_file.flush()
        if value < best["objective"]:
            best.update(row, x=np.array(x), gradient=np.array(gradient))
            if state["evaluations"] == 1 or state["evaluations"] % 10 == 0:
                save_params(args.out / "best_params.npz", constrain(x))
                write_json(args.out / "status.json", {"state": "running", "pid": os.getpid(), **row})
        if state["evaluations"] == 1 or state["evaluations"] % 25 == 0:
            print(json.dumps(row), flush=True)
        return value, gradient

    def callback(x):
        state["iterations"] += 1

    failure = None
    result = None
    try:
        initial, _ = evaluate(x0)
        write_json(args.out / "initial.json", {k: v for k, v in best.items() if k not in ("x", "gradient")})
        if not args.evaluate_only:
            options = ({"maxiter":args.maxiter,"gtol":1e-4,"xrtol":1e-12} if args.method == "full_bfgs" else
                       {"maxiter":args.maxiter,"maxls":40,"maxcor":args.maxcor,"ftol":1e-12,"gtol":1e-4})
            result = minimize(evaluate,x0,jac=True,method="BFGS" if args.method == "full_bfgs" else "L-BFGS-B",callback=callback,options=options)
            evaluate(result.x)
    except FloatingPointError as error:
        failure = str(error)
    finally:
        trace_file.close()
        if "x" in best:
            # Save the evaluated point before any optional biological diagnostics.
            save_params(args.out / "best_params.npz", constrain(best["x"]))
    if "x" not in best:
        write_json(args.out / "status.json", {"state": "failed", "reason": failure})
        raise RuntimeError(failure or "No finite point")
    terminal = {k: v for k, v in best.items() if k not in ("x", "gradient")}
    terminal.update(state=("failed_finite_checkpoint_saved" if failure else "evaluated" if args.evaluate_only else "finished"),
                    reason=failure or (str(result.message) if result is not None else "evaluation only"),
                    optimizer_success=bool(result.success) if result is not None else False,
                    stationarity_demonstrated=bool(best["gradient_max"] < 1e-4),
                    final_counts=state, seconds_total=time.time() - started)
    write_json(args.out / "result.json", terminal)
    write_json(args.out / "status.json", terminal)
    print(json.dumps(terminal), flush=True)
    if failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
