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
    args = parser.parse_args()
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
    tr = handlers.trace(handlers.seed(handlers.substitute(ap.build_model_2d, data=lat), args.seed)).get_trace(
        data=sb.data, prior_scale=1.)
    excluded = ["dispersal_logit_intercept", "dispersal_logit_slope"] if ap.VEIT_DEPARTURE else []
    coords = MapCoordinates(tr, exclude=excluded)
    del tr
    arrays, static = hc.split_data(sb.data)
    model = hc.hide_deterministics(hc.array_model(ap.build_model_2d, static))

    def score(vector, data_arrays):
        lp, trace = log_density(model, (), {"arrays": data_arrays}, coords.constrain(vector))
        parts = []
        for name in ("obs", "nest_loglik"):
            site = trace[name]
            value = site["fn"].log_prob(site["value"])
            parts.append(jnp.sum(value * (site["scale"] if site.get("scale") is not None else 1.)))
        data_nll = -jnp.stack(parts)
        return -lp, data_nll

    value_grad = jax.jit(jax.value_and_grad(score, has_aux=True))
    x0 = np.asarray(coords.x0, dtype=np.float64)
    if args.jitter:
        x0 = x0 + np.random.default_rng(args.seed).normal(0., args.jitter, x0.shape)
    save_params(args.out / "initial_params.npz", coords.constrain(x0))
    sources = {name: {"path": mod.__file__, "sha256": digest(mod.__file__)}
               for name, mod in list(sys.modules.items())
               if name.startswith("src.") and getattr(mod, "__file__", "").endswith(".py")}
    for path in (Path(__file__), Path(__file__).with_name("local_map_coordinates.py")):
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
                "objective_measure": "negative constrained-site log posterior; no coordinate Jacobian",
                "data_terms": ["obs", "nest_loglik"], "seed": args.seed, "jitter": args.jitter,
                "initial_checkpoint": str(args.init or os.environ["SBX_REFINED"]),
                "initial_override_sha256": digest(args.init) if args.init else None,
                "initial_params_sha256": digest(args.out / "initial_params.npz"),
                "command": sys.argv, "devices": [str(d) for d in jax.devices()],
                "versions": {"python": sys.version, "jax": jax.__version__, "numpy": np.__version__,
                             "numpyro": numpyro.__version__, "scipy": scipy.__version__},
                "n_coordinates": int(x0.size), "started_unix": started}
    write_json(args.out / "manifest.json", manifest)
    trace_file = (args.out / "trace.jsonl").open("x")
    best = {"objective": float("inf")}
    state = {"evaluations": 0, "iterations": 0}

    def evaluate(x):
        (value, data), gradient = value_grad(jnp.asarray(x), arrays)
        value, data, gradient = float(value), np.asarray(data), np.asarray(gradient, np.float64)
        if not np.isfinite(value) or not np.isfinite(gradient).all() or not np.isfinite(data).all():
            raise FloatingPointError("Nonfinite objective/gradient; last finite checkpoint preserved")
        state["evaluations"] += 1
        row = {**state, "objective": value, "data_nll": dict(zip(manifest["data_terms"], data.tolist())),
               "prior_and_regularizer_cost": value - float(data.sum()),
               "gradient_l2": float(np.linalg.norm(gradient)), "gradient_max": float(np.abs(gradient).max()),
               "seconds": time.time() - started}
        trace_file.write(json.dumps(row) + "\n")
        trace_file.flush()
        if value < best["objective"]:
            best.update(row, x=np.array(x), gradient=np.array(gradient))
            if state["evaluations"] == 1 or state["evaluations"] % 10 == 0:
                save_params(args.out / "best_params.npz", coords.constrain(x))
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
            result = minimize(evaluate, x0, jac=True, method="L-BFGS-B", callback=callback,
                              options={"maxiter": args.maxiter, "maxls": 40, "maxcor": 20,
                                       "ftol": 1e-12, "gtol": 1e-4})
            evaluate(result.x)
    except FloatingPointError as error:
        failure = str(error)
    finally:
        trace_file.close()
        if "x" in best:
            # Save the evaluated point before any optional biological diagnostics.
            save_params(args.out / "best_params.npz", coords.constrain(best["x"]))
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
