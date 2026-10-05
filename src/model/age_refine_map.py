"""Refine a MAP to an actual optimum with L-BFGS, before probing or sampling it.

Why this stage exists: NUTS started at run_18's quick90 MAP spent its entire run
descending -- the best draw sat 7,071 nats below the MAP's potential (+7,424 in fit,
-353 in prior), and both chains were still falling when they stopped. Adam's 400-step
budget, with the learning rate decayed to 1e-3, ends far from the mode; everything
local computed there (Laplace metric, negative-curvature counts, step sizes) describes
an arbitrary hillside. The intended workflow is MAP -> refine -> probe -> HMC.

L-BFGS (optax, zoom linesearch) minimizes the UNCONSTRAINED potential NUTS samples, in
float64 (float32 U has a ~1 nat noise floor). It starts from the MAP, checkpoints every
``ckpt_every`` iterations and resumes, and stops when U falls by less than ``tol_du``
over ``tol_window`` iterations or |grad| < ``tol_grad``. The result is written as a
MAP-format checkpoint (``map_checkpoint.pkl``) plus ``refine_report.json``; point the
probe and HMC at it with ``HOUFIN_HMC_MAP_DIR=<that directory>``.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import hmc_common as hc                            # noqa: E402  (sets x64)
import jax                                                         # noqa: E402
import jax.numpy as jnp                                            # noqa: E402
import numpy as np                                                 # noqa: E402
import optax                                                       # noqa: E402

from src.model.checkpoints import save_pickle_atomic               # noqa: E402

STATE_FILE = "refine_state.pkl"


def refine_settings(pcfg: dict) -> dict:
    r = pcfg["refine"]
    return {
        "max_iter": hc.setting(r, "max_iter", "HOUFIN_REFINE_MAX_ITER", int),
        "memory": hc.setting(r, "memory", "HOUFIN_REFINE_MEMORY", int),
        "tol_du": hc.setting(r, "tol_du", "HOUFIN_REFINE_TOL_DU", float),
        "tol_window": hc.setting(r, "tol_window", "HOUFIN_REFINE_TOL_WINDOW", int),
        "tol_grad": hc.setting(r, "tol_grad", "HOUFIN_REFINE_TOL_GRAD", float),
        "ckpt_every": hc.setting(r, "ckpt_every", "HOUFIN_REFINE_CKPT_EVERY", int),
        "log_every": hc.setting(r, "log_every", "HOUFIN_REFINE_LOG_EVERY", int),
    }


def converged(history: list, s: dict):
    """Reason string if the run should stop, else None. history: [(iter, U, |grad|)]."""
    if not history:
        return None
    if history[-1][2] < s["tol_grad"]:
        return f"|grad| {history[-1][2]:.3g} < tol_grad {s['tol_grad']}"
    w = s["tol_window"]
    if len(history) > w and history[-w - 1][1] - history[-1][1] < s["tol_du"]:
        return (f"U fell {history[-w - 1][1] - history[-1][1]:.3g} < tol_du {s['tol_du']} "
                f"over {w} iterations")
    return None


def lbfgs_minimize(f, x0, kw, s: dict, state_path: str, run_fingerprint: str, log=print):
    """Minimize ``f(x, kw)`` from ``x0``; resumable through ``state_path``.

    ``kw`` is passed to the jitted step as an argument (never closed over: see
    hc.split_data). Returns (x, history, reason). A non-finite value stops the run at
    the last finite point.
    """
    opt = optax.lbfgs(memory_size=s["memory"])

    @jax.jit
    def step(x, state, kw_):
        fn = lambda p: f(p, kw_)                                   # noqa: E731
        value, grad = optax.value_and_grad_from_state(fn)(x, state=state)
        updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=fn)
        return optax.apply_updates(x, updates), state, value, grad

    if os.path.exists(state_path):
        with open(state_path, "rb") as fh:
            st = pickle.load(fh)
        if st["fingerprint"] != run_fingerprint:
            raise RuntimeError(f"Refusing incompatible refine resume in {state_path}; "
                               "HOUFIN_REFINE_FRESH=1 archives it and starts over.")
        x, state, history = jnp.asarray(st["x"]), st["opt_state"], list(st["history"])
        # A max_iter stop is not terminal: raising HOUFIN_REFINE_MAX_ITER and resubmitting
        # continues the run. Convergence and non-finite stops are.
        if st.get("terminal"):
            log(f"[resume] already finished: {st['reason']}")
            return x, history, st["reason"]
        log(f"[resume] iteration {len(history)}, U={history[-1][1]:.4f}")
    else:
        x, state, history = jnp.asarray(x0), opt.init(jnp.asarray(x0)), []

    def save(reason=None, terminal=False):
        save_pickle_atomic({"fingerprint": run_fingerprint, "x": np.asarray(x),
                            "opt_state": jax.device_get(state), "history": history,
                            "reason": reason, "terminal": terminal}, state_path)

    t0, it0, reason = time.time(), len(history), None
    while len(history) < s["max_iter"]:
        x_new, state_new, value, grad = step(x, state, kw)
        value, gnorm = float(value), float(jnp.linalg.norm(grad))
        if not (np.isfinite(value) and np.isfinite(gnorm)):
            reason = f"non-finite U or grad at iteration {len(history)}; kept last finite point"
            save(reason, terminal=True)
            return x, history, reason
        # value/grad belong to x (the point before this update): record, then advance.
        history.append((len(history), value, gnorm))
        x, state = x_new, state_new
        n = len(history)
        if n % s["log_every"] == 0 or n == it0 + 1:
            rate = (n - it0) / max(time.time() - t0, 1e-9)
            drop = history[max(0, n - 1 - s["tol_window"])][1] - value
            log(f"[lbfgs] iter {n} U={value:.4f} |grad|={gnorm:.3g} "
                f"drop over last {s['tol_window']}={drop:.3g} ({rate:.2f} it/s)")
        reason = converged(history, s)
        if reason:
            save(reason, terminal=True)
            return x, history, reason
        if n % s["ckpt_every"] == 0:
            save()
    reason = f"max_iter {s['max_iter']} reached (not converged; raise HOUFIN_REFINE_MAX_ITER to continue)"
    save(reason, terminal=False)
    return x, history, reason


def run_refine():
    from src.config_utils import load_age_model_config
    from src.model.age_priors import build_model_2d
    from src.model.data_loading import load_data
    from src.model.runtime_diagnostics import memory_snapshot, require_gpu

    pcfg = hc.load_posterior_config()
    s = refine_settings(pcfg)
    if hc.PRECISION != "float64":
        print("[warn] refining in float32: U carries a ~1 nat noise floor, so tol_du below "
              "that cannot be met reliably. Set HOUFIN_MODEL_PRECISION=float64.")
    out_dir = hc.posterior_dir(pcfg, "refine", at_map_precision=True, precision=hc.PRECISION)
    hc.archive_if_fresh(out_dir, env="HOUFIN_REFINE_FRESH", files=(STATE_FILE,))
    os.makedirs(out_dir, exist_ok=True)
    print(f"--- L-BFGS refine of {hc.map_dir()} in {hc.PRECISION} | {s} -> {out_dir} ---")

    device = require_gpu("age-model MAP refinement")
    map_latents, map_ckpt = hc.load_map()
    hc.check_model_drift(map_ckpt["fingerprint_payload"])
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=hc.PRECISION, verbose=False)
    arrays, static = hc.split_data(data)
    kw = {"arrays": arrays}
    model = hc.hide_deterministics(hc.array_model(build_model_2d, static))
    z0, potential, postprocess = hc.potential_with_args(model, kw, map_latents)
    names, x0, unravel = hc.flatten_sorted(z0)
    memory_snapshot("refine-inputs-loaded", device)

    # max_iter and the logging/checkpoint cadences are left out, so raising max_iter and
    # resubmitting continues the run (the same rule age_run_map applies to its step target).
    defining = {k: v for k, v in s.items() if k not in ("max_iter", "ckpt_every", "log_every")}
    payload = {"settings": defining, "precision": hc.PRECISION, "map_fingerprint": map_ckpt["fingerprint"],
               "versions": hc.versions(), "sources": hc.source_identities([__file__])}
    run_fp = hc.fingerprint(payload)
    f = lambda x, kw_: potential(unravel(x), kw_)                  # noqa: E731
    x, history, reason = lbfgs_minimize(f, x0, kw, s, os.path.join(out_dir, STATE_FILE), run_fp)
    print(f"[stop] {reason}")

    latents = jax.device_get(postprocess(unravel(x), kw))
    latents = {k: np.asarray(latents[k]) for k in names}
    ckpt = copy.deepcopy(map_ckpt)
    ckpt.update({
        "params": {f"{k}_auto_loc": v for k, v in latents.items()},
        "losses": np.asarray([h[1] for h in history]),
        "step": len(history),
        # A new fingerprint, so a probe computed at the unrefined MAP is never reused here.
        "fingerprint": hc.fingerprint({"refine": run_fp, "x": np.asarray(x).tobytes().hex()}),
        "refine": {"source_map_dir": hc.map_dir(), "source_map_fingerprint": map_ckpt["fingerprint"],
                   "settings": s, "reason": reason, "precision": hc.PRECISION},
    })
    save_pickle_atomic(ckpt, os.path.join(out_dir, "map_checkpoint.pkl"))
    save_pickle_atomic(ckpt["params"], os.path.join(out_dir, "map_params.pkl"))

    dx = np.asarray(x, np.float64) - np.asarray(x0, np.float64)
    off = np.cumsum([0] + [int(np.size(z0[k])) for k in names])
    report = {
        "source_map_dir": hc.map_dir(), "out_dir": out_dir, "reason": reason,
        "iterations": len(history), "U_start": history[0][1] if history else None,
        "U_end": history[-1][1] if history else None,
        "grad_norm_start": history[0][2] if history else None,
        "grad_norm_end": history[-1][2] if history else None,
        "rms_change_by_site": {k: float(np.sqrt(np.mean(dx[off[i]:off[i + 1]] ** 2)))
                               for i, k in enumerate(names)},
    }
    with open(os.path.join(out_dir, "refine_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[done] U {report['U_start']:.2f} -> {report['U_end']:.2f} in {len(history)} "
          f"iterations; use HOUFIN_HMC_MAP_DIR={out_dir}")


if __name__ == "__main__":
    run_refine()
