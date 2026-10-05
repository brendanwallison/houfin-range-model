"""Is the potential smooth enough for HMC? A cheap diagnostic, run once per precision.

Both HMC arms on run_18 collapsed their step size during warmup (to 1e-8 for the chain
started exactly at MAP, with acceptance stuck near 0.75). For a smooth potential the
leapfrog energy error shrinks like eps^2, so acceptance -> 1 as eps -> 0; acceptance that
will not rise means the potential has a NOISE FLOOR -- U differs between points 1e-8
apart by more than its smooth variation. The suspect is float32: U ~ 2e5 summed over
~1e5 observations and a 120-year simulation.

Two measurements, written to ``roughness_<precision>.json`` in the MAP-precision probe
directory (run it as float32 and float64; ``STAGE=roughness`` in submit_hmc.sh does both):

1. NOISE FLOOR along a few directions v, scaled by the probe's Laplace metric so delta is
   in posterior sds: r(delta) = U(x + delta v) - U(x) - delta grad.v. A smooth U gives
   |r| ~ delta^2/2; a floor that persists as delta -> 0 is noise in U itself.
2. NON-FINITE REGIONS near MAP: U and |grad| at Laplace draws with jitter 0.25/0.5/1, the
   regime where the jittered chain diverged.
"""
from __future__ import annotations

import json
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import hmc_common as hc                            # noqa: E402  (sets x64)
import jax                                                         # noqa: E402
import jax.numpy as jnp                                            # noqa: E402
import numpy as np                                                 # noqa: E402

from src.config_utils import load_age_model_config                # noqa: E402
from src.model.age_priors import build_model_2d                    # noqa: E402
from src.model.data_loading import load_data                       # noqa: E402
from src.model.runtime_diagnostics import memory_snapshot, require_gpu  # noqa: E402

DELTAS = 10.0 ** np.arange(-9.0, 0.01, 0.5)


def noise_floor(U, x0, v, g_dot_v, deltas=DELTAS, small=1e-5):
    """Residuals of U along x0 + delta v after removing the linear term.

    Returns (rows, floor): rows of (delta, r(+delta), r(-delta)); floor = median |r| over
    delta <= ``small``, where a smooth U (unit curvature in metric units) has |r| <= 5e-11.
    """
    u0 = float(U(x0))
    rows = []
    for d in deltas:
        rp = float(U(x0 + d * v)) - u0 - d * g_dot_v
        rm = float(U(x0 - d * v)) + d * g_dot_v - u0
        rows.append((float(d), rp, rm))
    tiny = [abs(r) for d, rp, rm in rows if d <= small for r in (rp, rm)]
    return rows, float(np.median(tiny)) if tiny else float("nan")


def run_roughness():
    pcfg = hc.load_posterior_config()
    out_dir = hc.posterior_dir(pcfg, "probe", at_map_precision=True)
    laplace = hc.load_laplace(out_dir)
    device = require_gpu("age-model roughness probe")
    map_latents, map_ckpt = hc.load_map()
    hc.check_model_drift(map_ckpt["fingerprint_payload"])
    print(f"[roughness] evaluating in {hc.PRECISION}; MAP {hc.map_dir()} ({hc.MAP_PRECISION})")
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=hc.PRECISION, verbose=False)
    arrays, static = hc.split_data(data)
    kwargs = {"arrays": arrays}
    model = hc.hide_deterministics(hc.array_model(build_model_2d, static))
    z, potential, _ = hc.potential_with_args(model, kwargs, map_latents)
    names, x0, unravel = hc.flatten_sorted(z)
    if tuple(laplace["site_names"].tolist()) != names:
        raise RuntimeError("probe site layout differs from the current model; rerun the probe")
    dtype = x0.dtype

    U_jit = jax.jit(lambda x, kw: potential(unravel(x), kw))
    VG_jit = jax.jit(jax.value_and_grad(lambda x, kw: potential(unravel(x), kw)))
    U = lambda x: U_jit(jnp.asarray(x, dtype), kwargs)               # noqa: E731
    t0 = time.time()
    u0, g0 = VG_jit(x0, kwargs)
    g0 = np.asarray(g0, np.float64)
    print(f"[roughness] compile+first eval {time.time() - t0:.0f}s; U(MAP)={float(u0):.6f}")
    memory_snapshot("roughness-compiled", device)

    L = np.linalg.cholesky(laplace["inv_mass"] + 1e-12 * np.eye(len(x0)))
    H = laplace["hessian"]
    evals, evecs = np.linalg.eigh(H)
    rng = np.random.default_rng(0)
    pulse = np.zeros(len(x0))
    off = np.cumsum([0] + laplace["site_sizes"].tolist())
    i = names.index("log_inv_pulse_fraction")
    pulse[off[i]:off[i + 1]] = 1.0
    raw = {
        "random_1": L @ (u := rng.standard_normal(len(x0))) / np.linalg.norm(u),
        "random_2": L @ (u := rng.standard_normal(len(x0))) / np.linalg.norm(u),
        "stiffest_axis": evecs[:, -1] / np.sqrt(abs(evals[-1])),
        "most_negative_axis": evecs[:, 0] / np.sqrt(abs(evals[0])),
        "pulse_total": pulse / np.sqrt(pulse @ H @ pulse),
    }
    report = {"precision": hc.PRECISION, "map_precision": hc.MAP_PRECISION,
              "U_at_map": float(u0), "float_eps_times_U": float(np.finfo(dtype).eps * abs(float(u0))),
              "directions": {}}
    for label, v in raw.items():
        rows, floor = noise_floor(U, x0, jnp.asarray(v, dtype), float(g0 @ v))
        report["directions"][label] = {"noise_floor": floor, "rows": rows}
        print(f"[floor] {label:20s} median |r| for delta<=1e-5: {floor:.3g}")
        for d, rp, rm in rows[::2]:
            print(f"        delta={d:8.1e}  r(+)={rp: .3e}  r(-)={rm: .3e}  (smooth ~ {d * d / 2:.1e})")

    report["jitter"] = {}
    for jit in (0.25, 0.5, 1.0):
        us, gs, bad = [], [], 0
        for k in range(32):
            x = x0 + jnp.asarray(jit * (L @ rng.standard_normal(len(x0))), dtype)
            u, g = VG_jit(x, kwargs)
            u, gn = float(u), float(jnp.linalg.norm(g))
            if not (np.isfinite(u) and np.isfinite(gn)):
                bad += 1
                continue
            us.append(u - float(u0))
            gs.append(gn)
        q = lambda a: [float(np.quantile(a, p)) for p in (0.05, 0.5, 0.95)] if a else None  # noqa: E731
        report["jitter"][str(jit)] = {"non_finite": bad, "n": 32, "dU_q05_50_95": q(us),
                                      "grad_norm_q05_50_95": q(gs)}
        print(f"[jitter {jit}] non-finite {bad}/32; U-U(MAP) q05/50/95 {q(us)}; |grad| {q(gs)}")

    path = os.path.join(out_dir, f"roughness_{hc.PRECISION}.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(report, fh, indent=2)
    os.replace(path + ".tmp", path)
    print(f"[done] {path}")


if __name__ == "__main__":
    run_roughness()
