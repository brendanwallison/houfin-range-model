"""Probe the posterior at the MAP point before spending GPU-days sampling it.

Measures what decides whether NUTS on this model is feasible and how to configure it:

* the wall time of one gradient of the potential (a full 1902->present simulation),
  from which the cost of a NUTS draw at each tree depth follows (2^depth - 1 grads);
* the gradient norm at MAP in UNCONSTRAINED space. AutoDelta optimizes the joint in
  constrained space without the change-of-variables Jacobian, so the MAP point is not
  exactly the mode NUTS sees; a large norm here says the Laplace centre is off;
* the dense Hessian there, its eigen-spectrum (negative / near-zero eigenvalues =
  ridge or saddle, not a clean mode) and the SoftAbs Laplace inverse mass matrix
  that arm A's dense metrics and every chain's overdispersed start use.

Writes ``probe.json`` (human-readable report) and ``laplace.npz`` to
``results_dir/<run_names.probe>``. The Hessian is checkpointed per column block, so a
wall-clock kill resumes it.
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
import numpy as np                                                 # noqa: E402

from src.config_utils import load_age_model_config                # noqa: E402
from src.model.age_priors import build_model_2d                    # noqa: E402
from src.model.checkpoints import save_pickle_atomic               # noqa: E402
from src.model.data_loading import load_data                       # noqa: E402
from src.model.runtime_diagnostics import memory_snapshot, require_gpu  # noqa: E402


def run_probe():
    pcfg = hc.load_posterior_config()
    p = pcfg["probe"]
    grad_reps = hc.setting(p, "grad_reps", "HOUFIN_PROBE_GRAD_REPS", int)
    hvp_chunk = hc.setting(p, "hvp_chunk", "HOUFIN_PROBE_HVP_CHUNK", int)
    eig_floor = hc.setting(p, "eig_floor", "HOUFIN_PROBE_EIG_FLOOR", float)
    save_every = hc.setting(p, "save_every_chunks", "HOUFIN_PROBE_SAVE_EVERY", int)
    do_hessian = os.environ.get("HOUFIN_PROBE_HESSIAN", "1") == "1"
    out_dir = hc.posterior_dir(pcfg, "probe")
    hc.archive_if_fresh(out_dir, env="HOUFIN_PROBE_FRESH",
                        files=("probe.json", "hessian_partial.npz", "laplace.npz"))
    os.makedirs(out_dir, exist_ok=True)

    device = require_gpu("age-model posterior probe")
    map_latents, map_ckpt = hc.load_map()
    print(f"[map] {hc.map_dir()} step {map_ckpt['step']}, fingerprint {map_ckpt['fingerprint'][:12]}")
    hc.check_model_drift(map_ckpt["fingerprint_payload"])
    data = load_data(load_age_model_config()["input_dir"], target_device=device,
                     precision=hc.PRECISION)
    memory_snapshot("probe-inputs-loaded", device)

    # Data ARRAYS are jit arguments throughout (hc.split_data): closed over, every compile
    # embedded them as constants (5.4 GB each in float64) until host memory ran out.
    arrays, static = hc.split_data(data)
    kwargs = {"arrays": arrays}
    model = hc.hide_deterministics(hc.array_model(build_model_2d, static))
    z, potential_fn, _ = hc.potential_with_args(model, kwargs, map_latents)
    names, x0, unravel = hc.flatten_sorted(z)
    sizes = {k: int(np.size(z[k])) for k in names}
    print(f"[dims] d={x0.size} across {len(names)} sites; largest: "
          + ", ".join(f"{k}={v}" for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:6]))

    vg_ = jax.jit(jax.value_and_grad(lambda x, kw: potential_fn(unravel(x), kw)))
    vg = lambda x: vg_(x, kwargs)                                  # noqa: E731
    t0 = time.time()
    u0, g0 = jax.block_until_ready(vg(x0))
    compile_s = time.time() - t0
    times = []
    for _ in range(grad_reps):
        t0 = time.time()
        jax.block_until_ready(vg(x0))
        times.append(time.time() - t0)
    grad_s = float(np.median(times))
    memory_snapshot("probe-grad", device)
    g = np.asarray(g0, dtype=np.float64)
    g_tree = unravel(g0)
    grad_by_site = {k: float(np.linalg.norm(np.asarray(g_tree[k]))) for k in names}
    report = {
        "map_dir": hc.map_dir(),
        "map_step": int(map_ckpt["step"]),
        "map_fingerprint": map_ckpt["fingerprint"],
        "precision": hc.PRECISION,
        "d": int(x0.size),
        "site_sizes": sizes,
        "potential_at_map": float(u0),
        "finite": bool(np.isfinite(u0) and np.isfinite(g).all()),
        "grad_norm_at_map": float(np.linalg.norm(g)),
        "grad_norm_by_site_top10": dict(sorted(grad_by_site.items(), key=lambda kv: -kv[1])[:10]),
        "compile_seconds": compile_s,
        "grad_seconds_median": grad_s,
        "grad_seconds_all": times,
        "seconds_per_nuts_draw_at_depth": {d: grad_s * (2 ** d - 1) for d in range(3, 11)},
    }
    print(f"[grad] compile {compile_s:.1f}s, value_and_grad {grad_s:.3f}s median; "
          f"|grad| at MAP={report['grad_norm_at_map']:.3g}; potential={float(u0):.4f}")
    for depth, s in report["seconds_per_nuts_draw_at_depth"].items():
        print(f"  saturated depth {depth}: {s / 60:.1f} min/draw")
    _write(out_dir, report)

    if do_hessian and report["finite"]:
        names_h, xh, H = hc.dense_hessian(
            potential_fn, z, chunk=hvp_chunk,
            partial_path=os.path.join(out_dir, "hessian_partial.npz"), save_every=save_every,
            kwargs=kwargs)
        inv_mass, evals, evecs, eig_report = hc.laplace_inverse_mass(H, eig_floor)
        laplace_sd = np.sqrt(np.diag(inv_mass))
        offsets = np.cumsum([0] + [sizes[k] for k in names_h])
        sd_by_site = {k: [float(laplace_sd[offsets[i]:offsets[i + 1]].min()),
                          float(laplace_sd[offsets[i]:offsets[i + 1]].max())]
                      for i, k in enumerate(names_h)}
        # Which sites the worst-curved directions load on: the nastiness has an address.
        def loading(vec):
            w = {k: float((vec[offsets[i]:offsets[i + 1]] ** 2).sum()) for i, k in enumerate(names_h)}
            return dict(sorted(w.items(), key=lambda kv: -kv[1])[:4])
        eig_report["smallest_eigvec_site_loadings"] = [loading(evecs[:, j]) for j in range(5)]
        eig_report["largest_eigvec_site_loadings"] = [loading(evecs[:, -1 - j]) for j in range(3)]
        report["hessian"] = eig_report
        report["laplace_sd_range_by_site"] = sd_by_site
        np.savez(os.path.join(out_dir, "laplace.npz"),
                 site_names=np.array(names_h), site_sizes=np.array([sizes[k] for k in names_h]),
                 z_map=xh, hessian=H, eigvals=evals, inv_mass=inv_mass,
                 eig_floor=np.array(eig_floor), map_fingerprint=np.array(map_ckpt["fingerprint"]))
        print(f"[hessian] eig range [{eig_report['eig_min']:.3g}, {eig_report['eig_max']:.3g}], "
              f"{eig_report['n_negative']} negative, {eig_report['n_below_floor']} below floor "
              f"{eig_floor:g}; clipped condition number {eig_report['condition_number_clipped']:.3g}")
        _write(out_dir, report)
    save_pickle_atomic({"z_map_unconstrained": jax.device_get(z)},
                       os.path.join(out_dir, "z_map.pkl"))
    print(f"[done] {out_dir}/probe.json")


def _write(out_dir, report):
    tmp = os.path.join(out_dir, "probe.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    os.replace(tmp, os.path.join(out_dir, "probe.json"))


if __name__ == "__main__":
    run_probe()
