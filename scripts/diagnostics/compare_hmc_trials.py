"""Compare HMC trial arms: sampler health, efficiency and cross-chain agreement.

Reads every ``results_dir/hmc_<variant>__<map_run>/chain_XX/`` for the MAP selected by
AGE_MODEL_CONFIG + HOUFIN_MAP_PROFILE (same as the fitting jobs), written by
``age_resume_hmc`` (arm A: map_diag / map_laplace_fixed / map_laplace_adapt) and
``age_run_hmc`` (arm B: neutra_rank<r>) and reports, per variant:

* per chain: draws, GPU minutes, gradient evaluations, divergences, the fraction of
  draws that hit max_tree_depth, final step size, E-BFMI, mean acceptance;
* across chains (truncated to the shortest): worst split R-hat and smallest bulk ESS
  over every scalar component, ESS per 1000 gradients (the number to compare arms on),
  and the components responsible;
* the posterior mean's distance from MAP in posterior sds (large = MAP is not where
  the mass is, or chains have not left it).

    AGE_MODEL_CONFIG=config/overlays/map_new_z.json HOUFIN_MAP_PROFILE=quick90 \
    python scripts/diagnostics/compare_hmc_trials.py                 # all variants
    python scripts/diagnostics/compare_hmc_trials.py map_laplace_fixed neutra_rank10
    python scripts/diagnostics/compare_hmc_trials.py --no-map --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model import hmc_common as hc                            # noqa: E402
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin  # noqa: E402


def e_bfmi(energy: np.ndarray) -> float:
    energy = np.asarray(energy, dtype=np.float64)
    if energy.size < 3:
        return float("nan")
    return float(np.sum(np.diff(energy) ** 2) / np.sum((energy - energy.mean()) ** 2))


def chain_report(chain_dir: str):
    loaded = hc.load_chain(chain_dir)
    if loaded is None:
        return None, None
    samples, extra, seconds, n_chunks = loaded
    with open(os.path.join(chain_dir, hc.STATE_FILE), "rb") as fh:
        payload = pickle.load(fh)["payload"]
    depth = int(payload["settings"]["max_tree_depth"])
    steps = np.asarray(extra["num_steps"])
    rep = {
        "draws": int(steps.size),
        "chunks": n_chunks,
        "gpu_minutes": seconds / 60,
        "grads": int(steps.sum()),
        "grads_per_second": float(steps.sum() / max(seconds, 1e-9)),
        "divergent": int(np.asarray(extra["diverging"]).sum()),
        "frac_max_depth": float((steps >= 2 ** depth - 1).mean()),
        "mean_steps": float(steps.mean()),
        "step_size": float(np.asarray(extra["adapt_state.step_size"]).reshape(-1)[-1]),
        "e_bfmi": e_bfmi(extra["energy"]),
        "accept_mean": float(np.asarray(extra["accept_prob"]).mean()),
    }
    warm = os.path.join(chain_dir, "warmup.pkl")
    if os.path.exists(warm):
        with open(warm, "rb") as fh:
            w = pickle.load(fh)
        ws = np.asarray(w["extra"]["num_steps"])
        rep["warmup"] = {"minutes": w["seconds"] / 60, "grads": int(ws.sum()),
                         "divergent": int(np.asarray(w["extra"]["diverging"]).sum())}
    return rep, samples


def components(samples_by_chain: list, n: int):
    """{label: (chains, n) array} over every scalar component of every latent site."""
    out = {}
    for site in samples_by_chain[0]:
        if site == "auto_shared_latent":
            continue
        stack = np.stack([np.asarray(s[site])[:n] for s in samples_by_chain])
        flat = stack.reshape(stack.shape[0], n, -1)
        for j in range(flat.shape[-1]):
            out[f"{site}[{j}]" if flat.shape[-1] > 1 else site] = flat[:, :, j]
    return out


def variant_report(variant_dir: str, map_values: dict | None):
    chains, per_chain = [], {}
    for cdir in sorted(glob.glob(os.path.join(variant_dir, "chain_*"))):
        rep, samples = chain_report(cdir)
        if rep is not None:
            per_chain[os.path.basename(cdir)] = rep
            chains.append(samples)
    out = {"chains": per_chain}
    if not chains:
        return out
    n = min(next(iter(c.values())).shape[0] for c in chains)
    comps = components(chains, n)
    labels = list(comps)
    X = np.stack([comps[k] for k in labels], axis=-1)            # (chains, n, comps)
    ess = np.asarray(effective_sample_size(X))
    rhat = np.asarray(split_gelman_rubin(X)) if n >= 4 else np.full(len(labels), np.nan)
    total_grads = sum(c["grads"] for c in per_chain.values())
    worst_ess = np.argsort(ess)[:5]
    worst_rhat = np.argsort(-np.nan_to_num(rhat, nan=-np.inf))[:5]
    out.update({
        "n_chains": len(chains),
        "draws_per_chain_used": int(n),
        "min_ess": float(ess.min()),
        "median_ess": float(np.median(ess)),
        "ess_per_1000_grads": float(ess.min() / max(total_grads, 1) * 1000),
        "max_rhat": float(np.nanmax(rhat)) if np.isfinite(rhat).any() else None,
        "frac_rhat_gt_1_01": float(np.nanmean(rhat > 1.01)) if np.isfinite(rhat).any() else None,
        "worst_ess": {labels[i]: float(ess[i]) for i in worst_ess},
        "worst_rhat": {labels[i]: float(rhat[i]) for i in worst_rhat},
    })
    if map_values:
        mean, sd = X.mean(axis=(0, 1)), X.std(axis=(0, 1)) + 1e-30
        mv = np.concatenate([np.asarray(map_values[s]).reshape(-1)
                             for s in chains[0] if s != "auto_shared_latent"])
        if mv.size == mean.size:
            zs = (mean - mv) / sd
            top = np.argsort(-np.abs(zs))[:5]
            out["map_offset_sd"] = {"max_abs": float(np.abs(zs).max()),
                                    "median_abs": float(np.median(np.abs(zs))),
                                    "worst": {labels[i]: float(zs[i]) for i in top}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="*", help="variant names (default: every age_hmc_* dir)")
    ap.add_argument("--no-map", action="store_true", help="skip the MAP-offset comparison")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    pcfg = hc.load_posterior_config()
    dirs = sorted(glob.glob(hc.posterior_dir(pcfg, "hmc", variant="*")))
    if args.variants:
        wanted = {os.path.basename(hc.posterior_dir(pcfg, "hmc", variant=v)) for v in args.variants}
        dirs = [d for d in dirs if os.path.basename(d) in wanted]
    map_values = None
    if not args.no_map:
        try:
            map_values, _ = hc.load_map()
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[warn] MAP comparison skipped: {exc}")

    report = {os.path.basename(d): variant_report(d, map_values) for d in dirs}
    for name, r in report.items():
        print(f"\n=== {name} ===")
        for cname, c in r["chains"].items():
            print(f"  {cname}: {c['draws']} draws, {c['gpu_minutes']:.0f} min, "
                  f"{c['grads']} grads, div={c['divergent']}, maxdepth={c['frac_max_depth']:.0%}, "
                  f"eps={c['step_size']:.3g}, E-BFMI={c['e_bfmi']:.2f}, acc={c['accept_mean']:.2f}")
        if "min_ess" in r:
            print(f"  across {r['n_chains']} chains x {r['draws_per_chain_used']}: "
                  f"min ESS={r['min_ess']:.1f} (median {r['median_ess']:.1f}), "
                  f"ESS/1000 grads={r['ess_per_1000_grads']:.3f}, max R-hat={r['max_rhat']}")
            print(f"  worst ESS: {r['worst_ess']}")
            print(f"  worst R-hat: {r['worst_rhat']}")
        if "map_offset_sd" in r:
            print(f"  posterior mean - MAP (sd units): max |z|={r['map_offset_sd']['max_abs']:.2f}, "
                  f"worst {r['map_offset_sd']['worst']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
