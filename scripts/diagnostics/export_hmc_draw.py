"""Write one HMC draw as a MAP-format checkpoint, so the MAP tooling can read it.

The visualization suite (scripts/viz/map_diagnostics.py) and anything else built on
checkpoints.load_map_params expect ``<dir>/map_checkpoint.pkl``: format_version 2 with
constrained ``<site>_auto_loc`` params. This copies the source MAP's checkpoint (so the
fingerprint and step/loss bookkeeping stay verifiable) and swaps in the chosen draw,
recording where it came from under ``hmc_source``.

Run with the same environment as the HMC run (overlay, profile, precisions), e.g.

    AGE_MODEL_CONFIG=config/overlays/map_new_z_release_allee.json HOUFIN_MAP_PROFILE=quick90 \\
    HOUFIN_MODEL_PRECISION=float64 HOUFIN_HMC_MAP_PRECISION=float32 \\
    python scripts/diagnostics/export_hmc_draw.py --variant map_laplace_fixed
    # -> results_dir/hmcdraw_best__<map_run>__float64/map_checkpoint.pkl

then point the suite at it:  HOUFIN_VIZ_RUN_DIR=<that dir> HOUFIN_MODEL_PRECISION=float64 \\
    bash scripts/tacc/submit_map_viz.sh
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model import hmc_common as hc                            # noqa: E402
from src.model.checkpoints import save_pickle_atomic               # noqa: E402


def pick(variant_dir: str, which: str):
    """(chain_name, index, U, {site: value}) for 'best' (lowest potential) or 'chain_XX:N'."""
    chains = sorted(glob.glob(os.path.join(variant_dir, "chain_*")))
    if not chains:
        raise FileNotFoundError(f"no chain_* under {variant_dir}")
    best = None
    for cdir in chains:
        loaded = hc.load_chain(cdir)
        if loaded is None:
            continue
        samples, extra = loaded[0], loaded[1]
        u = np.asarray(extra["potential_energy"])
        name = os.path.basename(cdir)
        if which == "best":
            i = int(np.argmin(u))
        elif which.split(":")[0] == name:
            i = int(which.split(":")[1])
        else:
            continue
        cand = (name, i, float(u[i]), {k: np.asarray(v[i]) for k, v in samples.items()
                                         if k != "auto_shared_latent"})
        if best is None or cand[2] < best[2]:
            best = cand
    if best is None:
        raise ValueError(f"no draw matched {which!r} in {variant_dir}")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="e.g. map_laplace_fixed or neutra_rank10")
    ap.add_argument("--which", default="best", help="'best' (lowest U) or 'chain_00:389'")
    ap.add_argument("--name", default=None, help="output directory name under results_dir")
    args = ap.parse_args()

    pcfg = hc.load_posterior_config()
    variant_dir = hc.posterior_dir(pcfg, "hmc", variant=args.variant)
    chain, idx, u, latents = pick(variant_dir, args.which)
    _, map_ckpt = hc.load_map()
    ckpt = copy.deepcopy(map_ckpt)
    ckpt["params"] = {f"{k}_auto_loc": v for k, v in latents.items()}
    ckpt["hmc_source"] = {"variant_dir": variant_dir, "chain": chain, "draw": idx,
                          "potential_energy": u, "precision": hc.PRECISION}
    tag = "best" if args.which == "best" else args.which.replace(":", "_")
    map_run = os.path.basename(hc.map_dir().rstrip(os.sep))
    name = args.name or f"hmcdraw_{tag}__{map_run}__{hc.PRECISION}"
    out_dir = hc.results_dir(name)
    save_pickle_atomic(ckpt, os.path.join(out_dir, "map_checkpoint.pkl"))
    save_pickle_atomic(ckpt["params"], os.path.join(out_dir, "map_params.pkl"))
    print(f"{chain} draw {idx} (U={u:.1f}) -> {out_dir}")


if __name__ == "__main__":
    main()
