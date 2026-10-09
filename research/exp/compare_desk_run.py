"""Compare a local DESK training run with a TACC reference run (Phase 0.4 reproduction gate).

    python research/exp/compare_desk_run.py <local desk_output_dir> <reference desk_output_dir> [--floor 0.066]

The verdict is on the selection value (best val_kernel): WITHIN_FLOOR when the relative difference is no
larger than the measured seed floor (6.6%, spatial), else DIFFERS. The trajectories are printed at every
50th epoch so a difference can be placed (early dynamics vs the selected epoch). Exits 1 on DIFFERS.
"""
import argparse
import json
import os
import sys

KEYS = ("kernel_val_sp", "zmse_val_sp", "rot_ratio_val", "rot_ratio_withheld", "half_life", "loss_total")


def load(d):
    s = json.load(open(os.path.join(d, "run_summary.json"), encoding="utf-8"))
    traj = {}
    p = os.path.join(d, "train_trajectory.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            traj[int(r["epoch"])] = r
    return s, traj


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("local")
    ap.add_argument("reference")
    ap.add_argument("--floor", type=float, default=0.066)
    a = ap.parse_args()
    (sl, tl), (sr, tr) = load(a.local), load(a.reference)
    rel = (sl["best_val_kernel"] - sr["best_val_kernel"]) / sr["best_val_kernel"]
    verdict = "WITHIN_FLOOR" if abs(rel) <= a.floor else "DIFFERS"
    print(f"DESK reproduction: {verdict}  best val_kernel local {sl['best_val_kernel']:.6f} vs reference "
          f"{sr['best_val_kernel']:.6f} ({rel:+.1%}; floor {a.floor:.1%})")
    for k in ("best_epoch", "best_val_zmse", "epochs_run", "ema_half_life", "n_train_cells", "n_val_cells",
              "n_train_cell_years", "n_params"):
        print(f"  {k:20s} local {sl.get(k)!s:>22s}   reference {sr.get(k)!s:>22s}")
    common = sorted(set(tl) & set(tr))
    if common:
        print("  epoch  " + "  ".join(f"{k:>24s}" for k in KEYS))
        for ep in [e for e in common if e % 50 == 0 or e in (sl["best_epoch"], sr["best_epoch"])]:
            cells = []
            for k in KEYS:
                vl, vr = tl[ep].get(k), tr[ep].get(k)
                cells.append(f"{vl:11.5f}/{vr:<11.5f}" if isinstance(vl, (int, float))
                             and isinstance(vr, (int, float)) else f"{'-':>24s}")
            print(f"  {ep:5d}  " + "  ".join(cells))
    with open(os.path.join(a.local, "..", "compare.json"), "w", encoding="utf-8") as fh:
        json.dump({"verdict": verdict, "relative_difference": rel, "floor": a.floor,
                   "local": sl["best_val_kernel"], "reference": sr["best_val_kernel"]}, fh, indent=1)
    sys.exit(0 if verdict == "WITHIN_FLOOR" else 1)


if __name__ == "__main__":
    main()
