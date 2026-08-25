"""Rank sweep configurations, and fit best-epoch/best-config against data amount.

The two analysis steps the plan's objective needs and its file list omitted. Both read
``run_summary.json`` and ``train_trajectory.jsonl`` from every run under a sweep root -- no
cluster access, no logs.

    python scripts/sweep/analyze.py --root $HOUFIN_PROCESSED/sweeps/desk_hp --stage 1
    python scripts/sweep/analyze.py --root $HOUFIN_PROCESSED/sweeps/desk_hp --stage 2

**stage 1** ranks configurations at the production cell on the held-out kernel term at each
run's own best epoch, and applies the rule the plan states: a knob counts as moving the metric
only if it beats the baseline by more than the seed-to-seed spread. Until stage 3 measures that
spread the threshold is provisional, so it is reported as a threshold and every margin is shown
against it rather than a winner simply being declared.

**stage 2** reports best epoch and best value against the amount of training data (years x
training cells), which is the trajectory the production retrain is read off.

Selection values taken at a spike are flagged, not silently ranked: kernel_val swings ~3x
between adjacent epochs at high LR, so an argmin can be a lucky evaluation rather than a
property of the configuration.
"""
import argparse
import json
import os
import sys

import numpy as np


def load_runs(root):
    """Every finished run under ``root``: its summary, plus its trajectory rows."""
    out = []
    man_path = os.path.join(root, "sweep_manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else None
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        sp = os.path.join(d, "run_summary.json")
        if not os.path.isfile(sp):
            continue
        summ = json.load(open(sp))
        tp = os.path.join(d, "train_trajectory.jsonl")
        rows = [json.loads(l) for l in open(tp)] if os.path.exists(tp) else []
        summ["_run_id"] = name
        summ["_rows"] = rows
        out.append(summ)
    return out, man


def _cfg_and_cell(run_id):
    """``sweep_<cell>_<frac>_<config>[_s<seed>]`` -> (cell, config, seed)."""
    p = run_id.split("_")
    if p[0] != "sweep" or len(p) < 4:
        return None, None, None
    seed = None
    if p[-1].startswith("s") and p[-1][1:].isdigit():
        seed = int(p[-1][1:]); p = p[:-1]
    return f"{p[1]}_{p[2]}", "_".join(p[3:]), seed


def _spike_factor(rows, best_epoch, col):
    """How much worse the selected epoch's worst neighbour is. >=2 means a lucky evaluation."""
    by = {r["epoch"]: r.get(col) for r in rows}
    bv = by.get(best_epoch)
    if bv is None or not np.isfinite(bv) or bv <= 0:
        return float("nan")
    nb = [by[e] for e in (best_epoch - 1, best_epoch + 1)
          if e in by and by[e] is not None and np.isfinite(by[e])]
    return max(nb) / bv if nb else float("nan")


def _stable_tail(rows, col, frac=0.1):
    """Median of the last ``frac`` of epochs -- a spike-free alternative to the argmin.

    Reported alongside the argmin because ranking configurations on a value that may have been
    selected at a spike ranks the luck as well as the configuration. The two agreeing is what
    makes the ranking trustworthy; the two disagreeing is the signal to turn on
    desk.selection_smooth and rerun, not to pick whichever looks better.
    """
    v = [r[col] for r in rows if r.get(col) is not None and np.isfinite(r[col])]
    if not v:
        return float("nan")
    return float(np.median(v[-max(3, int(len(v) * frac)):]))


def stage1(runs, threshold):
    col = "kernel_val"
    rows = []
    for r in runs:
        cell, cfg, seed = _cfg_and_cell(r["_run_id"])
        if cell != "t0_f100" or seed is not None:
            continue
        rows.append({
            "config": cfg,
            "best_epoch": r["best_epoch"],
            "kernel": r.get("best_val_kernel", float("nan")),
            "zmse": r.get("best_val_zmse", float("nan")),
            "tail": _stable_tail(r["_rows"], col),
            "spike": _spike_factor(r["_rows"], r["best_epoch"], col),
            "epochs": r.get("epochs_budget"),
            "params": r.get("n_params"),
            "smooth": r.get("selection_smooth", 0),
        })
    if not rows:
        print("no stage-1 runs found under this root (expected cell t0_f100)")
        return
    base = next((x for x in rows if x["config"] == "base"), None)
    if base is None:
        print("WARNING: no `base` run -- every margin below is unanchored")
    rows.sort(key=lambda x: (np.inf if not np.isfinite(x["kernel"]) else x["kernel"]))

    print(f"{'config':<8} {'ep':>4} {'kernel':>10} {'tail':>10} {'spike':>6} "
          f"{'zmse':>8} {'vs base':>9}  verdict")
    print("-" * 78)
    for x in rows:
        margin = (base["kernel"] - x["kernel"]) if base else float("nan")
        rel = (margin / base["kernel"]) if base and base["kernel"] else float("nan")
        if x["config"] == "base":
            verdict = "(baseline)"
        elif not np.isfinite(rel):
            verdict = "?"
        elif rel > threshold:
            verdict = f"BEATS baseline by {100 * rel:.0f}%"
        elif rel < -threshold:
            verdict = f"worse by {100 * -rel:.0f}%"
        else:
            verdict = f"within +-{100 * threshold:.0f}% -- NOT distinguishable"
        flag = "  <-- SPIKE" if np.isfinite(x["spike"]) and x["spike"] >= 2 else ""
        print(f"{x['config']:<8} {x['best_epoch']:>4} {x['kernel']:>10.6f} "
              f"{x['tail']:>10.6f} {x['spike']:>6.2f} {x['zmse']:>8.4f} "
              f"{100 * rel:>8.1f}%  {verdict}{flag}")
    print()
    print(f"threshold = {100 * threshold:.0f}% relative. PROVISIONAL until stage 3 measures the "
          f"seed-to-seed spread; rerun with --threshold <measured> then.")
    n_spike = sum(1 for x in rows if np.isfinite(x["spike"]) and x["spike"] >= 2)
    if n_spike:
        print(f"{n_spike}/{len(rows)} runs selected at a spike (a neighbour >=2x the chosen "
              f"value). Their kernel column is biased low, so the ranking above partly ranks "
              f"which run got the luckier evaluation. Compare the `tail` column: if it "
              f"disagrees with the ranking, set desk.selection_smooth and rerun stage 1.")
    # Rank by the spike-free column too, and compare the two RANKINGS -- not the two top-4
    # SETS. Set equality was the first thing this printed and it was wrong: with a handful of
    # configurations the top-4 sets can coincide while the order is completely different, so a
    # run that leads only because of a lucky evaluation still gets reported as robust. That is
    # precisely the plausible-looking summary this script exists to prevent, and it produced it.
    by_tail = sorted((x for x in rows if np.isfinite(x["tail"])), key=lambda x: x["tail"])
    rank_k = {x["config"]: i for i, x in enumerate(rows)}
    rank_t = {x["config"]: i for i, x in enumerate(by_tail)}
    print("\nrank by best-epoch kernel vs by stable tail:")
    movers = []
    for cfg in sorted(rank_k, key=lambda c: rank_k[c]):
        rk, rt = rank_k[cfg], rank_t.get(cfg)
        if rt is None:
            continue
        shift = rt - rk
        note = ""
        # A config that ranks much better on the argmin than on the tail is a FALSE LEADER: its
        # winning value came from a spike, not from being a better configuration.
        if shift >= 2:
            note = f"  <-- FALSE LEADER: {shift} places worse on the spike-free column"
            movers.append(cfg)
        elif shift <= -2:
            note = f"  <-- UNDERRATED by the argmin: {-shift} places better on the tail"
            movers.append(cfg)
        print(f"  {cfg:<8} argmin #{rk + 1}  tail #{rt + 1}{note}")
    top_k = [x["config"] for x in rows[:4]]
    top_t = [x["config"] for x in by_tail[:4]]
    print(f"\ntop 4 by best-epoch kernel: {top_k}")
    print(f"top 4 by stable tail:       {top_t}")
    if movers or set(top_k) != set(top_t):
        print(f"  DO NOT carry the argmin top-4 forward as-is. {len(movers)} configuration(s) "
              f"move >=2 places between the two columns ({movers}), so the argmin ranking is "
              f"partly ranking luck. Either set desk.selection_smooth and rerun stage 1, or "
              f"carry the UNION of the two top-4 lists and pay for the extra stage-2 runs: "
              f"{sorted(set(top_k) | set(top_t))}")
    else:
        print("  the two rankings agree and nothing moved >=2 places, so the top 4 are robust "
              "to the spike problem")


def stage2(runs, threshold):
    col = "kernel_val"
    grid = {}
    for r in runs:
        cell, cfg, seed = _cfg_and_cell(r["_run_id"])
        if cell is None or seed is not None:
            continue
        grid.setdefault(cfg, {})[cell] = r
    if not grid:
        print("no stage-2 runs found")
        return
    # Ordered by DATA AMOUNT, descending: temporal row first (t0 has the most years), then
    # training fraction. Alphabetical order puts f100 next to f70 and reads as a scrambled
    # trajectory, which defeats the point of a table whose whole job is to show a trend.
    _t_order = {"t0": 0, "t1975": 1, "t1985": 2, "t1995": 3}
    _f_order = {"f100": 0, "f95": 1, "f85": 2, "f70": 3}
    cells = sorted({c for v in grid.values() for c in v},
                   key=lambda c: (_t_order.get(c.split("_")[0], 99),
                                  _f_order.get(c.split("_")[1], 99)))
    print("best EPOCH by configuration x data cell")
    print(f"{'config':<8} " + " ".join(f"{c:>12}" for c in cells))
    print("-" * (9 + 13 * len(cells)))
    for cfg, byc in sorted(grid.items()):
        print(f"{cfg:<8} " + " ".join(
            f"{byc[c]['best_epoch']:>12}" if c in byc else f"{'-':>12}" for c in cells))
    print()
    print("best held-out KERNEL by configuration x data cell")
    print(f"{'config':<8} " + " ".join(f"{c:>12}" for c in cells))
    print("-" * (9 + 13 * len(cells)))
    for cfg, byc in sorted(grid.items()):
        print(f"{cfg:<8} " + " ".join(
            f"{byc[c].get('best_val_kernel', float('nan')):>12.6f}" if c in byc
            else f"{'-':>12}" for c in cells))
    print()
    # the trajectory the production retrain is read off: does the winner change with data?
    print("winner per cell (lowest held-out kernel):")
    flips = []
    for c in cells:                                    # same data-amount order as the tables
        cand = [(v[c].get("best_val_kernel", np.inf), k) for k, v in grid.items() if c in v]
        if not cand:
            continue
        cand.sort()
        margin = (cand[1][0] - cand[0][0]) / cand[0][0] if len(cand) > 1 and cand[0][0] else 0
        tag = "" if margin > threshold else "  (not distinguishable from 2nd)"
        print(f"  {c:<12} {cand[0][1]:<8} {cand[0][0]:.6f}{tag}")
        flips.append(cand[0][1])
    if len(set(flips)) == 1:
        print(f"\nFLAT OPTIMUM: {flips[0]} wins at every data amount. That is the useful "
              f"outcome the plan named -- one configuration serves everywhere, and the "
              f"production retrain uses it with no extrapolation.")
    else:
        print(f"\nThe winner CHANGES with data amount ({flips}). The production retrain sits at "
              f"the largest-data end, so read the winner there -- and check the trend is "
              f"monotone rather than noise by comparing against the stage-3 seed spread.")
    # epoch vs data amount, at the production end
    print("\nbest epoch against training cell-years (the trajectory to extrapolate along):")
    for cfg, byc in sorted(grid.items()):
        pts = [(v.get("n_train_cell_years"), v["best_epoch"]) for c, v in sorted(byc.items())
               if v.get("n_train_cell_years")]
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            slope = np.polyfit(np.log(xs), ys, 1)[0] if len(pts) >= 2 else float("nan")
            print(f"  {cfg:<8} epochs {min(ys)}..{max(ys)} over {min(xs):,}..{max(xs):,} "
                  f"cell-years  (d(epoch)/d(log cell-years) = {slope:+.0f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="relative margin a configuration must beat the baseline by to count "
                         "(default 0.05, PROVISIONAL -- replace with the stage-3 seed spread)")
    args = ap.parse_args()
    if args.root.startswith(("/sweeps/", "/encoder/")):
        print("ERROR: --root begins at /sweeps or /encoder, so $HOUFIN_PROCESSED expanded to "
              "nothing.\n       Run:  source scripts/tacc/env.sh", file=sys.stderr)
        return 2
    if not os.path.isdir(args.root):
        print(f"ERROR: {args.root} is not a directory", file=sys.stderr)
        return 2
    runs, _man = load_runs(args.root)
    print(f"{len(runs)} finished run(s) under {args.root}\n")
    if not runs:
        return 1
    (stage1 if args.stage == 1 else stage2)(runs, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
