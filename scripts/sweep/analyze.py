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
    """``<prefix>_<cell>_<frac>_<config>[_s<seed>]`` -> (cell, config, seed).

    ``nest`` is accepted alongside ``sweep``: the nesting arm uses its own run-id prefix so its
    output dirs cannot collide with stage 1's, and a prefix check that only knew "sweep" dropped
    every one of those runs silently -- an empty table reads identically to a null result.
    """
    p = run_id.split("_")
    if p[0] not in ("sweep", "nest") or len(p) < 4:
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


def _end_of_training(rows, col, frac=0.1):
    """Median of the last ``frac`` of epochs -- the OVER-TRAINED value, not a robust optimum.

    Originally introduced as a "spike-free alternative to the argmin", and that description was
    wrong in a way that mattered. These runs peak at epoch 111-242 of 500 and degrade afterwards,
    so the last 10% of epochs is 451-500: this measures how far a configuration has over-trained
    past its own optimum, which is a different quantity from its best achievable kernel. Comparing
    it against the argmin ranking and calling the differences "false leaders" was therefore
    apples-to-oranges -- the two are SUPPOSED to disagree, which is why 18 of 19 configurations
    "moved".

    Kept, because over-training resistance is worth seeing, and renamed so it cannot be mistaken
    for a robust estimate of the optimum. The actual robustness check is now the rank shift between
    the raw and smoothed argmin, which compares two estimates of the SAME quantity.
    """
    v = [r[col] for r in rows if r.get(col) is not None and np.isfinite(r[col])]
    if not v:
        return float("nan")
    return float(np.median(v[-max(3, int(len(v) * frac)):]))


def _smoothed_min(series, window):
    """``(value, epoch)`` of the trailing-median minimum. The robust argmin.

    The plain argmin of a noisy series is not a property of the model: 14 of 19 stage-1
    configurations moved >= 2 places between the argmin ranking and the spike-free tail ranking,
    and one moved 15. A trailing median over a few epochs removes the lucky single evaluation
    without shifting the estimand, and it can be computed from the recorded trajectory -- so the
    robust ranking costs nothing, where re-running 19 configurations under
    ``desk.selection_smooth`` costs ~17 GPU-hours.

    The window is trailing rather than centred so the reported epoch is one the run actually
    reached, which matters because that epoch is what a production retrain is told to stop at.
    """
    vals = [(x["epoch"], x["v"]) for x in series if np.isfinite(x["v"])]
    if not vals:
        return float("nan"), None
    w = max(1, int(window))
    best = (float("inf"), None)
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        med = float(np.median([v for _e, v in vals[lo:i + 1]]))
        if i + 1 >= w and med < best[0]:
            best = (med, vals[i][0])
    return best if best[1] is not None else (vals[0][1], vals[0][0])


def stage1(runs, threshold, smooth=0):
    col = "kernel_val"
    rows = []
    for r in runs:
        cell, cfg, seed = _cfg_and_cell(r["_run_id"])
        if cell != "t0_f100" or seed is not None:
            continue
        # Ranked on the TRAJECTORY's own minimum, not on run_summary's recorded best value.
        # Those differ whenever min_delta rejected a genuine improvement, and a ranking built on
        # a mix of true minima and early-stopped values compares the selection epsilon as much
        # as the configurations. Reading the trajectory also means a min_delta bug is recoverable
        # from artifacts already on disk instead of costing a rerun.
        kv = [(x["kernel_val"], x["epoch"]) for x in r["_rows"]
              if x.get("kernel_val") is not None and np.isfinite(x["kernel_val"])]
        zv = [(x["zmse_val"], x["epoch"]) for x in r["_rows"]
              if x.get("zmse_val") is not None and np.isfinite(x["zmse_val"])]
        raw_min, raw_ep = min(kv) if kv else (float("nan"), None)
        sm_min, sm_ep = (_smoothed_min([{"epoch": e, "v": v} for v, e in kv], smooth)
                         if smooth > 1 else (raw_min, raw_ep))
        k_min, k_ep = (sm_min, sm_ep) if smooth > 1 else (raw_min, raw_ep)
        rows.append({
            "config": cfg,
            "best_epoch": k_ep if k_ep is not None else r["best_epoch"],
            "recorded_epoch": r["best_epoch"],
            "raw": raw_min, "raw_epoch": raw_ep,
            "smoothed": sm_min, "smoothed_epoch": sm_ep,
            "kernel": k_min,
            "zmse": (min(zv)[0] if zv else float("nan")),
            "endval": _end_of_training(r["_rows"], col),
            "spike": _spike_factor(r["_rows"], r["best_epoch"], col),
            "epochs": r.get("epochs_budget"),
            "params": r.get("n_params"),
            "smooth": r.get("selection_smooth", 0),
        })
    if not rows:
        print("no stage-1 runs found under this root (expected cell t0_f100)")
        return
    # A table mixing runs made under different estimator settings compares two instruments as
    # much as two configurations. Named here rather than left for the reader to infer from a
    # min_delta note, because the settings are recorded and the check is free.
    setts = {}
    for r in runs:
        cell, cfg, seed = _cfg_and_cell(r["_run_id"])
        if cell == "t0_f100" and seed is None and cfg in {x["config"] for x in rows}:
            setts.setdefault((r.get("metric_pairs"), r.get("eval_kernel_draws")), []).append(cfg)
    if len(setts) > 1:
        print("WARNING: this table mixes runs made under DIFFERENT metric/eval settings:")
        for (mp, dr), cfgs in sorted(setts.items(), key=lambda kv: str(kv[0])):
            print(f"  metric_pairs={mp} eval_kernel_draws={dr}: {len(cfgs)} run(s) {sorted(cfgs)}")
        print("  Those are different instruments, not just different configurations. Rerun so "
              "they share one setting before reading the ranking.\n")
    # Only meaningful without smoothing. With --smooth the table's epoch is the smoothed argmin
    # and run_summary's is the raw one, so they differ BY DESIGN -- reporting that as "min_delta
    # rejected genuine improvements" diagnosed a bug that is fixed and was not the cause.
    off = [x for x in rows if x["recorded_epoch"] != x["best_epoch"]] if smooth <= 1 else []
    if off:
        print(f"NOTE {len(off)}/{len(rows)} runs recorded a best_epoch that is not their "
              f"trajectory's argmin (min_delta rejected genuine improvements). The table below "
              f"is rebuilt from the trajectories, so the ranking is sound; the saved "
              f"CHECKPOINTS are from the recorded epoch and would need a rerun only if a "
              f"checkpoint itself is wanted. Stage 1 needs the ranking, not the weights.\n")
    # The estimator's own noise, pooled across runs. This is the number that says whether the
    # ranking below can resolve anything at all: a spread between configurations smaller than the
    # noise of the quantity they are ranked by is not a finding. Stage 1's total spread was 8%
    # with no error bar available, which is exactly the situation this closes.
    # Pooled ONLY over the runs in the table. Pooling over every run under the root attributed a
    # noise floor measured on a smoke run (8 draws) to a table of single-draw runs that have no
    # error bar at all -- a number borrowed from a different configuration and printed as if it
    # described these. The floor is a property of the estimator each run used, so it may only be
    # quoted for runs that used it.
    in_table = {x["config"] for x in rows}
    sds, n_with, n_without = [], 0, 0
    for r in runs:
        cell, cfg, seed = _cfg_and_cell(r["_run_id"])
        if cell != "t0_f100" or seed is not None or cfg not in in_table:
            continue
        v = [x.get("kernel_val_sd") for x in r["_rows"]]
        v = [x for x in v if x is not None and np.isfinite(x) and x > 0]
        k = [x.get("kernel_val") for x in r["_rows"]]
        k = [x for x in k if x is not None and np.isfinite(x)]
        if v and k:
            sds.append(float(np.median(v)) / float(np.median(k))); n_with += 1
        else:
            n_without += 1
    noise_floor = float(np.median(sds)) if sds else None
    if noise_floor is not None and n_without:
        print(f"WARNING: {n_with} of {n_with + n_without} runs in the table carry an error bar. "
              f"The floor below describes only those; the rest were run single-draw and their "
              f"margins cannot be judged against it. Rerun them with desk.eval_kernel_draws > 1 "
              f"before comparing.\n")
        noise_floor = None if n_with < (n_with + n_without) / 2 else noise_floor
    if noise_floor is not None:
        nd = int(next((x.get("kernel_val_draws", 1) for r in runs for x in r["_rows"]), 1))
        se = noise_floor / max(nd, 1) ** 0.5
        print(f"estimator noise floor: {100 * noise_floor:.2f}% per draw, "
              f"{100 * se:.2f}% on the mean of {nd} draws.")
        print(f"  A margin below {100 * se:.2f}% is sampling error, not a difference between "
              f"configurations.\n")
    else:
        print("estimator noise floor: UNAVAILABLE (single-draw runs). Raise "
              "desk.eval_kernel_draws -- without it, no margin below can be called resolvable.\n")

    base = next((x for x in rows if x["config"] == "base"), None)
    if base is None:
        print("WARNING: no `base` run -- every margin below is unanchored")
    rows.sort(key=lambda x: (np.inf if not np.isfinite(x["kernel"]) else x["kernel"]))

    if smooth > 1:
        print(f"ranked on the {smooth}-epoch TRAILING MEDIAN of {col}, not its raw argmin. The "
              f"argmin of a noisy series is not a property of the model, and this metric swings "
              f"~2x between adjacent epochs at high LR. Recomputed from the recorded "
              f"trajectories, so no rerun was needed.\n")
    print(f"{'config':<8} {'ep':>4} {'kernel':>10} {'endval':>10} {'spike':>6} "
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
              f"{x['endval']:>10.6f} {x['spike']:>6.2f} {x['zmse']:>8.4f} "
              f"{100 * rel:>8.1f}%  {verdict}{flag}")
    print()
    print(f"threshold = {100 * threshold:.0f}% relative. PROVISIONAL until stage 3 measures the "
          f"seed-to-seed spread; rerun with --threshold <measured> then.")
    if noise_floor is not None:
        nd = int(next((x.get("kernel_val_draws", 1) for r in runs for x in r["_rows"]), 1))
        se = noise_floor / max(nd, 1) ** 0.5
        unresolvable = [x["config"] for x in rows if base and base["kernel"]
                        and abs(base["kernel"] - x["kernel"]) / base["kernel"] < se
                        and x["config"] != "base"]
        if unresolvable:
            print(f"{len(unresolvable)} configuration(s) differ from baseline by LESS than the "
                  f"estimator's own standard error ({100 * se:.2f}%): {unresolvable}. Those are "
                  f"indistinguishable from measurement noise regardless of the threshold.")
        if threshold > 5 * se:
            print(f"NOTE the threshold ({100 * threshold:.1f}%) is {threshold / se:.0f}x the "
                  f"estimator's standard error ({100 * se:.2f}%), so it is limited by the "
                  f"SEED-TO-SEED spread it stands in for, not by the measurement. Stage 3 "
                  f"measures that spread; until then margins between {100 * se:.2f}% and "
                  f"{100 * threshold:.1f}% are resolvable by the instrument but unproven "
                  f"against training noise.")
        if threshold < se:
            print(f"WARNING: the threshold ({100 * threshold:.1f}%) is BELOW the estimator's "
                  f"standard error ({100 * se:.2f}%), so it cannot be met by evidence. Raise the "
                  f"threshold, raise eval_kernel_draws, or raise eval_kernel_pairs.")
    n_spike = sum(1 for x in rows if np.isfinite(x["spike"]) and x["spike"] >= 2)
    if n_spike:
        print(f"{n_spike}/{len(rows)} runs selected at a spike (a neighbour >=2x the chosen "
              f"value). Their kernel column is biased low, so the ranking above partly ranks "
              f"which run got the luckier evaluation. Use --smooth N: the rank-stability block "
              f"below compares the raw and smoothed argmin, which is the check that applies.")
    # ROBUSTNESS: compare the two estimates of the SAME quantity -- the raw argmin and the
    # smoothed argmin. The previous check compared the argmin against the end-of-training value,
    # which for runs that peak at epoch ~120 of 500 is the over-trained state: a different
    # quantity, so it disagreed for 18 of 19 configurations by construction and said nothing
    # about noise.
    if smooth > 1:
        by_raw = {x["config"]: i for i, x in
                  enumerate(sorted(rows, key=lambda y: (np.inf if not np.isfinite(y["raw"])
                                                        else y["raw"])))}
        by_sm = {x["config"]: i for i, x in
                 enumerate(sorted(rows, key=lambda y: (np.inf if not np.isfinite(y["smoothed"])
                                                       else y["smoothed"])))}
        shifts = {c: by_sm[c] - by_raw[c] for c in by_raw}
        movers = {c: d for c, d in shifts.items() if abs(d) >= 3}
        print(f"\nrank stability, raw argmin vs {smooth}-epoch smoothed argmin "
              f"(same quantity, two estimators):")
        print(f"  median |shift| {np.median([abs(d) for d in shifts.values()]):.0f} place(s), "
              f"max {max(abs(d) for d in shifts.values())}; "
              f"{len(movers)}/{len(shifts)} move >= 3 places")
        if movers:
            print(f"  unstable: {sorted(movers, key=lambda c: -abs(movers[c]))[:6]}")
            print("  A configuration whose rank depends on which estimator is used has not been "
                  "separated\n  from its neighbours by this measurement, whatever the table's "
                  "ordering says.")
        else:
            print("  The ordering is the same under both estimators, so it is not an artifact of "
                  "the spike.")
        top4 = [x["config"] for x in rows[:4]]
        stable_top4 = [c for c in top4 if abs(shifts.get(c, 99)) < 3]
        print(f"\ntop 4 on the smoothed ranking: {top4}")
        print(f"  of those, rank-stable across both estimators: {stable_top4 or 'NONE'}")
        if len(stable_top4) < 4:
            print(f"  Carry {stable_top4 or 'nothing'} forward on this evidence; the rest are "
                  f"not separated from the field.")
    else:
        print("\nrun again with --smooth 5 for a rank-stability check: the raw argmin of this "
              "metric is noise-dominated (it swings ~2x between adjacent epochs at high LR).")
    print(f"\nend-of-training value (epochs {int(0.9 * 500)}+, i.e. AFTER the optimum) is the "
          f"`endval` column above -- it measures over-training resistance, a different question "
          f"from the best achievable kernel. Not a robustness check.")

def eigenbasis_table(runs):
    """Basis quality across configurations, side by side. Diagnostic only -- never ranked on.

    Selection is the held-out kernel alone. This table exists for a different question: whether the
    representation is an ORDERED eigenbasis, which every dot-product metric is blind to and which
    the downstream's positional truncation depends on. Read across configurations rather than per
    run, because the interesting signal turned out to be a trend against the metric weight that no
    single run shows.
    """
    rows = []
    for r in runs:
        cell, cfg, seed = _cfg_and_cell(r["_run_id"])
        if cell != "t0_f100" or seed is not None:
            continue
        e = [x for x in r["_rows"] if "eig_nesting" in x]
        rc = sorted((int(k.rsplit("_r", 1)[1]), r["_rows"][-1][k]) for k in r["_rows"][-1]
                    if k.startswith("kernel_val_ema_r")) if r["_rows"] else []
        if not e and not rc:
            rows.append({"config": cfg, "missing": True})
            continue
        last = e[-1] if e else {}
        best_r, best_v = (min(rc, key=lambda rv: rv[1]) if rc else (None, None))
        full_v = rc[-1][1] if rc else None
        rows.append({
            "config": cfg, "missing": False,
            "metric_weight": r.get("metric_weight"),
            "best_rank": best_r,
            "penalty": (100 * (full_v / best_v - 1) if best_v else float("nan")),
            "inversions": last.get("eig_spectrum_inversions"),
            "first_inv": last.get("eig_first_inversion"),
            "offdiag": last.get("eig_max_offdiag"),
            "gap": last.get("eig_nesting_gap"),
            "gap_sd": last.get("eig_nesting_gap_sd"),
            "nest": last.get("eig_nesting"),
            "ratio": last.get("eig_nesting_ratio"),
            "sub24": last.get("eig_subspace_r24"),
        })
    if not rows:
        return
    miss = [x["config"] for x in rows if x["missing"]]
    rows = [x for x in rows if not x["missing"]]
    print("\n=== eigenbasis diagnostics (NOT selected on) ===")
    if miss:
        print(f"no diagnostics recorded for {sorted(miss)} -- those runs predate them or ran "
              f"with desk.eigenbasis_batch=0")
    if not rows:
        return
    # Ordered by the NESTING GAP, which is the NeuralSVD loss against ESK's value on the same
    # batch. It is one scalar (operator term + metric term), not a composite of many pieces, and
    # it is the closest thing here to "how far is this a genuine ordered eigenbasis".
    print(f"{'config':<8} {'w':>4} {'bestR':>6} {'all-64':>7} {'inv':>4} {'1st':>4} "
          f"{'offdiag':>8} {'sub@24':>7} {'nest gap':>9} {'+-':>7} {'op/met':>7}")
    print("-" * 84)
    def _g(x):
        return x["gap"] if x["gap"] is not None and np.isfinite(x["gap"]) else 1e9
    for x in sorted(rows, key=_g):
        print(f"{x['config']:<8} {x['metric_weight'] or 0:>4.0f} {x['best_rank'] or 0:>6} "
              f"{x['penalty']:>6.1f}% {x['inversions'] or 0:>4} {x['first_inv'] or 0:>4} "
              f"{(x['offdiag'] if x['offdiag'] is not None else float('nan')):>8.3f} "
              f"{(x['sub24'] if x['sub24'] is not None else float('nan')):>7.3f} "
              f"{(x['gap'] if x['gap'] is not None else float('nan')):>9.4f} "
              f"{(x['gap_sd'] if x['gap_sd'] is not None else float('nan')):>7.4f} "
              f"{(x['ratio'] if x['ratio'] is not None else float('nan')):>7.3f}")
    # Is the gap's spread across configurations bigger than the gap's own sampling noise? Without
    # this the ordering above is just an ordering.
    gaps = [x["gap"] for x in rows if x["gap"] is not None and np.isfinite(x["gap"])]
    sds = [x["gap_sd"] for x in rows
           if x["gap_sd"] is not None and np.isfinite(x["gap_sd"]) and x["gap_sd"] > 0]
    if len(gaps) > 1:
        spread = 100 * (max(gaps) / min(gaps) - 1)
        if sds:
            nd = int(next((r.get("eig_nesting_gap_draws", 1) for x in rows for r in [{}]), 1))
            rel = 100 * float(np.median(sds)) / float(np.median(gaps))
            print(f"\nnesting gap: {spread:.0f}% spread across configurations, against a "
                  f"per-batch sd of {rel:.1f}%.")
            if spread > 3 * rel:
                print("  The spread is well above the diagnostic's own noise, so the ordering "
                      "above is a real\n  difference between configurations -- and note it is a "
                      "DIFFERENT ordering from the kernel's.")
            else:
                print("  The spread is NOT clearly above the noise; do not read the ordering.")
        else:
            print(f"\nnesting gap: {spread:.0f}% spread across configurations, but NO error bar "
                  f"(single batch).\n  Raise desk.eigenbasis_draws before reading this ordering "
                  f"-- a more discriminating number\n  without its noise is how the kernel "
                  f"metric misled.")
    # The trend that no single run shows: does pushing the kernel objective make more of the
    # 64 dimensions usable? That is the evidence for or against replacing the stabilizing
    # mixture with an explicit orthogonality term.
    byw = {}
    for x in rows:
        w = x["metric_weight"]
        if w is not None and np.isfinite(x["penalty"]):
            byw.setdefault(float(w), []).append(x["penalty"])
    if len(byw) > 2:
        print("\nall-64 cost against metric weight (median over configurations at each weight):")
        for w in sorted(byw):
            print(f"  w={w:>5.0f}  {np.median(byw[w]):>5.1f}%   ({len(byw[w])} run(s))")
        ws = sorted(byw)
        if np.median(byw[ws[-1]]) < np.median(byw[ws[0]]):
            print("  The cost FALLS as the metric weight rises: pushing the kernel objective makes "
                  "more\n  of the 64 dimensions carry signal. The weight is doing measurable work "
                  "that the\n  full-rank kernel value -- what selection reads -- does not show.")
    n_inv = sum(1 for x in rows if x["best_rank"] and x["best_rank"] < 64)
    if n_inv:
        print(f"\n{n_inv}/{len(rows)} configurations do best at a rank BELOW 64, so latent_dim=64 "
              f"is wider than\nthe data supports in every one of them. The trailing components are "
              f"not merely idle -- they\ndegrade the kernel, and they are not eigen-ordered, so a "
              f"downstream truncating positionally\nto 24 or 32 inherits that.")


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


def nesting_table(runs):
    """The nesting arm, on the numbers that decide it.

    Selection is val_kernel and nothing else. The al_* columns exist because the pure-nesting run
    has no stabilizing term, so nothing pins its basis to ESK's and the RAW rotation-sensitive
    metrics describe an arbitrary rotation rather than the model -- but they are only readable when
    al_fit shows an alignment was actually found. eigbasis columns are the direct test of what the
    nesting objective is FOR: an ordered, orthogonal basis means offdiag -> 0, inv = 0, and the two
    independent eigenvalue estimators agreeing (disagree -> 1).
    """
    print("\n=== nesting arm ===")
    hdr = (f"{'config':<12} {'best_ep':>7} {'k_val':>9} {'vs_base':>8} | "
           f"{'al_fit':>7} {'al_vs':>7} {'al_dcos':>8} {'al_mag':>7} | "
           f"{'offdiag':>8} {'inv':>4} {'disagr':>7} {'ordered':>8}")
    print(hdr); print("-" * len(hdr))
    base_k = None
    got = []
    for r in runs:
        _cell, cfg, _s = _cfg_and_cell(r["_run_id"])
        if cfg is None:
            continue
        rows = r.get("_rows") or []
        if not rows:
            continue
        # kernel_val, NOT val_kernel: val_kernel is the SELECTION-METRIC name (desk.selection_metric)
        # while the trajectory column is kernel_val. Using the wrong one matched nothing and printed
        # an empty table, which is indistinguishable from "the runs are missing".
        kv = [(x.get("kernel_val"), x) for x in rows
              if isinstance(x.get("kernel_val"), (int, float))
              and np.isfinite(x.get("kernel_val"))]
        if not kv:
            continue
        k, row = min(kv, key=lambda t: t[0])
        got.append((cfg, row.get("epoch"), k, row))
        if cfg in ("base", "nest_probe"):
            base_k = k if base_k is None else min(base_k, k)
    for cfg, ep, k, row in sorted(got, key=lambda t: t[2]):
        rel = ((k / base_k - 1.0) * 100.0) if base_k else float("nan")
        def g(key, d=float("nan")):
            v = row.get(key, d)
            return v if isinstance(v, (int, float)) else d
        print(f"{cfg:<12} {ep if ep else '-':>7} {k:>9.5f} "
              f"{rel:>+7.1f}% | {g('al_train_fit'):>7.3f} {g('al_val_zmse'):>7.4f} "
              f"{g('al_dcos_val'):>8.3f} {g('al_mag_val'):>7.3f} | "
              f"{g('eig_offdiag_mean'):>8.3f} {g('eig_spectrum_inversions'):>4.0f} "
              f"{g('eig_estimator_disagreement'):>7.3f} "
              f"{str(row.get('eig_spectrum_descending')):>8}")
    # Name what is absent. An arm run that has not finished has no run_summary.json and so is not
    # loaded at all; without this the table just renders short and reads like a result.
    have = {c for c, _e, _k, _r in got}
    missing = [c for c in ("nest_probe", "tiles32", "nest_only") if c not in have]
    if missing:
        print(f"\n  MISSING from the arm: {', '.join(missing)}. A run still training has no "
              f"run_summary.json and is not loaded, so its absence here is NOT a result. The arm "
              f"is only interpretable complete: nest_probe is the control, tiles32 separates "
              f"tiling from the objective, nest_only is the test.")
    if base_k is None:
        print("  NOTE no base/nest_probe run found, so vs_base is undefined -- the pure-nesting "
              "number alone says nothing without the control.")
    print("\n  Reference points: base's val_kernel ~0.00837, the no-covariate IDW direction "
          "baseline dcos 0.19, and the standard model's dcos ceiling 0.218. An al_dcos below "
          "0.19 is worse than using no covariates at all.")
    print("  Read al_fit FIRST: near 0 means no alignment existed to find, so the al_* columns "
          "beside it are describing noise, not a result.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    # Not type=int: the nesting arm's stage is the string "nesting", and an int-typed argument
    # rejects it outright.
    ap.add_argument("--stage", default="1",
                    help="1 | 2 | nesting")
    ap.add_argument("--smooth", type=int, default=0, metavar="N",
                    help="rank on the N-epoch trailing median of the kernel instead of its raw "
                         "argmin. Removes the lucky-evaluation bias without retraining, since it "
                         "is recomputed from the saved trajectories.")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="relative margin a configuration must beat the baseline by to count "
                         "(default 0.05, PROVISIONAL -- replace with the stage-3 seed spread)")
    args = ap.parse_args()
    runs, _man = load_runs(args.root)
    print(f"{len(runs)} finished run(s) under {args.root}\n")
    if not runs:
        return 1
    # args.stage is a string so "nesting" is accepted; numeric stages become ints.
    stage = int(args.stage) if str(args.stage).strip().isdigit() else str(args.stage).strip()
    if stage == "nesting":
        nesting_table(runs)
        eigenbasis_table(runs)
        return 0
    if stage == 1:
        stage1(runs, args.threshold, smooth=args.smooth)
    else:
        stage2(runs, args.threshold)
    if stage == 1:
        eigenbasis_table(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
