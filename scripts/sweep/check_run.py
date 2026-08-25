"""Verify a finished DESK run's instrumentation, from its artifacts alone.

Run after the smoke check and after any grid run you want to trust. Exists because "the log
looked fine" is how this project has repeatedly accepted a stale or absent number: every check
here is on a file's CONTENT, and several of them are the exact failures already measured --
an empty validation selection scoring a perfect 0.0000, a best epoch recorded nowhere, a
trajectory that only existed in a job log.

    python scripts/sweep/check_run.py $HOUFIN_PROCESSED/sweeps/desk_hp/smoke30ep_base

Exits non-zero if anything is missing or internally inconsistent, so it can gate a submission.
"""
import argparse
import json
import os
import sys

import numpy as np


def _fail(msgs, msg):
    msgs.append(f"FAIL  {msg}")


def _ok(msgs, msg):
    msgs.append(f"ok    {msg}")


def check(run_dir):
    msgs, hard = [], 0
    need = ("desk_meta.npz", "env_model_semisup.pth", "output_ema.pth",
            "train_trajectory.jsonl", "run_summary.json", "holdout_cells.npy",
            "buffer_cells.npy")
    for f in need:
        p = os.path.join(run_dir, f)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            _ok(msgs, f"{f} present ({os.path.getsize(p):,} B)")
        else:
            _fail(msgs, f"{f} MISSING or empty"); hard += 1
    if hard:
        return msgs, hard

    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    summ = json.load(open(os.path.join(run_dir, "run_summary.json")))
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "train_trajectory.jsonl"))]

    if not rows:
        _fail(msgs, "trajectory is empty"); return msgs, hard + 1
    _ok(msgs, f"trajectory has {len(rows)} evaluated epoch(s), "
              f"{rows[0]['epoch']}..{rows[-1]['epoch']}")

    # The best epoch must be recorded AND be the argmin of the column it claims to have
    # selected on. Recorded-but-wrong is worse than absent: it is the number the production
    # retrain is told to stop at.
    sel = summ["selection_metric"]
    col = {"val_kernel": "kernel_val", "val_zmse": "zmse_val"}[sel]
    be = summ["best_epoch"]
    if "best_epoch" not in dm:
        _fail(msgs, "desk_meta.npz has no best_epoch"); hard += 1
    elif int(dm["best_epoch"]) != int(be):
        _fail(msgs, f"best_epoch disagrees: meta {int(dm['best_epoch'])} vs summary {be}")
        hard += 1
    else:
        _ok(msgs, f"best_epoch {be} agrees between desk_meta.npz and run_summary.json")

    smooth = int(summ.get("selection_smooth", 0))
    if summ["restored_best"] and smooth > 1:
        # With a trailing median the selected epoch is deliberately NOT the raw argmin, so
        # comparing against it would hard-fail every smoothed run. Check the smoothed series
        # instead -- the property that still has to hold is that selection used the signal it
        # says it used.
        ser = [r[col] for r in rows]
        med = [float(np.median([v for v in ser[max(0, i - smooth + 1):i + 1]
                                if np.isfinite(v)] or [np.nan])) for i in range(len(ser))]
        cand = [(m, rows[i]["epoch"]) for i, m in enumerate(med) if np.isfinite(m)]
        if cand and min(cand)[1] != be:
            _fail(msgs, f"best_epoch {be} is not the argmin of the {smooth}-epoch trailing "
                        f"median of {col} ({min(cand)[1]})"); hard += 1
        else:
            _ok(msgs, f"best_epoch {be} is the argmin of the {smooth}-epoch trailing median "
                      f"of {col}")
    elif summ["restored_best"]:
        elig = [r for r in rows if np.isfinite(r[col])]
        if not elig:
            _fail(msgs, f"selected on {sel} but every {col} is non-finite"); hard += 1
        else:
            want = min(elig, key=lambda r: r[col])["epoch"]
            # the warmup epochs are excluded from selection, so a best epoch inside the warmup
            # window is the one legitimate way these can differ
            if want != be and be > min(r["epoch"] for r in rows):
                _fail(msgs, f"best_epoch {be} is not the argmin of {col} ({want}) -- selection "
                            f"and the recorded epoch disagree"); hard += 1
            else:
                _ok(msgs, f"best_epoch {be} is the argmin of {col} "
                          f"({min(r[col] for r in elig):.6g})")
    else:
        _ok(msgs, "no epoch selected (expected only for a no-holdout production run)")

    # The val kernel pool must have produced a real number. nan here means it was never wired
    # or held no points -- and if selection was val_kernel that is a run selected on nothing.
    finite_k = sum(np.isfinite(r["kernel_val"]) for r in rows)
    if finite_k == 0:
        lvl = "FAIL" if sel == "val_kernel" else "warn"
        msgs.append(f"{lvl}  kernel_val is non-finite in every epoch -- the validation kernel "
                    f"pool produced nothing")
        hard += (sel == "val_kernel")
    else:
        _ok(msgs, f"kernel_val finite in {finite_k}/{len(rows)} epochs "
                  f"(first {rows[0]['kernel_val']:.6g}, last {rows[-1]['kernel_val']:.6g})")

    # A perfect zero on held-out cells is the measured symptom of an EMPTY val selection, not a
    # perfect fit. Refuse to call such a run healthy.
    for c in ("zmse_val", "kernel_val"):
        z = [r[c] for r in rows if r[c] == 0.0]
        if z:
            _fail(msgs, f"{c} is exactly 0.0 in {len(z)} epoch(s) -- that is what an empty "
                        f"validation set scores, not a perfect fit"); hard += 1

    nv = int(dm["val_cells"]) if "val_cells" in dm else -1
    nt = int(dm["train_cells"]) if "train_cells" in dm else -1
    _ok(msgs, f"{nt:,} train cells / {nv:,} val cells, "
              f"{int(dm['buffer_cells'])} buffered, "
              f"{int(dm.get('train_dropped_cells', 0))} dropped by train_frac")
    if nv == 0 and summ["restored_best"]:
        _fail(msgs, "no val cells yet an epoch was selected -- impossible"); hard += 1

    # Held-out and training cells must not overlap. This is the property the whole measurement
    # rests on, and it is cheap to check from the saved masks.
    ho = np.load(os.path.join(run_dir, "holdout_cells.npy"))
    bf = np.load(os.path.join(run_dir, "buffer_cells.npy"))
    if (ho & bf).any():
        _fail(msgs, "holdout and buffer masks overlap"); hard += 1
    else:
        _ok(msgs, f"holdout ({int(ho.sum())}) and buffer ({int(bf.sum())}) masks are disjoint")

    # Is the selected epoch an isolated spike or a stable basin? The argmin of a noisy series is
    # not a property of the model, and this metric is measurably noisy while the LR is high: on
    # the first real 30-epoch run kernel_val swung 2.9x between adjacent epochs near peak LR and
    # 1.017x once the cosine took it below 4e-4. A best epoch whose neighbours are far worse was
    # selected on a lucky evaluation, and its value is biased low -- which matters because
    # configurations are ranked against each other by exactly this number.
    if summ.get("restored_best") and len(rows) >= 7:
        by_ep = {r["epoch"]: r[col] for r in rows}
        bv = by_ep.get(be)
        nb = [by_ep[e] for e in (be - 1, be + 1) if e in by_ep and np.isfinite(by_ep[e])]
        finite = [v for v in by_ep.values() if np.isfinite(v)]
        if bv is not None and np.isfinite(bv) and nb and bv > 0:
            worst_nb = max(nb) / bv
            med_all = float(np.median(finite))
            below = 1 - bv / med_all if med_all else 0
            if worst_nb >= 2.0:
                msgs.append(f"warn  best epoch {be} is an ISOLATED SPIKE: a neighbour is "
                            f"{worst_nb:.1f}x its value, and it sits {100 * below:.0f}% below "
                            f"the median of all epochs. It was selected on a lucky evaluation, "
                            f"so its value is biased low -- treat cross-configuration rankings "
                            f"built on it with suspicion. desk.selection_smooth applies a "
                            f"trailing median if this persists at full length.")
            else:
                msgs.append(f"ok    best epoch {be} sits in a stable region "
                            f"(worst neighbour {worst_nb:.2f}x its value)")
        if smooth > 1:
            msgs.append(f"note  selection used a trailing median over {smooth} epochs "
                        f"(raw value at the selected epoch: "
                        f"{summ.get('best_selection_raw', float('nan')):.6g}). A run under a "
                        f"nonzero window is NOT comparable with one under 0.")

    # Is the kernel still falling at the end? That is the §5-step-3 question: if it is, the
    # epoch budget is too SMALL, not too large, and no epoch decision should be made yet.
    if finite_k >= 10:
        k = [r["kernel_val"] for r in rows if np.isfinite(r["kernel_val"])]
        tail = k[-max(3, len(k) // 10):]
        best_k = min(k)
        best_k_ep = [r["epoch"] for r in rows if r["kernel_val"] == best_k][0]
        still = np.mean(tail) <= best_k * 1.02
        # A run whose budget is not comfortably longer than its LR warmup cannot support ANY
        # statement about where the optimum lies. _warmup_cosine ramps the LR over warmup_epochs
        # and only then anneals, so at epochs=30 against warmup=20 two thirds of the run is ramp:
        # the LR peaks near epoch 20 and the "minimum" a few epochs later is a transient of the
        # schedule, not a property of the model. Saying "the budget covers the optimum" there is
        # exactly the kind of plausible-looking claim this script exists to prevent, and the
        # smoke run produced it.
        warm = int(summ.get("warmup_epochs", 0))
        budget = int(summ.get("epochs_budget", rows[-1]["epoch"]))
        if warm and budget < 3 * warm:
            msgs.append(f"note  kernel_val best {best_k:.6g} at epoch {best_k_ep}, but this run "
                        f"is {budget} epochs against {warm} of LR warmup "
                        f"({100 * warm // max(budget, 1)}% ramp) -- NO conclusion about the "
                        f"optimum or the budget is available from it. Use a run of at least "
                        f"{3 * warm} epochs.")
        else:
            msgs.append(f"note  kernel_val best {best_k:.6g} at epoch {best_k_ep}; "
                        f"tail mean {np.mean(tail):.6g} -> "
                        + ("STILL FALLING at the end: the budget is too SMALL, do not pick an "
                           "epoch from this run" if still else "has turned over: the budget "
                           "covers the optimum"))
        z = [r["zmse_val"] for r in rows if np.isfinite(r["zmse_val"])]
        if z:
            bz = min(z)
            bz_ep = [r["epoch"] for r in rows if r["zmse_val"] == bz][0]
            gap = ("they AGREE" if bz_ep == best_k_ep else
                   f"they DISAGREE by {abs(bz_ep - best_k_ep)} epochs")
            msgs.append(f"note  val z-MSE best {bz:.6g} at epoch {bz_ep} vs kernel at "
                        f"{best_k_ep}: {gap}. A large gap is diagnostic about the MODEL, not "
                        f"about which metric to select on -- a cell-specific offset ruins "
                        f"coordinate accuracy while cancelling in every similarity comparison.")
            if bz_ep == rows[-1]["epoch"]:
                msgs.append("note  val z-MSE was still improving at the LAST epoch, so its own "
                            "optimum is outside this budget")
    return msgs, hard


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+")
    args = ap.parse_args()
    bad = 0
    for d in args.run_dir:
        print(f"=== {d}")
        if not os.path.isdir(d):
            print("  FAIL  not a directory"); bad += 1; continue
        msgs, hard = check(d)
        for m in msgs:
            print("  " + m)
        bad += hard
        print(f"  --> {'FAILED' if hard else 'OK'} ({hard} hard failure(s))")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
