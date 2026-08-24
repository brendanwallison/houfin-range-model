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

    if summ["restored_best"]:
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

    # Is the kernel still falling at the end? That is the §5-step-3 question: if it is, the
    # epoch budget is too SMALL, not too large, and no epoch decision should be made yet.
    if finite_k >= 10:
        k = [r["kernel_val"] for r in rows if np.isfinite(r["kernel_val"])]
        tail = k[-max(3, len(k) // 10):]
        best_k = min(k)
        still = np.mean(tail) <= best_k * 1.02
        msgs.append(f"note  kernel_val best {best_k:.6g} at epoch "
                    f"{[r['epoch'] for r in rows if r['kernel_val'] == best_k][0]}; "
                    f"tail mean {np.mean(tail):.6g} -> "
                    + ("STILL FALLING at the end: the budget is too SMALL, do not pick an "
                       "epoch from this run" if still else "has turned over: the budget covers "
                       "the optimum"))
        z = [r["zmse_val"] for r in rows if np.isfinite(r["zmse_val"])]
        if z:
            bz = min(z)
            msgs.append(f"note  val z-MSE best {bz:.6g} at epoch "
                        f"{[r['epoch'] for r in rows if r['zmse_val'] == bz][0]} -- if that "
                        f"differs from the kernel's best epoch, THAT is the finding")
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
