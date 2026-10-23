"""E025: build a research cache from a DESK run's LATE-epoch state (its resume_checkpoint.pt.completed, ~epoch 490), so
backcast skill can be compared between the selected epoch (~100-160) and the end of training without retraining.

    python research/exp/epoch_variant.py --run <job dir of a DESK run> --overlay <the run's overlay> --cache-out <dir>

Copies the run's desk/ directory into the job's own directory, overwrites the model and output-EMA weights with the
checkpoint's ``model`` / ``ema`` states, writes an overlay pointing desk_output_dir there, and runs
research/exp/cache.py on it. Nothing in the source run is touched.
E055: --checkpoint takes weight snapshots (desk.snapshot_every) instead; several are averaged weight by weight.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def average_states(states):
    """Element-wise mean of state dicts (floating tensors); integer buffers are taken from the first."""
    out = {}
    for k, v in states[0].items():
        out[k] = (sum(s[k] for s in states) / len(states)).to(v.dtype) if v.is_floating_point() else v.clone()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="job directory of the DESK run (contains desk/)")
    ap.add_argument("--overlay", required=True, help="the overlay the run trained with (for paths and holdout years)")
    ap.add_argument("--cache-out", required=True)
    ap.add_argument("--work", required=True, help="this job's directory (the late-epoch model is written here)")
    ap.add_argument("--checkpoint", nargs="+", default=None,
                    help="E055: snapshot file(s) (desk/snapshots/epNNNN.pt) instead of the final checkpoint; several "
                         "are AVERAGED weight by weight (SWA-style)")
    a = ap.parse_args()
    import torch
    src = os.path.join(a.run, "desk")
    if a.checkpoint:
        cks = [torch.load(p, map_location="cpu", weights_only=False) for p in a.checkpoint]
        ck = {"epoch": int(cks[-1]["epoch"]), "best_epoch": int(cks[-1]["best_epoch"]),
              "model": average_states([c["model"] for c in cks]), "ema": average_states([c["ema"] for c in cks]),
              "averaged_epochs": [int(c["epoch"]) for c in cks]}
    else:
        ck_path = os.path.join(src, "resume_checkpoint.pt.completed")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    dst = os.path.join(a.work, "desk_late")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f.startswith("resume_checkpoint"):
            continue
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, f))
    torch.save(ck["model"], os.path.join(dst, "env_model_semisup.pth"))
    torch.save(ck["ema"], os.path.join(dst, "output_ema.pth"))
    with open(os.path.join(dst, "late_epoch.json"), "w", encoding="utf-8") as fh:
        json.dump({"source_run": a.run, "checkpoint_epoch": int(ck["epoch"]), "selected_epoch": int(ck["best_epoch"]),
                   "averaged_epochs": ck.get("averaged_epochs")}, fh, indent=1)
    ov = json.load(open(a.overlay, encoding="utf-8"))
    ov.setdefault("paths", {})["desk_output_dir"] = dst
    ov_path = os.path.join(a.work, "overlay_late.json")
    json.dump(ov, open(ov_path, "w", encoding="utf-8"), indent=1)
    env = dict(os.environ, ESK_DESK_CONFIG=ov_path)
    print(f"[epoch-variant] {a.run}: checkpoint epoch {ck['epoch']} (selected {ck['best_epoch']}) -> {dst}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(_HERE, "cache.py"), "--out", a.cache_out], env=env)
    if r.returncode == 0:
        # cache.py copies best_epoch from desk_meta.npz, which still names the SELECTED epoch: record what this is
        mp = os.path.join(a.cache_out, "meta.json")
        m = json.load(open(mp, encoding="utf-8"))
        m["epoch_variant"] = {"checkpoint_epoch": int(ck["epoch"]), "selected_epoch": int(ck["best_epoch"]),
                              "source_run": a.run, "averaged_epochs": ck.get("averaged_epochs")}
        m["best_epoch"] = None
        json.dump(m, open(mp, "w", encoding="utf-8"), indent=1)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
