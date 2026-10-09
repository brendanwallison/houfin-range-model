"""The ONLY reader of the sealed confirmation species (research/lib/seal.py). Every read is logged.

    python research/adopt.py --cache <cache dir> --reason "E0NN: confirm <mechanism> before adoption"

The loop reads sealed species exactly once per adoption checkpoint (PROTOCOL.md "Guards"), never to
explore. This script does nothing clever: it appends a line to research/sealed_reads.csv and returns
the sealed matrix to the caller (import ``load_sealed``) so the confirmation analysis is the same
code that ran on the dev species, pointed at the other half.
"""
import argparse
import csv
import datetime as _dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import paths  # noqa: E402

LOG = paths.RESEARCH / "sealed_reads.csv"


def load_sealed(cache_dir, reason):
    if not reason or len(reason) < 12:
        raise SystemExit("a sealed read needs a reason naming the experiment and the claim")
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new:
            w.writerow(["ts", "cache", "reason"])
        w.writerow([_dt.datetime.now().astimezone().isoformat(timespec="seconds"), cache_dir,
                    reason])
    return np.load(os.path.join(cache_dir, "sealed", "X_sealed.npy"), mmap_mode="r")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--reason", required=True)
    a = ap.parse_args()
    X = load_sealed(a.cache, a.reason)
    print(f"sealed matrix {X.shape}; read logged to {LOG}")
