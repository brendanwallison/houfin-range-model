"""Check treatment caches match a control on keys/cells/years/X/split (never opens sealed/); copy the esk files.

    python research/exp/verify_cache.py <control cache> <cache> [<cache> ...]   (exits non-zero on any mismatch)
"""
import os
import shutil
import sys

import numpy as np

ctrl, *treats = sys.argv[1:]
for t in treats:
    bad = []
    for f in ("keys.npy", "cells.npy", "years.npy", "key_cell_index.npy", "X_comm.npy", "X_dev.npy"):
        a, b = np.load(os.path.join(ctrl, f)), np.load(os.path.join(t, f))
        if a.shape != b.shape or not np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)):
            bad.append(f)
    sa, sb = np.load(os.path.join(ctrl, "split.npz")), np.load(os.path.join(t, "split.npz"))
    for k in sa.files:
        if k not in sb.files or not np.array_equal(sa[k], sb[k]):
            bad.append("split:" + k)
    if bad:
        print(t, "MISMATCH", bad)
        sys.exit(1)
    for f in ("esk_annual.npy", "esk_epochs.npz"):
        dst = os.path.join(t, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(ctrl, f), dst)
    print(t, "OK: keys/cells/years/X/split identical; esk files present;",
          "F_cov", np.load(os.path.join(t, "F_cov.npy"), mmap_mode="r").shape, "vs control",
          np.load(os.path.join(ctrl, "F_cov.npy"), mmap_mode="r").shape)
