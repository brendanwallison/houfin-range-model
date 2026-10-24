"""E086ab / E086z: a DERIVED research cache whose z is the sum-kernel feature map [z_summer (first S dims), z_winter
(the model's private block)], so fair_rivals can score the sum kernel with --rank S + W --desk-split S.

    python research/exp/e086ab_sum_cache.py --cache <cache with zpriv_*_all.npy> --out <new cache dir> [--summer 24]

Every file of the source cache is symlinked except z_raw_all / z_raw / z_ema_all / z_ema, which are rewritten as the
concatenation (the keys view is taken through key_cell_index and years, as the cache builder does). meta.json gains
latent_dim = S + W and e086_sum = {summer, winter}.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--summer", type=int, default=24)
    a = ap.parse_args()
    j = lambda f: os.path.join(a.cache, f)
    os.makedirs(a.out, exist_ok=True)
    rewrite = {"z_raw_all.npy", "z_raw.npy", "z_ema_all.npy", "z_ema.npy", "meta.json"}
    for f in os.listdir(a.cache):
        if f not in rewrite and not os.path.exists(os.path.join(a.out, f)):
            os.symlink(os.path.realpath(j(f)), os.path.join(a.out, f))
    ci, yrs = np.load(j("key_cell_index.npy")), np.load(j("years.npy"))
    keys = np.load(j("keys.npy")); yi = keys[:, 2] - int(yrs[0])
    W = None
    for kind in ("raw", "ema"):
        z = np.load(j(f"z_{kind}_all.npy"), mmap_mode="r"); p = np.load(j(f"zpriv_{kind}_all.npy"), mmap_mode="r")
        full = np.concatenate([np.asarray(z[..., :a.summer], np.float32), np.asarray(p, np.float32)], axis=-1)
        W = p.shape[-1]
        np.save(os.path.join(a.out, f"z_{kind}_all.npy"), full)
        np.save(os.path.join(a.out, f"z_{kind}.npy"), full[ci, yi])
    meta = json.load(open(j("meta.json")))
    meta.update({"latent_dim": a.summer + W, "e086_sum": {"summer": a.summer, "winter": int(W), "source": a.cache}})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"[sum-cache] {a.cache} -> {a.out}: z = [summer {a.summer}, winter {W}]", flush=True)


if __name__ == "__main__":
    main()
