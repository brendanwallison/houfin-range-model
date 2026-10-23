"""Paired comparison of two fast-arm runs on the same rows and species (both run with --dump-change).

    python research/exp/paired_arms.py <arm A dir> <arm B dir> [--sets time space_time] [--n-boot 1000]

Per change set: the median over resolvable species of skill_A - skill_B (skill = 1 - sqrt(SSE / SSE_no_change),
per species), pooled and in the common prevalence tiers, total and place-specific (each species' mean change over
the cells removed from truth and both predictions). The 95% CI resamples 6x6-cell blocks, recomputing every
species' SSEs from the resampled cells, so A and B are compared on the SAME draws -- the paired test the two arms'
separate CIs cannot give. Prevalence comes from A's per_species.csv (training-row detection share).
"""
import argparse
import json
import os

import numpy as np

TIERS = (("all", 0.0, 1.01), ("0.1-0.3", 0.1, 0.3), ("0.3-1.01", 0.3, 1.01))


def load(arm, sname):
    d = np.load(os.path.join(arm, f"change_{sname}.npz"))
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--sets", nargs="+", default=("time", "space_time", "space"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import pandas as pd
    prev = pd.read_csv(os.path.join(a.a, "per_species.csv"))["prevalence"].to_numpy()
    res = {"a": a.a, "b": a.b, "sets": {}}
    rng = np.random.default_rng(0)
    for s in a.sets:
        if not (os.path.exists(os.path.join(a.a, f"change_{s}.npz"))
                and os.path.exists(os.path.join(a.b, f"change_{s}.npz"))):
            continue
        A, B = load(a.a, s), load(a.b, s)
        assert np.array_equal(A["cells"], B["cells"]) and np.array_equal(A["species"], B["species"]), \
            "the two runs differ in cells or species"
        assert np.array_equal(A["d_full"], B["d_full"]), "the two runs grade different truths"
        cells, d, res_mask = A["cells"], A["d_full"], A["resolvable"].astype(bool)
        blocks = (cells[:, 0] // 6) * 100000 + cells[:, 1] // 6
        ub, inv = np.unique(blocks, return_inverse=True)
        out = {}
        for kind in ("total", "place"):
            dm = (lambda v: v - v.mean(0, keepdims=True)) if kind == "place" else (lambda v: v)
            dd, pa, pb, pn = dm(d), dm(A["dp_model"]), dm(B["dp_model"]), dm(A["dp_no_change"])
            # per-block SSE tables (blocks x species)
            bs = lambda e: np.stack([np.bincount(inv, weights=e[:, j], minlength=len(ub))
                                     for j in range(e.shape[1])], 1)
            SA, SB, SN = bs((pa - dd) ** 2), bs((pb - dd) ** 2), bs((pn - dd) ** 2)

            def diff(w):
                sa, sb, sn = w @ SA, w @ SB, w @ SN
                with np.errstate(invalid="ignore", divide="ignore"):
                    ka, kb = 1 - np.sqrt(sa / sn), 1 - np.sqrt(sb / sn)
                return ka - kb
            point = diff(np.ones(len(ub)))
            W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)
            boots = np.stack([diff(w) for w in W])
            for name, lo, hi in TIERS:
                m = res_mask & (prev >= lo) & (prev < hi) & np.isfinite(point)
                if m.sum() < 3:
                    continue
                bm = np.array([np.nanmedian(b[m]) for b in boots])
                bm = bm[np.isfinite(bm)]
                out[f"{kind}/{name}"] = {"n": int(m.sum()), "median_diff": float(np.median(point[m])),
                                         "ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                         "share_a_better": float((point[m] > 0).mean())}
        res["sets"][s] = out
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    print(f"paired: A = {a.a}\n        B = {a.b}\n  median over species of skill(A) - skill(B) [95% block CI]")
    for s, out in res["sets"].items():
        for k, v in out.items():
            print(f"  {s:10s} {k:16s} n={v['n']:3d}  {v['median_diff']:+.4f} [{v['ci'][0]:+.4f}, "
                  f"{v['ci'][1]:+.4f}]  A better for {v['share_a_better']:.0%}")


if __name__ == "__main__":
    main()
