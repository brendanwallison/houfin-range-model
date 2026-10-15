"""E075: did E069b's presence term improve held-out early CBC lists MORE where land cover changed most?

    python research/exp/e075_cover_strata.py --lists-dir DIR --labels s1_pres01_late s2_pres01_late ...
        --cbc cbc_circles_comm.npz --cover cover_change_1920_1966.npz --out DIR

Per-list gain = arm (backcast - modern) - control (backcast - modern) (e069b_heldout_lists.py --save-lists; held-out
cells, resident species). Strata: cover-change TOP tercile vs the rest (cover_change_1920_1966.npz, terciles over all
valid cells); also Great Plains (zone 2) vs the rest. Era: 1953-1995 pooled (primary), and 1953-65 / 1966-95. Mean gain
per stratum and the difference top - rest, 95% bootstrap over circles (circles resampled within each stratum).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lists-dir", required=True); ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--cbc", required=True); ap.add_argument("--cover", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default="residents"); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    D = np.load(a.cbc, allow_pickle=False)
    M = np.load(a.cover)
    ch, zo, t = M["change"], M["zones"], M["terciles"]
    res = {}
    for lab in a.labels:
        L = np.load(os.path.join(a.lists_dir, f"lists_{lab}_{a.subset}.npz"))
        rows, gain = [], []
        for era in ("1953_1965", "1966_1995"):
            r = L[f"{era}_rows"]; assert (r == L[f"{era}_rows_c"]).all()
            rows.append(r); gain.append((L[f"{era}_lba"] - L[f"{era}_lma"]) - (L[f"{era}_lbc"] - L[f"{era}_lmc"]))
        out = {}
        for era_name, sel in (("1953_1995", slice(None)), ("1953_1965", 0), ("1966_1995", 1)):
            r = np.concatenate(rows) if sel == slice(None) else rows[sel]
            g = np.concatenate(gain) if sel == slice(None) else gain[sel]
            rr, cc = D["row"][r], D["col"][r]
            cov = ch[rr, cc]; z = zo[rr, cc]; circ = D["circle_idx"][r]
            o = {}
            for sname, top in (("cover_top_tercile", cov > t[1]), ("great_plains", z == 2)):
                ok = np.isfinite(cov) if sname.startswith("cover") else np.ones(len(r), bool)
                A, B = top & ok, ~top & ok
                if len(np.unique(circ[A])) < 3 or len(np.unique(circ[B])) < 3:
                    o[sname] = {"lists_top": int(A.sum()), "circles_top": int(len(np.unique(circ[A]))), "note": "too few"}
                    continue
                rng = np.random.default_rng(75)

                def boot(mask):
                    uc = np.unique(circ[mask]); idx = {c: np.where(mask & (circ == c))[0] for c in uc}
                    return np.array([g[np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])].mean()
                                     for _ in range(a.boot)])
                ba, bb = boot(A), boot(B)
                o[sname] = {"lists_top": int(A.sum()), "circles_top": int(len(np.unique(circ[A]))),
                            "lists_rest": int(B.sum()), "circles_rest": int(len(np.unique(circ[B]))),
                            "gain_top": float(g[A].mean()), "gain_rest": float(g[B].mean()),
                            "diff_top_minus_rest": float(g[A].mean() - g[B].mean()),
                            "diff_ci95": [float(x) for x in np.percentile(ba - bb, [2.5, 97.5])]}
            out[era_name] = o
        res[lab] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e075.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
