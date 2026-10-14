"""E068: why do real early CBC counts show most species DECLINING from 1910-39 to 1950-65, rare species most (E065 C)?

    python research/exp/e068_early_decline.py --data cbc_circles_v2.npz --completeness completeness_v3.npz
        --fit NAME=PATH ... --out DIR

(a) EFFORT EXTRAPOLATION. Early counts have ~2 field counters (modern ~18), so each species' effort exponent p[s] is
    extrapolated far below the data's centre; if rare species have larger p, their early indices are inflated more.
    Per fit: Spearman of the species-level change 1910-39 -> 1950-65 with p[s] and with modern prevalence, and each
    partial on the other (rank residuals). Fits with --p-fixed (one exponent for all species) show what is left.
(b) SELECTIVE REPORTING of notable species. In circles with counts both before 1940 and from 1966, per list-length bin
    (species recorded: 5-14, 15-24, 25-39), each species' inclusion rate on early vs modern lists; modern rates are
    weighted by each circle's number of early lists in the bin (same circles, same mix). Spearman of log(early/modern)
    with modern prevalence, per bin. Notable-first reporting predicts rare species over-included, MOST on short lists;
    a real decline of rare species predicts the same in every bin.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e065_eval import change  # noqa: E402


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    return float(np.corrcoef(res(rx, rz), res(ry, rz))[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--completeness", required=True)
    ap.add_argument("--fit", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = np.load(a.data, allow_pickle=False)
    Y, yr, ci = D["Y"], D["year_end"], D["circle_idx"]
    prev = (Y[yr >= 1966] > 0).mean(0)
    res = {"a": {}, "b": {}}
    for s in a.fit:
        name, p = s.split("=", 1)
        f = np.load(p, allow_pickle=False)
        d, ok = change(f, 1910, 1939, 1950, 1965)
        sl = np.nanmean(np.where(ok, d, np.nan), 0)
        k = np.isfinite(sl) & (ok.sum(0) >= 3)
        pv = f["p"]
        r = {"species": int(k.sum()), "median_change": float(np.median(sl[k])),
             "share_positive": float((sl[k] > 0).mean()),
             "spearman_change_prevalence": float(spearmanr(sl[k], prev[k]).correlation),
             "p_median": float(np.median(pv))}
        if np.std(pv) > 1e-6:
            r["spearman_change_p"] = float(spearmanr(sl[k], pv[k]).correlation)
            r["spearman_p_prevalence"] = float(spearmanr(pv[k], prev[k]).correlation)
            r["partial_change_p_given_prevalence"] = partial_spearman(sl[k], pv[k], prev[k])
            r["partial_change_prevalence_given_p"] = partial_spearman(sl[k], prev[k], pv[k])
        res["a"][name] = r
    # (b)
    L = np.load(a.completeness, allow_pickle=False)["list_len"]
    early, modern = yr < 1940, yr >= 1966
    both = np.intersect1d(np.unique(ci[early]), np.unique(ci[modern]))
    inb = np.isin(ci, both)
    for lo, hi in ((5, 15), (15, 25), (25, 40)):
        m = (L >= lo) & (L < hi) & inb
        e_rows, m_rows = np.where(m & early)[0], np.where(m & modern)[0]
        ce = np.intersect1d(np.unique(ci[e_rows]), np.unique(ci[m_rows]))
        if len(ce) < 10:
            res["b"][f"{lo}-{hi - 1}"] = {"circles": int(len(ce))}
            continue
        num_e = np.zeros(Y.shape[1]); num_m = np.zeros(Y.shape[1]); w_tot = 0
        n_m_rows = 0
        for c in ce:
            re_, rm_ = e_rows[ci[e_rows] == c], m_rows[ci[m_rows] == c]
            w = len(re_)
            num_e += (Y[re_] > 0).sum(0)
            num_m += w * (Y[rm_] > 0).mean(0)
            w_tot += w; n_m_rows += len(rm_)
        rate_e, rate_m = num_e / w_tot, num_m / w_tot
        keep = (num_e + num_m) >= 10
        lr = np.log((rate_e + 0.5 / w_tot) / (rate_m + 0.5 / w_tot))
        res["b"][f"{lo}-{hi - 1}"] = {
            "circles": int(len(ce)), "early_lists": int(w_tot), "modern_lists": int(n_m_rows), "species": int(keep.sum()),
            "spearman_logratio_prevalence": float(spearmanr(lr[keep], prev[keep]).correlation),
            "median_logratio": float(np.median(lr[keep]))}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e068.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
