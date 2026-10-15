"""E073 B2 skeptic, post-hoc on saved predictions (no fitting): species jackknife of the headline difference,
a magnitude-sensitive noise-free MSE skill, per-species win share, and held-out split overlap between the two caches.

    python research/lib/launch.py research/skeptic/E073/post.py
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "4"
import json  # noqa: E402

import numpy as np  # noqa: E402

REPO = "/mnt/c/Dev/houfin-range-model"
for p in (REPO, REPO + "/src", REPO + "/research", REPO + "/research/exp"):
    if p not in sys.path:
        sys.path.insert(0, p)
import anchored_delta as ad  # noqa: E402

SIM = "/home/breallis/houfin/work/houfin/research/E067/composition-d2e6a98f0eb7/species_similarity.csv"
C = "/home/breallis/houfin/work/houfin/research_cache"
O = "/home/breallis/houfin/work/houfin/research/E073"
S = "/home/breallis/houfin/scratch/e073_skeptic"
RUNS = {"t1985": ("desk_tempho_1985_s1", "B2-t1985-s1-88539656cd2a"),
        "t1995": ("desk_tempho_1995_s1", "B2-t1995-s1-906e8121d59f")}
ARMS = ("desk_split_level", "desk_within", "vc_desk4_no_clock_h25", "vc_cov4_no_clock_h25", "vc_desk4_no_clock_h10",
        "within_no_clock_h25", "placebo")


def pooled(P, Ta, Tb, w=None):
    w = np.ones(len(P)) if w is None else w
    sw = w.sum()
    mu = lambda X: (w[:, None] * X).sum(0) / sw
    Pc, Ac, Bc = P - mu(P), Ta - mu(Ta), Tb - mu(Tb)
    vp = (w[:, None] * Pc * Pc).sum()
    vt = (w[:, None] * Ac * Bc).sum()
    cv = 0.5 * ((w[:, None] * Pc * Ac).sum() + (w[:, None] * Pc * Bc).sum())
    # noise-free MSE skill vs "no change beyond the mean": 1 - E[(P - T)^2] / var(T), with E[(P-T)^2] estimated by the
    # cross-product of the two independent halves' residuals (noise cancels in expectation)
    mse = (w[:, None] * (Pc - Ac) * (Pc - Bc)).sum()
    return cv / np.sqrt(vp * vt), 1 - mse / vt


def main():
    out = {}
    hold = {}
    for tag, (cache, b2) in RUNS.items():
        est = ad.Estimand(os.path.join(C, cache), 200, SIM)
        hold[tag] = set(map(tuple, est.cells_ev.tolist()))
        sv = np.load(os.path.join(O, b2, "preds_b2_dev.npz"))
        P = {k: np.asarray(sv[k], "float64") for k in ARMS}
        Ta, Tb = est.T["dev"]
        o = {}
        for sname in ("sedentary", "all"):
            m = est.strata[sname]
            r = {"corr_and_mse_skill": {}, "mse_skill_diffs": {}}
            for k, v in P.items():
                c, sk = pooled(v[:, m], Ta[:, m], Tb[:, m])
                r["corr_and_mse_skill"][k] = {"corr": float(c), "mse_skill": float(sk)}
            for x in ("vc_desk4_no_clock_h25", "vc_cov4_no_clock_h25"):
                d = np.array([pooled(P[x][:, m], Ta[:, m], Tb[:, m], w)[1]
                              - pooled(P["desk_split_level"][:, m], Ta[:, m], Tb[:, m], w)[1] for w in est.Wb])
                r["mse_skill_diffs"][f"{x} - desk_split_level"] = {"diff": float(d[0]), "ci": ad.ci(d)}
            # species jackknife of the corr difference
            idx = np.where(m)[0]
            jk = []
            for j in idx:
                mm = m.copy()
                mm[j] = False
                jk.append(pooled(P["vc_desk4_no_clock_h25"][:, mm], Ta[:, mm], Tb[:, mm])[0]
                          - pooled(P["desk_split_level"][:, mm], Ta[:, mm], Tb[:, mm])[0])
            jk = np.array(jk)
            full = (pooled(P["vc_desk4_no_clock_h25"][:, m], Ta[:, m], Tb[:, m])[0]
                    - pooled(P["desk_split_level"][:, m], Ta[:, m], Tb[:, m])[0])
            top = idx[np.argsort(jk)[:3]]
            r["jackknife_vc_desk4_minus_desk_split"] = {
                "full": float(full), "min": float(jk.min()), "max": float(jk.max()),
                "most_influential_species": [est.species[j] for j in top]}
            # drop the 3 most influential species jointly
            mm = m.copy()
            mm[top] = False
            r["jackknife_vc_desk4_minus_desk_split"]["drop_top3"] = float(
                pooled(P["vc_desk4_no_clock_h25"][:, mm], Ta[:, mm], Tb[:, mm])[0]
                - pooled(P["desk_split_level"][:, mm], Ta[:, mm], Tb[:, mm])[0])
            # per-species noise-free corr: share of species where vc beats DESK
            def per_sp(Pm):
                Pc, Ac, Bc = Pm - Pm.mean(0), Ta - Ta.mean(0), Tb - Tb.mean(0)
                return 0.5 * ((Pc * Ac).sum(0) + (Pc * Bc).sum(0)) / np.sqrt((Pc * Pc).sum(0) * np.abs((Ac * Bc).sum(0)) + 1e-30)
            a_, b_ = per_sp(P["vc_desk4_no_clock_h25"])[m], per_sp(P["desk_split_level"])[m]
            r["per_species_share_vc_beats_desk"] = float((a_ > b_).mean())
            r["per_species_median_diff"] = float(np.median(a_ - b_))
            o[sname] = r
        out[tag] = o
        print(tag, json.dumps(o, indent=1), flush=True)
    a, b = hold["t1985"], hold["t1995"]
    out["heldout_cells"] = {"t1985": len(a), "t1995": len(b), "shared": len(a & b)}
    print(out["heldout_cells"])
    res_path = os.path.join(REPO, "research/skeptic/E073/check_results.json")
    allres = json.load(open(res_path)) if os.path.exists(res_path) else {}
    allres["post"] = out
    json.dump(allres, open(res_path, "w"), indent=1)


if __name__ == "__main__":
    main()
