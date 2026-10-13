"""E065 evaluation: can sampling-completeness covariates make raw CBC usable before 1940?

    python research/exp/e065_eval.py --data cbc_circles_v2.npz --products-csv <Audubon csv> --taxonomy <eBird taxonomy>
        --fit V0:full=PATH --fit V0:h0=PATH --fit V0:h1=PATH --fit V0:thin=PATH --fit V0:omit=PATH ... --out DIR
        [--degraded-data thin=PATH omit=PATH]

Variants V0..V3 (cbc_effort_model.py): V0 counters only; V1 + list length; V2 + benchmark completeness (both relative to
the circle's own modern mean); V3 = V2 as a detection (presence) model. Parts: full (all circles), h0/h1 (random circle
halves), thin/omit (fits to the pseudo-early copies: 1966-85 counts degraded to 1920-39 completeness, cbc_pseudo_early.py).
Changes are log(mean index, window 2) - log(mean index, window 1) per stratum x species, on strata with >= 3 fitted
circle-years in each window (in every fit compared).
  A  PSEUDO-EARLY RECOVERY (known truth): change 1966-85 -> 2005-24 in the degraded fit vs the same variant's full fit
     on the real data: species-level (mean over strata) median signed bias, median |bias|, corr; pooled stratum-level
     corr with each species' mean removed.
  B  SPLIT-HALF RELIABILITY on the real early data: change 1910-39 -> 1950-65 from the two independent circle halves,
     pooled stratum-level corr (raw and species-centred); same for 1967-86 -> 2005-24 for scale.
  C  EFFORT-ARTIFACT CHECK (real data): species-level change 1910-39 -> 1950-65 from the full fit: median, share > 0, and
     Spearman with the species' modern prevalence (share of 1966+ counts recording it). Under an effort artifact, every
     species "rises" and rare ones rise most (negative Spearman). Diagnostic only: real change could do either.
  D  MODERN GUARD vs Audubon (1967-86 -> 2005-24): species-level corr, stratum-level corr species-centred.
  E  HELD-OUT by era (full fits): median over species of Spearman (counts; NB variants) and AUC (detection; all).
     The completeness covariate uses the same list (other species), so E is inflated for V1-V3 in every era.
  F  SCENARIO REALISM: for the species and circles of each degraded copy, ratio of median non-zero count and zero share
     degraded / undegraded in 1966-85, vs real pre-1940 / post-1966 at <= 6 field counters.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def change(fit, w0, w1, v0, v1, min_support=3):
    I, sup, years = fit["index"].astype(float), fit["support"], fit["years"]
    m0, m1 = (years >= w0) & (years <= w1), (years >= v0) & (years <= v1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log(I[:, m1].mean(1)) - np.log(I[:, m0].mean(1))
    ok = (sup[:, m0].sum(1) >= min_support) & (sup[:, m1].sum(1) >= min_support)
    return d, np.broadcast_to(ok[:, None], d.shape) & np.isfinite(d)


def aligned(f_ref, f, d):
    """Re-index change array d of fit f onto the strata x species of f_ref (NaN where absent)."""
    rs = {s: i for i, s in enumerate(f["strata"])}; ss = {s: i for i, s in enumerate(f["species"])}
    out = np.full((len(f_ref["strata"]), len(f_ref["species"])), np.nan)
    ri = [rs.get(s, -1) for s in f_ref["strata"]]; si = [ss.get(s, -1) for s in f_ref["species"]]
    for a, r in enumerate(ri):
        if r >= 0:
            for b, s in enumerate(si):
                if s >= 0:
                    out[a, b] = d[r, s]
    return out


def pair_stats(x, y, ok):
    """species-level (mean over shared strata) and pooled stratum-level species-centred agreement of x with truth y."""
    x = np.where(ok, x, np.nan); y = np.where(ok, y, np.nan)
    sx, sy = np.nanmean(x, 0), np.nanmean(y, 0)
    keep = np.isfinite(sx) & np.isfinite(sy) & (ok.sum(0) >= 3)
    bias = sx[keep] - sy[keep]
    xc, yc = x - sx, y - sy
    m = ok & np.isfinite(xc) & np.isfinite(yc) & keep[None, :]
    return {"species": int(keep.sum()), "pairs": int(m.sum()),
            "species_bias_median": float(np.median(bias)), "species_absbias_median": float(np.median(np.abs(bias))),
            "species_corr": float(np.corrcoef(sx[keep], sy[keep])[0, 1]),
            "stratum_corr_centred": float(np.corrcoef(xc[m], yc[m])[0, 1])}


def auc(score, y):
    from scipy.stats import rankdata
    pos = y > 0
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 < 20 or n0 < 20:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--products-csv", required=True)
    ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--fit", action="append", required=True, help="VARIANT:PART=PATH")
    ap.add_argument("--degraded-data", nargs="*", default=[], help="MECH=PATH of the degraded prepared npz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    from scipy.stats import spearmanr
    os.makedirs(a.out, exist_ok=True)
    D = np.load(a.data, allow_pickle=False)
    Y, yr = D["Y"], D["year_end"]
    fits = defaultdict(dict)
    for s in a.fit:
        k, p = s.split("=", 1); v, part = k.split(":")
        fits[v][part] = np.load(p, allow_pickle=False)
    res = {}
    prev = (Y[yr >= 1966] > 0).mean(0)
    # Audubon
    tax = pd.read_csv(a.taxonomy, usecols=["SCIENTIFIC_NAME", "SPECIES_CODE", "CATEGORY"])
    tax = tax[tax["CATEGORY"] == "species"]
    prod = pd.read_csv(a.products_csv, encoding="latin-1")
    prod = prod[(prod["region_type"] == "stratum") & prod["survey_suitability"].isin(["medium", "high"])]
    prod["code"] = prod["scientific_name"].map(dict(zip(tax["SCIENTIFIC_NAME"], tax["SPECIES_CODE"])))
    pm = lambda y0, y1: prod[(prod.year_end >= y0) & (prod.year_end <= y1)].groupby(["code", "region"])["annual_index"].mean()
    aud = (np.log(pm(2005, 2024)) - np.log(pm(1967, 1986))).dropna()
    for v in sorted(fits):
        F = fits[v]; r = {}
        full = F.get("full")
        # A pseudo-early recovery
        for mech in ("thin", "omit"):
            if full is None or mech not in F:
                continue
            dt, okt = change(full, 1966, 1985, 2005, 2024)
            dd, okd = change(F[mech], 1966, 1985, 2005, 2024)
            dd = aligned(full, F[mech], dd); okd = aligned(full, F[mech], okd.astype(float)) == 1
            r[f"A_{mech}"] = pair_stats(dd, dt, okt & okd)
        # B split-half reliability
        if "h0" in F and "h1" in F:
            for lab, w in (("early", (1910, 1939, 1950, 1965)), ("modern", (1967, 1986, 2005, 2024))):
                d0, ok0 = change(F["h0"], *w); d1, ok1 = change(F["h1"], *w)
                m = ok0 & ok1
                c0 = d0 - np.nanmean(np.where(m, d0, np.nan), 0); c1 = d1 - np.nanmean(np.where(m, d1, np.nan), 0)
                mm = m & np.isfinite(c0) & np.isfinite(c1)
                r[f"B_{lab}"] = {"pairs": int(m.sum()),
                                 "corr_raw": float(np.corrcoef(d0[m], d1[m])[0, 1]) if m.sum() > 10 else None,
                                 "corr_centred": float(np.corrcoef(c0[mm], c1[mm])[0, 1]) if mm.sum() > 10 else None}
        if full is not None:
            # C effort-artifact check
            d, ok = change(full, 1910, 1939, 1950, 1965)
            sl = np.nanmean(np.where(ok, d, np.nan), 0)
            k = np.isfinite(sl) & (ok.sum(0) >= 3)
            r["C_early_change"] = {"species": int(k.sum()), "median": float(np.median(sl[k])),
                                   "share_positive": float((sl[k] > 0).mean()),
                                   "spearman_with_prevalence": float(spearmanr(sl[k], prev[k]).correlation)}
            # D modern guard
            d, ok = change(full, 1967, 1986, 2005, 2024)
            st, sp_ = list(full["strata"]), list(full["species"])
            rows = [(c, g, d[st.index(g), sp_.index(c)], x) for (c, g), x in aud.items()
                    if g in st and c in sp_ and ok[st.index(g), sp_.index(c)]]
            df = pd.DataFrame(rows, columns=["code", "region", "ours", "aud"])
            dc = df.copy(); dc[["ours", "aud"]] = dc.groupby("code")[["ours", "aud"]].transform(lambda z: z - z.mean())
            sm = df.groupby("code")[["ours", "aud"]].mean()
            r["D_audubon"] = {"pairs": len(df), "species_corr": float(np.corrcoef(sm.ours, sm.aud)[0, 1]),
                              "stratum_corr_centred": float(np.corrcoef(dc.ours, dc.aud)[0, 1])}
            # E held-out by era
            rows_h = full["hold_rows"]
            fam = str(full["family"]) if "family" in full.files else "nb"
            for era, (y0, y1) in (("pre1940", (1901, 1939)), ("1940_1965", (1940, 1965)), ("post1966", (1966, 2025))):
                m = (yr[rows_h] >= y0) & (yr[rows_h] <= y1)
                sp_v, au_v = [], []
                for s in range(Y.shape[1]):
                    obs = Y[rows_h[m], s]
                    if fam == "nb" and (obs > 0).sum() >= 20:
                        sp_v.append(spearmanr(full["hold_pred"][m, s], obs).correlation)
                    au_v.append(auc(full["hold_pdet"][m, s], obs))
                r[f"E_{era}"] = {"spearman_median": float(np.nanmedian(sp_v)) if sp_v else None,
                                 "auc_median": float(np.nanmedian(au_v)), "auc_species": int(np.isfinite(au_v).sum())}
            r["k_median"] = float(np.median(full["k"])) if "k" in full.files else None
        res[v] = r
    # F scenario realism
    fc = D["field_counters"]
    early = (yr < 1940) & (fc <= 6); late = (yr >= 1966) & (fc <= 6)
    def nz_med(Yx, m):
        v = Yx[m]; return float(np.median(v[v > 0])), float((v == 0).mean())
    e_nz, e_z = nz_med(Y, early); l_nz, l_z = nz_med(Y, late)
    res["F_real"] = {"pre1940_vs_post1966_le6_counters_nonzero_ratio": e_nz / l_nz, "zero_share_pre1940": e_z,
                     "zero_share_post1966_le6": l_z}
    w = (yr >= 1966) & (yr <= 1985)
    u_nz, u_z = nz_med(Y, w)
    for s in a.degraded_data:
        mech, p = s.split("=", 1)
        Dd = np.load(p, allow_pickle=False)
        sp_i = [list(Dd["species"]).index(x) for x in D["species"]]
        wd = (Dd["year_end"] >= 1966) & (Dd["year_end"] <= 1985)
        d_nz, d_z = nz_med(Dd["Y"][:, sp_i], wd)
        res[f"F_{mech}"] = {"nonzero_median_ratio": d_nz / u_nz, "zero_share_degraded": d_z, "zero_share_undegraded": u_z}
    json.dump(res, open(os.path.join(a.out, "e065.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
