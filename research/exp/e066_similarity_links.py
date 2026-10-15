"""E066: can early CBC supervise DESK's Ruzicka space through an observation model? Two preconditions, no training.

    python research/exp/e066_similarity_links.py t1 --cache <research cache> --cbc cbc_circles_comm.npz --out DIR
    python research/exp/e066_similarity_links.py t2 --real cbc_circles_comm.npz --degraded MECH=DIR ... --out DIR

Similarity throughout is DESK's: Ruzicka on log1p counts over the reference (community) species that both sources carry.
T1 SEASON LINK (modern data). Cell-years with both BBS (route mean, as in the cache) and CBC (mean over the cell's
circles; counts standardized to the median party-hours with exponent 0.8, E064's median -- raw also reported), winters
ending 1966-2024 matched to the BBS summer of the same year. Pairs drawn three ways: SPATIAL (two cells, one year),
TEMPORAL (one cell, years >= 10 apart), SPACETIME (two cells, years >= 5 apart). Each pair is also taken one year later
(both points +1 year), so every similarity has an independent replicate. Reported per pair type: Spearman of BBS with
CBC similarity, ABBA form (BBS at t vs CBC at t+1, and CBC at t vs BBS at t+1, averaged), and disattenuated by the
square root of the BBS-BBS and CBC-CBC replicate Spearmans (approximate for rank correlations).
T2 ORDER UNDER DEGRADATION (pseudo-early copies, cbc_pseudo_early.py). True similarity from the real counts, observed
from the copy where 1966-1985 counts were degraded to 1920-39 completeness. Pairs: EARLY-EARLY (two degraded counts,
different circles), EARLY-MODERN (a degraded count with a 2005-2024 count of another circle), SAME-CIRCLE CROSS-ERA (a
degraded count with the same circle's 2005-2024 count: the temporal signal). Completeness of a degraded count = its
benchmark completeness relative to its circle's mean over 1986+ counts (M30), in tertile bins. Reported: Spearman of
observed with true similarity, pooled and within completeness bins (the bin pair for early-early), and the median shift
observed - true per bin (what an ordinal loss would discard and a calibrated forward model would have to supply).
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr


def ruzicka(a, b):
    a, b = np.log1p(a), np.log1p(b)
    den = np.maximum(a, b).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, np.minimum(a, b).sum(1) / den, np.nan)


def rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[m], y[m]).correlation) if m.sum() > 50 else float("nan")


def t1(a):
    import pandas as pd
    C = a.cache
    meta = json.load(open(os.path.join(C, "meta.json")))
    keys = np.load(os.path.join(C, "keys.npy")); Xb = np.load(os.path.join(C, "X_comm.npy"))
    D = np.load(a.cbc, allow_pickle=False)
    comm = list(meta["community"]); cs = list(D["species"])
    sp = [s for s in comm if s in cs]
    if a.migration:
        # E066c: restrict to AVONET migration classes (1 sedentary, 2 partial, 3 migratory), the classification of
        # species_similarity.csv used by E059g/E067
        import pandas as pd
        sim = pd.read_csv(a.similarity)
        mig = dict(zip(sim["species_code"], sim["migration"]))
        want = {float(x) for x in a.migration.split(",")}
        sp = [s for s in sp if mig.get(s) in want]
    assert "houfin" not in sp
    Xb = Xb[:, [comm.index(s) for s in sp]]
    Yc = D["Y"][:, [cs.index(s) for s in sp]].astype(float)
    yr, ph = D["year_end"], D["party_hours"]
    ok = (yr >= 1966) & (yr <= 2024) & np.isfinite(ph) & (ph > 0) & (D["row"] >= 0)
    med = np.median(ph[ok])
    res = {"species": len(sp)}
    for std in ("std", "raw"):
        Ys = Yc * ((med / ph)[:, None] ** 0.8) if std == "std" else Yc
        df = pd.DataFrame(Ys[ok]); df["r"], df["c"], df["y"] = D["row"][ok], D["col"][ok], yr[ok]
        cbc = df.groupby(["r", "c", "y"]).mean()
        bbs = pd.DataFrame(Xb); bbs["r"], bbs["c"], bbs["y"] = keys[:, 0], keys[:, 1], keys[:, 2]
        bbs = bbs.groupby(["r", "c", "y"]).mean()
        both = cbc.index.intersection(bbs.index)
        if a.cell_mask:
            # E079: restrict to cells in a stratum (an (H, W) bool array in an npz, key --cell-mask-key)
            cm = np.load(a.cell_mask)[a.cell_mask_key]
            both = both[[bool(cm[int(r_), int(c_)]) for r_, c_, _ in both]]
        cbc, bbs = cbc.loc[both], bbs.loc[both]
        cell = {(r, c) for r, c, _ in both}
        have = set(both)
        # (cell, year) usable as a base point: present at y and y+1 in both seasons
        base = [(r, c, y) for r, c, y in both if (r, c, y + 1) in have]
        rng = np.random.default_rng(66)
        by_year, by_cell = {}, {}
        for r, c, y in base:
            by_year.setdefault(y, []).append((r, c)); by_cell.setdefault((r, c), []).append(y)
        Cv = {k: v for k, v in zip(cbc.index, cbc.to_numpy())}; Bv = {k: v for k, v in zip(bbs.index, bbs.to_numpy())}
        years_l, cells_l = list(by_year), list(by_cell)
        out = {"cell_years_both": int(len(both)), "cells_both": len(cell), "base_points": len(base)}
        for kind in ("spatial", "temporal", "spacetime"):
            P = []
            for _ in range(a.pairs * 3):
                if len(P) >= a.pairs:
                    break
                if kind == "spatial":
                    y = years_l[rng.integers(len(years_l))]; L = by_year[y]
                    if len(L) < 2:
                        continue
                    i, j = rng.choice(len(L), 2, replace=False); P.append(((*L[i], y), (*L[j], y)))
                elif kind == "temporal":
                    k = cells_l[rng.integers(len(cells_l))]; ys = by_cell[k]
                    y1, y2 = rng.choice(ys), rng.choice(ys)
                    if abs(y1 - y2) >= 10:
                        P.append(((*k, y1), (*k, y2)))
                else:
                    p1, p2 = base[rng.integers(len(base))], base[rng.integers(len(base))]
                    if p1[:2] != p2[:2] and abs(p1[2] - p2[2]) >= 5:
                        P.append((p1, p2))
            sh = lambda p: (p[0], p[1], p[2] + 1)
            g = lambda V, ps: np.array([V[p] for p in ps])
            A0, A1 = [p for p, _ in P], [q for _, q in P]
            B0 = ruzicka(g(Bv, A0), g(Bv, A1)); B1 = ruzicka(g(Bv, map(sh, A0)), g(Bv, map(sh, A1)))
            C0 = ruzicka(g(Cv, A0), g(Cv, A1)); C1 = ruzicka(g(Cv, map(sh, A0)), g(Cv, map(sh, A1)))
            rbb, rcc = rho(B0, B1), rho(C0, C1)
            cross = (rho(B0, C1) + rho(C0, B1)) / 2
            out[kind] = {"pairs": len(P), "bbs_bbs": rbb, "cbc_cbc": rcc, "same_time": rho(B0, C0), "abba": cross,
                         "disattenuated": cross / np.sqrt(rbb * rcc) if rbb > 0 and rcc > 0 else None}
        res[std] = out
    return res


def t2(a):
    R = np.load(a.real, allow_pickle=False)
    res = {}
    rkey = {(c, y): i for i, (c, y) in enumerate(zip(R["circles"][R["circle_idx"]], R["year_end"]))}
    for s in a.degraded:
        mech, d = s.split("=", 1)
        Dg = np.load(os.path.join(d, "cbc_circles_comm.npz"), allow_pickle=False)
        Cf = np.load(os.path.join(d, "completeness.npz"), allow_pickle=False)
        D73 = np.load(os.path.join(d, "cbc_circles.npz"), allow_pickle=False)
        assert (D73["circle_idx"] == Dg["circle_idx"]).all() and (D73["year_end"] == Dg["year_end"]).all(), "row order"
        sp = [x for x in R["species"] if x in set(Dg["species"])]
        ri, di = [list(R["species"]).index(x) for x in sp], [list(Dg["species"]).index(x) for x in sp]
        circ, yr = Dg["circles"][Dg["circle_idx"]], Dg["year_end"]
        j = np.array([rkey.get((c, y), -1) for c, y in zip(circ, yr)])
        Yt = np.where(j[:, None] >= 0, R["Y"][np.maximum(j, 0)][:, ri], np.nan)
        Yo = Dg["Y"][:, di].astype(float)
        comp = np.where(Cf["bench_n"] > 0, Cf["bench_hit"] / np.maximum(Cf["bench_n"], 1), np.nan)
        ci = Dg["circle_idx"]
        late = (yr >= 1986) & np.isfinite(comp)
        ref = np.bincount(ci[late], comp[late], minlength=len(Dg["circles"])) / np.maximum(
            np.bincount(ci[late], minlength=len(Dg["circles"])), 1)
        rel = np.where(ref[ci] > 0, comp / np.where(ref[ci] > 0, ref[ci], 1), np.nan)
        E = np.where((yr >= 1966) & (yr <= 1985) & (j >= 0) & np.isfinite(rel))[0]
        M = np.where((yr >= 2005) & (yr <= 2024) & (j >= 0))[0]
        q = np.quantile(rel[E], [1 / 3, 2 / 3])
        b = np.digitize(rel, q)                                  # 0 low, 1 mid, 2 high completeness
        rng = np.random.default_rng(66)
        n = a.pairs
        out = {"species": len(sp), "early_rows": int(len(E)), "modern_rows": int(len(M)),
               "rel_completeness_tertiles": [float(x) for x in q], "rel_completeness_median": float(np.median(rel[E]))}
        if a.agg:
            out["agg"] = {str(k): agg_units(k, E, M, ci, Yt, Yo, rel, q, rng) for k in a.agg}
            res[mech] = out
            continue
        mbyc = {}
        for i in M:
            mbyc.setdefault(ci[i], []).append(i)
        Ec = np.array([i for i in E if ci[i] in mbyc])
        designs = {
            "early_early": (rng.choice(E, n), rng.choice(E, n)),
            "early_modern": (rng.choice(E, n), rng.choice(M, n)),
            "same_circle_cross_era": None,
        }
        e = rng.choice(Ec, n)
        designs["same_circle_cross_era"] = (e, np.array([rng.choice(mbyc[ci[i]]) for i in e]))
        for kind, (I, J) in designs.items():
            if kind != "same_circle_cross_era":
                keep = ci[I] != ci[J]; I, J = I[keep], J[keep]
            st = ruzicka(Yt[I], Yt[J]); so = ruzicka(Yo[I], Yo[J])
            key = np.minimum(b[I], b[J]) * 3 + np.maximum(b[I], b[J]) if kind == "early_early" else b[I]
            within, w, shift = [], [], {}
            for k in np.unique(key):
                m = key == k
                if m.sum() >= 200:
                    within.append(rho(so[m], st[m])); w.append(m.sum())
                    shift[int(k)] = float(np.nanmedian(so[m] - st[m]))
            out[kind] = {"pairs": int(len(I)), "pooled": rho(so, st),
                         "within_bins": float(np.average(within, weights=w)) if within else None,
                         "within_bins_min": float(np.min(within)) if within else None,
                         "median_shift_by_bin": shift, "true_similarity_median": float(np.nanmedian(st))}
        res[mech] = out
    if a.agg:
        # what real early data offers: counts per circle per decade before 1940
        yr_r, ci_r = R["year_end"], R["circle_idx"]
        m = yr_r < 1940
        n_cd = np.unique(np.column_stack([ci_r[m], yr_r[m] // 10]), axis=0, return_counts=True)[1]
        res["real_pre1940_counts_per_circle_decade"] = {
            "circle_decades": int(len(n_cd)), "median": float(np.median(n_cd)),
            "share_ge3": float((n_cd >= 3).mean()), "share_ge5": float((n_cd >= 5).mean()),
            "share_of_counts_in_circle_decades_ge5": float(n_cd[n_cd >= 5].sum() / n_cd.sum())}
    return res


def agg_units(k, E, M, ci, Yt, Yo, rel, q, rng):
    """E066b: one unit per circle and era = the mean of k random counts of that circle in the era (circles with >= k
    counts there); pairs among units as in T2; bins on the unit's mean relative completeness."""
    def units(rows, degrade):
        by = {}
        for i in rows:
            by.setdefault(ci[i], []).append(i)
        U = {}
        for c, L in by.items():
            if len(L) >= k:
                s = rng.choice(L, k, replace=False)
                U[c] = (Yt[s].mean(0), (Yo if degrade else Yt)[s].mean(0), float(np.mean(rel[s])) if degrade else 1.0)
        return U
    Ue, Um = units(E, True), units(M, False)
    ce, cm = list(Ue), list(Um)
    bin_of = lambda r: int(np.digitize(r, q))
    out = {"early_units": len(ce), "modern_units": len(cm)}
    n = 100000
    designs = {
        "early_early": [(ce[i], ce[j]) for i, j in rng.integers(len(ce), size=(n, 2)) if i != j],
        "early_modern": [(ce[i], cm[j]) for i, j in zip(rng.integers(len(ce), size=n), rng.integers(len(cm), size=n))
                         if ce[i] != cm[j]],
        "same_circle_cross_era": [(c, c) for c in ce if c in Um],
    }
    for kind, P in designs.items():
        A = Ue; B = Ue if kind == "early_early" else Um
        st = ruzicka(np.array([A[x][0] for x, _ in P]), np.array([B[y][0] for _, y in P]))
        so = ruzicka(np.array([A[x][1] for x, _ in P]), np.array([B[y][1] for _, y in P]))
        bx = np.array([bin_of(A[x][2]) for x, _ in P])
        key = (np.minimum(bx, [bin_of(B[y][2]) for _, y in P]) * 3 + np.maximum(bx, [bin_of(B[y][2]) for _, y in P])
               if kind == "early_early" else bx)
        within, w = [], []
        for kk in np.unique(key):
            m = key == kk
            if m.sum() >= 100:
                within.append(rho(so[m], st[m])); w.append(m.sum())
        out[kind] = {"pairs": len(P), "pooled": rho(so, st),
                     "within_bins": float(np.average(within, weights=w)) if within else None,
                     "median_shift": float(np.nanmedian(so - st))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("test", choices=["t1", "t2"])
    ap.add_argument("--cache"); ap.add_argument("--cbc"); ap.add_argument("--real")
    ap.add_argument("--degraded", nargs="*", default=[])
    ap.add_argument("--pairs", type=int, default=100000)
    ap.add_argument("--migration", default=None, help="E066c (t1): AVONET migration classes to keep, e.g. 1 or 1,2")
    ap.add_argument("--cell-mask", default=None, help="E079 (t1): npz of (H, W) bool strata masks")
    ap.add_argument("--cell-mask-key", default=None)
    ap.add_argument("--similarity", default=None, help="E066c: species_similarity.csv (species_code, migration)")
    ap.add_argument("--agg", type=int, nargs="*", default=[],
                    help="E066b: average k counts per circle and era before comparing (one run per k)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = t1(a) if a.test == "t1" else t2(a)
    tag = a.test + ("_agg" if a.agg else "") + (f"_mig{a.migration.replace(',', '')}" if a.migration else "") + (f"_{a.cell_mask_key}" if a.cell_mask else "")
    json.dump(res, open(os.path.join(a.out, f"e066_{tag}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
