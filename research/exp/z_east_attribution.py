"""Which Z dims carry the east's change in the B1-dyn House Finch SDM, and is that change shared by the community?
Pre-registered in research/experiments/GRID54_Z_EAST_ATTRIBUTION_2026-10-30.md. CPU, NumPy/pandas/SciPy.

    python z_east_attribution.py --design <design_step3> --cache <research_cache dir> --similarity <csv> --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_b0_score import load_design  # noqa: E402
from sdm_b1_sdm import Features, fit_nb  # noqa: E402

LAM_B1 = 1000.0
LAM_RIDGE = 1000.0
ERA_E, ERA_L, ERA_P = (1966, 1975), (2010, 2020), (1940, 1955)


def bands(lon, lat):
    lat_ok = (lat >= 25) & (lat < 49)
    return {"east80": (lon >= -80) & lat_ok, "central_east": (lon >= -95) & (lon < -80) & lat_ok,
            "all": np.ones(lon.shape, bool)}


def era_mask(years, era):
    return (years >= era[0]) & (years <= era[1])


def corr_ci(o, p, n=2000, seed=0):
    o, p = np.asarray(o), np.asarray(p)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    if o.size < 5:
        return {"n": int(o.size)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, o.size, (n, o.size))
    rs = np.array([np.corrcoef(o[i], p[i])[0, 1] for i in idx])
    ks = np.array([np.polyfit(p[i], o[i], 1)[0] for i in idx])
    return {"n": int(o.size), "corr": float(np.corrcoef(o, p)[0, 1]), "corr_lo90": float(np.nanpercentile(rs, 5)),
            "corr_hi90": float(np.nanpercentile(rs, 95)), "k": float(np.polyfit(p, o, 1)[0]),
            "k_lo90": float(np.nanpercentile(ks, 5)), "k_hi90": float(np.nanpercentile(ks, 95)),
            "mean_O": float(o.mean()), "mean_P": float(p.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--similarity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse-coef", action="store_true", help="load b1dyn_coef.npz instead of refitting B1-dyn")
    ap.add_argument("--readout-exclude-band", action="store_true",
                    help="C2: fit each species' Z readout WITHOUT the band's cells (spatial transfer), then predict the band")
    a = ap.parse_args()
    D = load_design(a.design)
    Z = np.load(os.path.join(a.design, "z_cells.npy")).astype(np.float64)
    years = D["years"]; nT, nL, M = Z.shape
    lr, lc = D["land_rows"], D["land_cols"]
    lut = -np.ones((lr.max() + 1, lc.max() + 1), int); lut[lr, lc] = np.arange(nL)
    cell = lut[D["row"], D["col"]]
    B = bands(D["land_lon"], D["land_lat"])
    out = {}

    # ---- A. B1-dyn refit (lambda 1000, as selected in B1) and exact attribution of band-mean log expected count
    real_train = ~D["holdout"] & ~D["pseudo"] & (D["weight"] > 0)
    zrows = Z[D["t"], cell]
    feat = Features(zrows, real_train)
    X = feat(zrows)
    g = np.where(D["obsroute"] >= 0, D["obsroute"], -1)
    coefp = os.path.join(os.path.dirname(a.out), "b1dyn_coef.npz")
    if a.reuse_coef and os.path.exists(coefp):
        cz = np.load(coefp); fit = {"b": cz["b"], "phi": float(cz["phi"])}
        assert np.allclose(cz["mu"], feat.mu) and np.allclose(cz["sd"], feat.sd)
    else:
        fit = fit_nb(X[real_train], D["y"][real_train], D["weight"][real_train], g[real_train], int(D["obsroute"].max()) + 1, LAM_B1)
    b = fit["b"]; b_lin, b_sq = b[1:1 + M], b[1 + M:1 + 2 * M]
    np.savez(os.path.join(os.path.dirname(a.out), "b1dyn_coef.npz"), b=b, mu=feat.mu, sd=feat.sd, phi=fit["phi"])
    zt = (Z - feat.mu[:M]) / feat.sd[:M]
    qt = (Z ** 2 - feat.mu[M:]) / feat.sd[M:]
    attr = {}
    for bn, bm in B.items():
        for early_name, early in (("1966-75", ERA_E), ("1940-55", ERA_P)):
            e, l = era_mask(years, early), era_mask(years, ERA_L)
            dz = zt[l][:, bm].mean((0, 1)) - zt[e][:, bm].mean((0, 1))
            dq = qt[l][:, bm].mean((0, 1)) - qt[e][:, bm].mean((0, 1))
            contrib = b_lin * dz + b_sq * dq
            tot = float(contrib.sum())
            order = np.argsort(-contrib) if tot > 0 else np.argsort(contrib)
            cum = np.cumsum(contrib[order]) / tot
            dens = lambda m_: np.exp(np.clip(X_band(zt, qt, m_, bm, b), -30, 15)).mean()
            attr[f"{bn}|{early_name}->2010-20"] = {
                "total_dlog": tot, "ratio_of_mean_counts": float(dens(e) / dens(l)),
                "top": [{"dim": int(d), "contrib": float(contrib[d]), "share": float(contrib[d] / tot),
                         "lin": float(b_lin[d] * dz[d]), "sq": float(b_sq[d] * dq[d]), "dz_std": float(dz[d])}
                        for d in order[:10]],
                "n_dims_50": int(np.argmax(cum >= .5) + 1), "n_dims_80": int(np.argmax(cum >= .8) + 1),
                "abs_share_top5": float(np.abs(contrib[order[:5]]).sum() / np.abs(contrib).sum()),
                "rank_of_age_top": {f"Z_{d}": int(np.where(order == d)[0][0]) + 1 for d in (19, 17, 18, 7)},
                "contrib_all": contrib.tolist()}
            print(bn, early_name, "total dlog %.3f" % tot, "n50", attr[f"{bn}|{early_name}->2010-20"]["n_dims_50"],
                  "top", [(t["dim"], round(t["share"], 3)) for t in attr[f"{bn}|{early_name}->2010-20"]["top"][:6]],
                  "age-top ranks", attr[f"{bn}|{early_name}->2010-20"]["rank_of_age_top"], flush=True)
    out["attribution_b1dyn"] = attr
    key = "east80|1966-75->2010-20"
    c = np.array(attr[key]["contrib_all"]); tot = c.sum()
    order = np.argsort(-c); cum = np.cumsum(c[order]) / tot
    hf_dims = order[:int(np.argmax(cum >= .8) + 1)]
    out["hf_dims_80pct_east80"] = [int(d) for d in hf_dims]

    # ---- C. community: 27 km cache -> 54 km cell-years
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    codes = list(meta["community"]) + list(meta["dev_species"])
    keys = np.load(os.path.join(a.cache, "keys.npy"))
    Xs = np.hstack([np.load(os.path.join(a.cache, "X_comm.npy")), np.load(os.path.join(a.cache, "X_dev.npy"))]).astype(np.float64)
    pr, pc = keys[:, 0] // 2, keys[:, 1] // 2
    inb = (pr <= lr.max()) & (pc <= lc.max())
    lc54 = np.full(keys.shape[0], -1); lc54[inb] = lut[pr[inb], pc[inb]]
    keep = (lc54 >= 0) & (keys[:, 2] >= 1966) & (keys[:, 2] <= 2020)
    out["cache_rows"] = {"total": int(keys.shape[0]), "dropped_nonland_parent": int(((lc54 < 0)).sum()), "kept_1966_2020": int(keep.sum())}
    prevalence = (Xs[(keys[:, 2] >= 1966) & (keys[:, 2] <= 2020)] > 0).mean(0)
    df = pd.DataFrame(Xs[keep])
    df["cell"], df["year"] = lc54[keep], keys[keep, 2]
    agg = df.groupby(["cell", "year"]).mean()                            # 54 km cell-year mean counts
    ck = np.array([list(k) for k in agg.index])
    Y = agg.to_numpy()
    LY = np.log1p(Y)
    c54, y54 = ck[:, 0], ck[:, 1]
    # House Finch on the same scale: real BBS rows 1966-2020 -> cell-year mean count
    hm = ~D["pseudo"] & (D["year"] <= 2020)
    hdf = pd.DataFrame({"cell": cell[hm], "year": D["year"][hm], "y": D["y"][hm]}).groupby(["cell", "year"])["y"].mean()
    hc, hy = np.array([k[0] for k in hdf.index]), np.array([k[1] for k in hdf.index])
    HY = np.log1p(hdf.to_numpy())

    def F_of(cells_, years_):
        zz = Z[years_ - years[0], cells_]
        return np.hstack([(zz - feat.mu[:M]) / feat.sd[:M], (zz ** 2 - feat.mu[M:]) / feat.sd[M:]])

    def ridge(F, Ymat):
        F1 = np.hstack([np.ones((F.shape[0], 1)), F])
        P = np.eye(F1.shape[1]) * LAM_RIDGE; P[0, 0] = 0
        return np.linalg.solve(F1.T @ F1 + P, F1.T @ Ymat)

    Fs = F_of(c54, y54)
    Ws_all = ridge(Fs, LY)[1:]                  # (2M, S)
    Wh_all = ridge(F_of(hc, hy), HY[:, None])[1:, 0]
    Fh = F_of(hc, hy)
    sim = pd.read_csv(a.similarity).set_index("species_code")
    mig = np.array([sim.loc[s, "migration"] if s in sim.index else np.nan for s in codes])
    res_c = {}
    for bn in ("east80", "central_east"):
        bm = B[bn]
        if a.readout_exclude_band:
            Ws = ridge(Fs[~bm[c54]], LY[~bm[c54]])[1:]
            Wh = ridge(Fh[~bm[hc]], HY[~bm[hc]][:, None])[1:, 0]
        else:
            Ws, Wh = Ws_all, Wh_all
        dF = np.concatenate([zt[era_mask(years, ERA_L)][:, bm].mean((0, 1)) - zt[era_mask(years, ERA_E)][:, bm].mean((0, 1)),
                             qt[era_mask(years, ERA_L)][:, bm].mean((0, 1)) - qt[era_mask(years, ERA_E)][:, bm].mean((0, 1))])
        sel = np.zeros(2 * M, bool); sel[hf_dims] = True; sel[M + hf_dims] = True
        P = dF @ Ws; PHF = (dF * sel) @ Ws
        Ph, PHFh = float(dF @ Wh), float((dF * sel) @ Wh)
        # observed change over cells surveyed in both eras
        inb_ = bm[c54]
        e_ = inb_ & (y54 >= ERA_E[0]) & (y54 <= ERA_E[1]); l_ = inb_ & (y54 >= ERA_L[0]) & (y54 <= ERA_L[1])
        cells_both = np.intersect1d(np.unique(c54[e_]), np.unique(c54[l_]))
        def cell_mean(mask, V):
            d_ = pd.DataFrame(V[mask]); d_["c"] = c54[mask]
            return d_.groupby("c").mean().reindex(cells_both)
        O = (cell_mean(l_, LY).to_numpy() - cell_mean(e_, LY).to_numpy()).mean(0)
        pres_e = (Y[e_] > 0).mean(0); pres_l = (Y[l_] > 0).mean(0)
        coloniser = (pres_e < .05) & (pres_l >= .20)
        he = bm[hc] & (hy >= ERA_E[0]) & (hy <= ERA_E[1]); hl = bm[hc] & (hy >= ERA_L[0]) & (hy <= ERA_L[1])
        hb = np.intersect1d(np.unique(hc[he]), np.unique(hc[hl]))
        hO = float(pd.Series(HY[hl], index=hc[hl]).groupby(level=0).mean().reindex(hb).mean()
                   - pd.Series(HY[he], index=hc[he]).groupby(level=0).mean().reindex(hb).mean())
        strata = {
            "sedentary_common_noncoloniser": (mig == 1) & (prevalence >= .08) & ~coloniser,
            "sedentary_common_all": (mig == 1) & (prevalence >= .08),
            "sedentary_all": mig == 1, "residents_1_2": (mig == 1) | (mig == 2), "migrants_3": mig == 3,
            "colonisers": coloniser, "all_species": np.ones(len(codes), bool)}
        st = {}
        for sn, sm in strata.items():
            sm = sm & np.isfinite(O)
            st[sn] = {"O_vs_P": corr_ci(O[sm], P[sm]), "O_vs_PHF": corr_ci(O[sm], PHF[sm]),
                      "hf_P_rank_from_top": int((P[sm] > Ph).sum()) + 1, "n": int(sm.sum()),
                      "hf_P_z": float((Ph - P[sm].mean()) / (P[sm].std() + 1e-12)) if sm.sum() > 2 else None}
        top_pos = np.argsort(-PHF)[:12]
        res_c[bn] = {"n_cells_both_eras": int(len(cells_both)), "hf": {"P": Ph, "P_hfdims": PHFh, "O": hO,
                     "n_cells_both": int(len(hb))}, "strata": st,
                     "colonisers": [codes[i] for i in np.where(coloniser)[0]],
                     "top_species_by_P_hfdims": [{"code": codes[i], "P_hf": float(PHF[i]), "P": float(P[i]), "O": float(O[i]),
                                                  "mig": None if np.isnan(mig[i]) else int(mig[i]), "prev": float(prevalence[i]),
                                                  "coloniser": bool(coloniser[i])} for i in top_pos]}
        print("\n", bn, "HF P %.3f P_hfdims %.3f O %.3f" % (Ph, PHFh, hO), "colonisers", res_c[bn]["colonisers"], flush=True)
        for sn, v in st.items():
            print("  ", sn, "n", v["n"], "O~P", {k: round(x, 3) for k, x in v["O_vs_P"].items() if k in ("corr", "corr_lo90", "corr_hi90", "k")},
                  "O~P_HF", {k: round(x, 3) for k, x in v["O_vs_PHF"].items() if k in ("corr", "corr_lo90", "corr_hi90", "k")},
                  "HF rank", v["hf_P_rank_from_top"], "z", v["hf_P_z"] and round(v["hf_P_z"], 2), flush=True)
    out["shared_change"] = res_c
    json.dump(out, open(a.out, "w"), indent=1)
    print("DONE")


def X_band(zt, qt, era_m, bm, b):
    M = zt.shape[2]
    return b[0] + zt[era_m][:, bm] @ b[1:1 + M] + qt[era_m][:, bm] @ b[1 + M:1 + 2 * M]


if __name__ == "__main__":
    main()
