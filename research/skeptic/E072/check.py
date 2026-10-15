"""Skeptic (step d) for E072/E072b: attack the 'honest change uncertainty = const + gU * U' claim.

    python research/lib/launch.py research/skeptic/E072/check.py --cache <tempho cache> --similarity <csv> --out <dir>

Rebuilds the E072 arrays (delta representation only; E073_COVER as set in the environment) with e072_honest's own
functions, then runs checks (a)-(e) (see the skeptic brief). Writes <out>/check.json.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(REPO, "research", "exp"), os.path.join(REPO, "research"), REPO]

import e072_honest as H  # noqa: E402
from anchored_delta import (ZONES, ZONES_TIF, Cov, Estimand, cell_mean, cover_trend_groups,  # noqa: E402
                            fit_blocks)

NC = H.NC


def nnls_fit(cols, y):
    return H.nnls(np.stack(cols, 1), y)


def setup(a):
    est = Estimand(a.cache, 20, a.similarity)
    keys, Y = est.keys, est.Y
    cov = Cov(keys, ~est.ho & ~est.bf & est.ty, est.ty)
    fin = cov.fin
    trn = np.where(~est.ho & ~est.bf & est.ty & fin)[0]
    uc, slope, thr, trn_cells, inv = cover_trend_groups(est, cov)
    keep = ~(trn_cells & (np.abs(slope) > thr))
    designs = {"full": trn, "cover": trn[keep[inv[trn]]]}
    ev_cid = est.cells_ev[:, 0].astype(np.int64) * 100000 + est.cells_ev[:, 1]
    beyond = np.abs(slope[np.searchsorted(uc, ev_cid)]) > thr
    Yw = Y - cell_mean(keys, Y, est.ty & fin)
    Zf = np.asarray(np.load(os.path.join(a.cache, "esk_annual.npy"), mmap_mode="r"), "float64")
    Zw = Zf - cell_mean(keys, Zf, est.ty & fin)
    Wg = {g: cov.W[(H.HL, g)] for g in NC}
    W = np.hstack([Wg[g] for g in NC])
    dW_ev = {g: est.delta(Wg[g]) for g in NC}
    dWall_ev = np.hstack([dW_ev[g] for g in NC])
    S = H.modern_level_S(est, Y)
    try:
        import rasterio
        with rasterio.open(ZONES_TIF) as src:
            zg = src.read(1)
        zone_ev = np.array([ZONES.get(int(zg[r, c]), "other") for r, c in est.cells_ev])
        zone_all = np.array([ZONES.get(int(zg[r, c]), "other") for r, c in cov.cells])
    except Exception:
        zone_ev = zone_all = None
    out = {}
    for dname, rows in designs.items():
        B, Bs = H.comm_maps(W, Zw, rows, keys, a.n_map_boot)
        dl = W @ B
        m = fit_blocks([dl], Yw, rows)
        dd = dWall_ev @ B
        coef = m["coef"]
        P = dd @ coef.T
        Pb = np.stack([(dWall_ev @ Bb) @ coef.T for Bb in Bs])
        U = Pb.var(0, ddof=1)
        V = np.einsum("cp,spq,cq->cs", dd, m["Ainv"], dd)
        out[dname] = dict(B=B, Bs=Bs, coef=coef, P=P, U=U, V=V, rows=rows)
    return est, cov, S, beyond, dW_ev, dWall_ev, out, zone_ev, zone_all


def block_of(cells):
    blk = (cells[:, 0] // 6).astype(np.int64) * 100000 + cells[:, 1] // 6
    return np.unique(blk, return_inverse=True)[1]


def law_cols(name, U, S, sp, cells, extra):
    """Columns (cell-level, ratio-of-sums units) and the per cell x species predicted-variance builder."""
    SS = S[sp].sum()
    if name == "M1":
        return [np.ones(len(cells))], [lambda: np.ones_like(U) * S[None, :]]
    if name == "M4":
        return ([np.ones(len(cells)), U[np.ix_(cells, sp)].sum(1) / SS],
                [lambda: np.ones_like(U) * S[None, :], lambda: U])
    # extra[name] = (per cell x species array X in variance units) -> const + c * X
    X = extra[name]
    return ([np.ones(len(cells)), X[np.ix_(cells, sp)].sum(1) / SS], [lambda: np.ones_like(U) * S[None, :], lambda: X])


def fit_law(name, e2, V, U, S, sp, cells, extra):
    SS = S[sp].sum()
    y = (e2[np.ix_(cells, sp)] - V[np.ix_(cells, sp)]).sum(1) / SS
    cols, builders = law_cols(name, U, S, sp, cells, extra)
    g = nnls_fit(cols, y)
    pv = V.copy()
    for gi, b in zip(g, builders):
        pv = pv + gi * b()
    return g, pv


def R(e2, pv, cells, sp):
    return float(e2[np.ix_(cells, sp)].sum() / pv[np.ix_(cells, sp)].sum())


def abba_cov_fraction(n, rho):
    """AR(1) noise, n consecutive years split ABBA: corr(mean A, mean B)."""
    idx = np.arange(n)
    a = np.array([(i % 4) in (0, 3) for i in idx])
    C = rho ** np.abs(idx[:, None] - idx[None, :])
    wa, wb = a / a.sum(), (~a) / (~a).sum()
    return float(wa @ C @ wb / np.sqrt((wa @ C @ wa) * (wb @ C @ wb)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--similarity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-map-boot", type=int, default=30)
    ap.add_argument("--n-cell-boot", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    est, cov, S, beyond, dW_ev, dWall_ev, reps, zone_ev, zone_all = setup(a)
    Ta, Tb = est.T["dev"]
    Ta, Tb = np.asarray(Ta, float), np.asarray(Tb, float)
    sp = np.where(est.strata["sedentary"] & (S > 0))[0]
    inside_c, beyond_c = np.where(~beyond)[0], np.where(beyond)[0]
    res = {"cache": a.cache, "E073_COVER": os.environ.get("E073_COVER"), "n_sp": int(len(sp)),
           "n_inside": int(len(inside_c)), "n_beyond": int(len(beyond_c))}

    d = reps["cover"]
    P, U, V = d["P"], d["U"], d["V"]
    e2 = (Ta - P) * (Tb - P)
    # stand-ins for (d)
    dw2 = (dWall_ev ** 2).sum(1)
    dwc2 = (dW_ev["cover"] ** 2).sum(1)
    extra = {"dW2": dw2[:, None] * S[None, :], "dWcover2": dwc2[:, None] * S[None, :], "P2": P ** 2,
             "logdW2": np.log1p(dw2)[:, None] * S[None, :]}
    laws = {}
    for name in ("M1", "M4", "dW2", "dWcover2", "P2", "logdW2"):
        g, pv = fit_law(name, e2, V, U, S, sp, inside_c, extra)
        laws[name] = pv
        res.setdefault("laws", {})[name] = {"gamma": [float(x) for x in g], "R_inside": R(e2, pv, inside_c, sp),
                                           "R_beyond": R(e2, pv, beyond_c, sp)}
    # ---------------- (a) species dominance, per-species R, ABBA independence
    ra = {}
    Ssp = S[sp]
    order = np.argsort(-Ssp)
    ra["S_share_top5"] = float(Ssp[order[:5]].sum() / Ssp.sum())
    eb = e2[np.ix_(beyond_c, sp)].sum(0)
    ra["e2_beyond_share_top5_species"] = float(np.sort(eb)[::-1][:5].sum() / eb.sum())
    for name in ("M1", "M4"):
        pv = laws[name]
        rec = {}
        for tag, cells in (("inside", inside_c), ("beyond", beyond_c)):
            Es, Ps = e2[np.ix_(cells, sp)].sum(0), pv[np.ix_(cells, sp)].sum(0)
            Rs = Es / Ps
            w_eq = 1 / Ssp
            w_cap = 1 / np.maximum(Ssp, np.median(Ssp))
            loo = [(Es.sum() - Es[i]) / (Ps.sum() - Ps[i]) for i in range(len(sp))]
            rec[tag] = {"R_pooled": float(Es.sum() / Ps.sum()),
                        "R_equal_species_weight": float((w_eq * Es).sum() / (w_eq * Ps).sum()),
                        "R_capped_weight": float((w_cap * Es).sum() / (w_cap * Ps).sum()),
                        "R_species_median": float(np.median(Rs)),
                        "R_species_q25_q75": [float(np.percentile(Rs, 25)), float(np.percentile(Rs, 75))],
                        "frac_species_R_gt_1.5": float((Rs > 1.5).mean()),
                        "frac_species_R_lt_0.67": float((Rs < 0.67).mean()),
                        "R_leave_one_species_out_range": [float(min(loo)), float(max(loo))]}
        ra[name] = rec
    # ABBA: lag-1 autocorrelation of within-epoch yearly deviations (log1p) on held-out cells
    yrs = est.keys[:, 2]
    num = np.zeros(est.Y.shape[1]); den = np.zeros(est.Y.shape[1]); nyrs = []
    for lists in (est.early, est.modern):
        for r in lists:
            r = r[np.argsort(yrs[r])]
            nyrs.append(len(r))
            if len(r) < 3:
                continue
            Dv = est.Y[r] - est.Y[r].mean(0)
            consec = np.diff(yrs[r]) == 1
            num += (Dv[:-1][consec] * Dv[1:][consec]).sum(0)
            den += (0.5 * (Dv[:-1][consec] ** 2 + Dv[1:][consec] ** 2)).sum(0)
    ra["median_years_per_epoch"] = float(np.median(nyrs))
    nmed = int(np.median(nyrs))
    rho_raw = num[sp].sum() / den[sp].sum()
    rho = rho_raw + 1.0 / max(nmed, 2)            # rough small-sample (mean-removal) correction
    ra["lag1_autocorr_sedentary_pooled_raw"] = float(rho_raw)
    ra["lag1_autocorr_corrected"] = float(rho)
    ra["withheld_years"] = [int(min(est.withheld)), int(max(est.withheld))] if est.withheld else None
    frac = abba_cov_fraction(max(nmed, 4), max(rho, 0.0))
    ra["ABBA_corr_halfmeans_AR1"] = frac
    # noise variance per half (assuming independence): var(na) ~ E[(Ta-Tb)^2]/2 ; bias of e2 = frac * var(na)
    nv_half = ((Ta - Tb) ** 2 / 2)[:, sp].sum() / len(Ta)
    ra["noise_var_per_half_over_S"] = float(nv_half / Ssp.sum())
    ra["implied_e2_bias_over_S"] = float(frac * nv_half / Ssp.sum())
    ra["mean_e2_over_S"] = float(e2[:, sp].sum() / len(Ta) / Ssp.sum())
    ra["mean_Ta_minus_Tb_over_sdS"] = float(np.mean((Ta - Tb)[:, sp]) / np.sqrt(Ssp.mean()))
    res["a"] = ra

    # ---------------- (b) distribution: coverage / z of Tm = (Ta+Tb)/2
    rb = {}
    h = np.array([1.0 / max(len(e), 1) + 1.0 / max(len(m), 1) for e, m in zip(est.early, est.modern)])
    k_s = ((Ta - Tb) ** 2 / 4).sum(0) / h.sum()          # var((na+nb)/2) = E[(Ta-Tb)^2]/4 under independence
    nv = h[:, None] * k_s[None, :]
    Tm = 0.5 * (Ta + Tb)
    nz = (Ta != 0) | (Tb != 0)
    from scipy.stats import norm
    for name in ("M1", "M4"):
        pv = laws[name]
        z = (Tm - P) / np.sqrt(np.maximum(pv + nv, 1e-12))
        pit = norm.cdf(z)
        rec = {}
        for tag, cells in (("inside", inside_c), ("beyond", beyond_c)):
            for occ in ("all", "occupied"):
                mk = np.zeros_like(z, bool)
                mk[np.ix_(cells, sp)] = True
                if occ == "occupied":
                    mk &= nz
                zz = z[mk]
                p = pit[mk]
                rec[f"{tag}_{occ}"] = {"n": int(mk.sum()), "cover80": float((np.abs(zz) < 1.2816).mean()),
                                       "cover95": float((np.abs(zz) < 1.96).mean()), "sd_z": float(zz.std()),
                                       "robust_sd_z": float(1.4826 * np.median(np.abs(zz - np.median(zz)))),
                                       "frac_abs_z_gt3": float((np.abs(zz) > 3).mean()),
                                       "PIT_deciles_hist": np.histogram(p, bins=10, range=(0, 1))[0].tolist()}
        # noise-free R split by occupancy (no noise estimate needed): where does the S-uniform law misplace variance?
        for tag, cells in (("inside", inside_c), ("beyond", beyond_c)):
            ee, pp, oo = e2[np.ix_(cells, sp)], pv[np.ix_(cells, sp)], nz[np.ix_(cells, sp)]
            rec[f"R_{tag}_occupied_vs_empty"] = [float(ee[oo].sum() / pp[oo].sum()), float(ee[~oo].sum() / pp[~oo].sum())]
            rec[f"share_e2_{tag}_occupied"] = float(ee[oo].sum() / ee.sum())
            rec[f"share_predvar_{tag}_occupied"] = float(pp[oo].sum() / pp.sum())
            # z with occupancy-specific noise (k_s fitted on occupied cell-species only)
            ko = (((Ta - Tb) ** 2 / 4) * nz).sum(0) / np.maximum((h[:, None] * nz).sum(0), 1e-12)
            nvo = h[:, None] * ko[None, :] * nz
            zo = ((Tm - P) / np.sqrt(np.maximum(pv + nvo, 1e-12)))[np.ix_(cells, sp)][oo]
            rec[f"{tag}_occupied_occnoise"] = {"cover80": float((np.abs(zo) < 1.2816).mean()),
                                               "cover95": float((np.abs(zo) < 1.96).mean()),
                                               "sd_z": float(zo.std()), "frac_abs_z_gt3": float((np.abs(zo) > 3).mean())}
        rec["mean_noise_var_over_S"] = [float(nv[np.ix_(c, sp)].sum(1).mean() / Ssp.sum()) for c in (inside_c, beyond_c)]
        rec["mean_pred_var_over_S"] = [float(pv[np.ix_(c, sp)].sum(1).mean() / Ssp.sum()) for c in (inside_c, beyond_c)]
        # reliability: beyond+inside cells binned by the law's cell-level predicted variance (quintiles)
        cv = pv[:, sp].sum(1)
        ce = e2[:, sp].sum(1)
        qs = np.quantile(cv, [.2, .4, .6, .8])
        bins = np.digitize(cv, qs)
        rec["R_by_predvar_quintile_all_cells"] = [float(ce[bins == b].sum() / cv[bins == b].sum()) for b in range(5)]
        qb = np.quantile(cv[beyond_c], [1 / 3, 2 / 3])
        bb = np.digitize(cv[beyond_c], qb)
        rec["R_by_predvar_tercile_beyond"] = [float(ce[beyond_c][bb == b].sum() / cv[beyond_c][bb == b].sum())
                                              for b in range(3)]
        rb[name] = rec
    res["b"] = rb

    # ---------------- (c) does gU matter / is the beyond improvement significant
    rc = {}
    for tag, cells in (("all", np.arange(len(Ta))), ("inside", inside_c), ("beyond", beyond_c)):
        g, _ = fit_law("M4", e2, V, U, S, sp, cells, extra)
        rc[f"M4_gamma_fit_on_{tag}"] = [float(x) for x in g]
    SS = Ssp.sum()
    rc["U_share_of_M4_predvar"] = [float((laws["M4"] - V - res["laws"]["M4"]["gamma"][0] * S[None, :])[np.ix_(c, sp)].sum()
                                         / laws["M4"][np.ix_(c, sp)].sum()) for c in (inside_c, beyond_c)]
    # cell-level correlation of e2 with U (inside, beyond): does U predict WHICH cells err?
    ce = (e2[:, sp] - V[:, sp]).sum(1) / SS
    cu = U[:, sp].sum(1) / SS
    from scipy.stats import spearmanr
    rc["spearman_cell_e2_vs_U"] = {t: float(spearmanr(ce[c], cu[c])[0]) for t, c in (("inside", inside_c),
                                                                                       ("beyond", beyond_c))}
    rc["spearman_cell_e2_vs_dW2"] = {t: float(spearmanr(ce[c], dw2[c])[0]) for t, c in (("inside", inside_c),
                                                                                          ("beyond", beyond_c))}
    # species-half cross-fit (E072's halves): inside-fitted law, beyond R per half
    rng = np.random.default_rng(5)
    half = rng.random(len(S)) < 0.5
    for name in ("M1", "M4"):
        pv = laws[name]
        rc[f"{name}_R_beyond_by_species_half"] = [R(e2, pv, beyond_c, sp[half[sp] == hh]) for hh in (True, False)]
        rc[f"{name}_R_inside_by_species_half"] = [R(e2, pv, inside_c, sp[half[sp] == hh]) for hh in (True, False)]
    # block bootstrap over cells (6x6 blocks), refitting inside laws each draw
    blk = block_of(est.cells_ev)
    ub = np.unique(blk)
    names = ("M1", "M4", "dW2", "P2")
    draws = {n: [] for n in names}
    gU = []
    brg = np.random.default_rng(17)
    members = [np.where(blk == b)[0] for b in ub]
    for _ in range(a.n_cell_boot):
        pick = brg.integers(0, len(ub), len(ub))
        cells = np.concatenate([members[i] for i in pick])
        ins, bey = cells[~beyond[cells]], cells[beyond[cells]]
        if len(bey) == 0:
            continue
        for n in names:
            g, pv = fit_law(n, e2, V, U, S, sp, ins, extra)
            draws[n].append(R(e2, pv, bey, sp))
            if n == "M4":
                gU.append(g[1])
    D = {n: np.array(v) for n, v in draws.items()}
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 50)), float(np.percentile(v, 97.5))]
    rc["boot_R_beyond"] = {n: pct(v) for n, v in D.items()}
    rc["boot_M1_minus_M4"] = pct(D["M1"] - D["M4"])
    rc["boot_frac_M4_closer_to_1_than_M1"] = float((np.abs(D["M4"] - 1) < np.abs(D["M1"] - 1)).mean())
    rc["boot_gU"] = pct(np.array(gU))
    rc["boot_frac_gU_zero"] = float((np.array(gU) <= 1e-9).mean())
    rc["boot_M1_R_beyond_frac_gt_1"] = float((D["M1"] > 1).mean())
    res["c"] = rc

    # ---------------- (d) novelty vs magnitude
    rd = {}
    cu_ = U[:, sp].sum(1)
    for nm, q in (("dW2", dw2), ("P2", (P[:, sp] ** 2).sum(1))):
        ratio = cu_ / np.maximum(q, 1e-12)
        rd[f"median_U_over_{nm}"] = [float(np.median(ratio[inside_c])), float(np.median(ratio[beyond_c]))]
        rd[f"spearman_U_vs_{nm}"] = float(spearmanr(cu_, q)[0])
    rd["median_dW2"] = [float(np.median(dw2[inside_c])), float(np.median(dw2[beyond_c]))]
    rd["median_dWcover2"] = [float(np.median(dwc2[inside_c])), float(np.median(dwc2[beyond_c]))]
    rd["mean_U_over_S"] = [float(cu_[inside_c].mean() / SS), float(cu_[beyond_c].mean() / SS)]
    # regress log U on log |dW|^2 (all cells), then the residual inside vs beyond
    lx, ly = np.log(np.maximum(dw2, 1e-12)), np.log(np.maximum(cu_, 1e-15))
    A = np.stack([np.ones_like(lx), lx], 1)
    cf = np.linalg.lstsq(A[inside_c], ly[inside_c], rcond=None)[0]
    resid = ly - A @ cf
    rd["logU_on_logdW2_inside_fit"] = [float(c) for c in cf]
    rd["mean_logU_residual_beyond"] = float(resid[beyond_c].mean())
    rd["laws_R_beyond"] = {n: res["laws"][n]["R_beyond"] for n in res["laws"]}
    res["d"] = rd

    # ---------------- (e) 1920 decomposition (full design)
    re_ = {}
    f = reps["full"]
    coef, Bs, B = f["coef"], f["Bs"], f["B"]
    SSs = S[sp].sum()
    dW1920 = H.dense_dW(cov, NC, H.HL)
    widths = [dW1920[g].shape[1] for g in NC]
    offs = np.cumsum([0] + widths)

    def Ub_of(X):
        return np.stack([(X @ Bb) @ coef[sp].T for Bb in Bs]).var(0, ddof=1).sum(1) / SSs

    X1920 = np.hstack([dW1920[g] for g in NC])
    Ue_full = Ub_of(dWall_ev)
    U1920 = Ub_of(X1920)
    zones = ("west", "great_plains", "east")

    def summ(v, ev=False):
        zarr = zone_ev if ev else zone_all
        out = {"median": float(np.median(v)), "mean": float(np.mean(v))}
        if zarr is not None:
            out["by_zone_median"] = {z: float(np.median(v[zarr == z])) for z in zones if (zarr == z).any()}
            out["by_zone_mean"] = {z: float(np.mean(v[zarr == z])) for z in zones if (zarr == z).any()}
        return out
    re_["U_over_S_eval"] = summ(Ue_full, ev=True)
    re_["U_over_S_1920"] = summ(U1920)
    re_["U_over_S_eval_inside_q95_max"] = [float(np.percentile(Ue_full[inside_c], 95)), float(Ue_full[inside_c].max())]
    re_["frac_1920_cells_U_above_eval_inside_max"] = float((U1920 > Ue_full[inside_c].max()).mean())
    for i, g in enumerate(NC):
        Xo = np.zeros_like(X1920); Xo[:, offs[i]:offs[i + 1]] = X1920[:, offs[i]:offs[i + 1]]
        Xz = X1920.copy(); Xz[:, offs[i]:offs[i + 1]] = 0.0
        re_[f"U1920_only_{g}"] = summ(Ub_of(Xo))
        re_[f"U1920_without_{g}"] = summ(Ub_of(Xz))
        re_[f"dW2_{g}_median_eval_vs_1920"] = [float(np.median((dW_ev[g] ** 2).sum(1))),
                                               float(np.median((dW1920[g] ** 2).sum(1)))]
    # M4 law (inside-fitted, full design) applied: mean vs median; true-change-implied corr
    Pf, Uf, Vf = f["P"], f["U"], f["V"]
    e2f = (Ta - Pf) * (Tb - Pf)
    g, _ = fit_law("M4", e2f, Vf, Uf, S, sp, inside_c, extra)
    g1, _ = fit_law("M1", e2f, Vf, Uf, S, sp, inside_c, extra)
    v1920 = g[0] + g[1] * U1920
    pc2 = (((X1920 @ B) @ coef[sp].T) ** 2).sum(1) / SSs
    re_["M4_full_gamma"] = [float(x) for x in g]
    re_["M1_full_gamma"] = [float(x) for x in g1]
    re_["M4_var1920"] = summ(v1920)
    re_["pred_change2_1920"] = summ(pc2)
    re_["corr_pred_vs_true_1920 (=1-var/2)"] = summ(1 - v1920 / 2)
    re_["corr_H1920_H2000_if_pred_change_real (=1-(var+pc2)/2)"] = summ(1 - (v1920 + pc2) / 2)
    # constant span scaling: if the unpredictable part grows linearly with span (random walk), const x (85.5/39)
    span_eval = 2015.0 - 1976.0                     # EPOCH_EARLY 1966-86 -> EPOCH_MODERN 2005-25 midpoints
    span_1920 = 2015 - 1929.5
    re_["span_eval_y"], re_["span_1920_y"] = float(span_eval), float(span_1920)
    v_rw = g[0] * span_1920 / span_eval + g[1] * U1920
    re_["M4_var1920_const_scaled_by_span"] = summ(v_rw)
    # in eval data: does e2 grow with span? not testable within one epoch; report e2 at the full design
    re_["mean_e2_over_S_full_inside_beyond"] = [float(e2f[np.ix_(c, sp)].sum(1).mean() / SSs) for c in (inside_c, beyond_c)]
    res["e"] = re_
    json.dump(res, open(os.path.join(a.out, "check.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
