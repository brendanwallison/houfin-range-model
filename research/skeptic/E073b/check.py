"""E073 skeptic 2 (verification d) on three conclusions: C1 (B3 containment), C2 (B4 estimation variance),
C3 (C3/C3b cover extrapolation).

    python research/lib/launch.py research/skeptic/E073b/check.py c1 --out <scratch dir>
    python research/lib/launch.py research/skeptic/E073b/check.py c2 --out <scratch dir>
    python research/lib/launch.py research/skeptic/E073b/check.py c3 --out <scratch dir>

Reuses anchored_delta's Estimand / Cov / fit_blocks / scaled_coef (BLR_MAX_ITER=100 BLR_STARTS=0.5, as the jobs).
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"
os.environ.setdefault("BLR_MAX_ITER", "100")
os.environ.setdefault("BLR_STARTS", "0.5")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

REPO = "/mnt/c/Dev/houfin-range-model"
for p in (REPO, REPO + "/src", REPO + "/research", REPO + "/research/exp"):
    if p not in sys.path:
        sys.path.insert(0, p)
import anchored_delta as ad  # noqa: E402

SIM = "/home/breallis/houfin/work/houfin/research/E067/composition-d2e6a98f0eb7/species_similarity.csv"
CACHE = "/home/breallis/houfin/work/houfin/research_cache"
JOBS = "/home/breallis/houfin/work/houfin/research/E073"
RESULTS = os.path.join(REPO, "research/skeptic/E073b/check_results.json")
T0 = time.time()


def log(msg):
    print(f"[skeptic2 {time.time() - T0:7.0f}s] {msg}", flush=True)


def install_safe_inv():
    import torch
    torch.set_num_threads(4)

    def safe_inv(A):
        L, info = torch.linalg.cholesky_ex(A)
        out = torch.cholesky_inverse(L)
        bad = info > 0
        if bool(bad.any()):
            for i in torch.where(bad)[0].tolist():
                out[i] = torch.linalg.pinv(A[i])
        return out
    torch.linalg.inv = safe_inv
    log("safe_inv (Cholesky) installed")


def save(part, res):
    allres = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    allres[part] = res
    json.dump(allres, open(RESULTS, "w"), indent=1, default=float)


# ------------------------------------------------------------------------------------------------ scoring helpers
def terms(est, P, m, cells=None):
    """Per-species noise-free sums over cells (centred over cells): cv, vp, vt, sse."""
    Ta, Tb = est.T["dev"]
    P, Ta, Tb = P[:, m], Ta[:, m], Tb[:, m]
    if cells is not None:
        P, Ta, Tb = P[cells], Ta[cells], Tb[cells]
    Pc, Ac, Bc = P - P.mean(0), Ta - Ta.mean(0), Tb - Tb.mean(0)
    return (0.5 * ((Pc * Ac).sum(0) + (Pc * Bc).sum(0)), (Pc * Pc).sum(0), (Ac * Bc).sum(0),
            ((Pc - Ac) * (Pc - Bc)).sum(0))


def pooled(t, w=None):
    cv, vp, vt, sse = t
    w = np.ones_like(cv) if w is None else w
    return (float((w * cv).sum() / np.sqrt((w * vp).sum() * (w * vt).sum())), float((w * cv).sum() / (w * vp).sum()),
            float(1 - (w * sse).sum() / (w * vt).sum()))


def jk_species(tx, ty):
    """Leave-one-species-out diffs x - y (corr, mse): min / max."""
    S = len(tx[0])
    dc, dm = [], []
    for j in range(S):
        k = np.ones(S, bool)
        k[j] = False
        a = pooled([v[k] for v in tx])
        b = pooled([v[k] for v in ty])
        dc.append(a[0] - b[0])
        dm.append(a[2] - b[2])
    return {"corr_min": float(np.min(dc)), "corr_max": float(np.max(dc)), "mse_min": float(np.min(dm)),
            "mse_max": float(np.max(dm))}


def arm(D):
    return {"corr": float(D[0, 0]), "corr_ci": ad.ci(D[:, 0]), "k": float(D[0, 1]), "mse": float(D[0, 2]),
            "mse_ci": ad.ci(D[:, 2])}


def diff(Dx, Dy):
    dc, dm = Dx[:, 0] - Dy[:, 0], Dx[:, 2] - Dy[:, 2]
    return {"corr": float(dc[0]), "corr_ci": ad.ci(dc), "mse": float(dm[0]), "mse_ci": ad.ci(dm)}


def species_detail(est, Px, Py, m, cells=None):
    """Equal-weight-species pooled diff (weights 1/vt_s, vt_s > 0), per-species sign share, species jackknife."""
    tx, ty = terms(est, Px, m, cells), terms(est, Py, m, cells)
    vt = tx[2]
    ok = vt > 0
    w = np.where(ok, 1.0 / np.where(ok, vt, 1.0), 0.0)
    ex, ey = pooled(tx, w), pooled(ty, w)
    # per-species corr difference (same vt_s for both, so sign of cv/sqrt(vp) difference)
    rx = tx[0] / np.sqrt(np.maximum(tx[1], 1e-30))
    ry = ty[0] / np.sqrt(np.maximum(ty[1], 1e-30))
    d = (rx - ry)[ok] / np.sqrt(vt[ok])
    # capped weights: species with truth variance below the median are not up-weighted beyond the median
    med = np.median(vt[ok])
    wc = np.where(ok, 1.0 / np.maximum(np.where(ok, vt, med), med), 0.0)
    cx_, cy_ = pooled(tx, wc), pooled(ty, wc)
    # per-species contribution to the pooled corr difference (denominators held at pooled values)
    contrib = tx[0] / np.sqrt(tx[1].sum() * vt.sum()) - ty[0] / np.sqrt(ty[1].sum() * vt.sum())
    names = np.array(est.species)[m]
    top = np.argsort(-np.abs(contrib))[:5]
    # pooled diff without the 10 highest-truth-variance species
    keep = np.ones(len(vt), bool)
    keep[np.argsort(-vt)[:10]] = False
    kx, ky = pooled([v[keep] for v in tx]), pooled([v[keep] for v in ty])
    return {"equal_weight": {"corr": ex[0] - ey[0], "mse": ex[2] - ey[2], "n_species_vt_pos": int(ok.sum())},
            "capped_weight": {"corr": cx_[0] - cy_[0], "mse": cx_[2] - cy_[2]},
            "drop_top10_vt": {"corr": kx[0] - ky[0], "mse": kx[2] - ky[2]},
            "top5_contrib": {str(names[i]): float(contrib[i]) for i in top},
            "top5_share_of_total": float(contrib[top].sum() / contrib.sum()) if contrib.sum() != 0 else None,
            "top10_vt_share_of_vt": float(np.sort(vt)[::-1][:10].sum() / vt.sum()),
            "per_species_corr_diff": {"median": float(np.median(d)), "share_pos": float((d > 0).mean())},
            "jackknife": jk_species(tx, ty)}


def setup(cache, n_boot):
    est = ad.Estimand(cache, n_boot, SIM)
    log(f"estimand {cache}")
    cov = ad.Cov(est.keys, ~est.ho & ~est.bf & est.ty, est.ty)
    log("covariates")
    keys, fin = est.keys, cov.fin
    z = np.asarray(np.load(os.path.join(cache, "z_raw.npy"), mmap_mode="r")[:, :24], "float64")
    z = np.where(np.isfinite(z), z, 0.0)
    zbar = ad.cell_mean(keys, z, est.ty & fin)
    trn = np.where(~est.ho & ~est.bf & est.ty & fin)[0]
    Yw = est.Y - ad.cell_mean(keys, est.Y, est.ty & fin)
    ctx = ad.ctx_feats(cov, {"zbar": zbar}, "desk", 4, trn)
    return est, cov, z, zbar, trn, Yw, ctx


# ------------------------------------------------------------------------------------------------ C1
def run_c1(a):
    res = {"saved": {}, "refit": {}}
    # (1) saved B3 predictions: strata, MSE, equal-weight species, jackknife (all species), all four runs
    runs = {"t1995_s1": ("desk_tempho_1995_s1", "B3-t1995-s1-dd417580ddd9"),
            "t1985_s1": ("desk_tempho_1985_s1", "B3-t1985-s1-d4e5c52a3197"),
            "t1995_s2": ("desk_tempho_1995_s2", "B3-t1995-s2-c198332e95f2")}
    est0 = None
    for tag, (cache, job) in runs.items():
        est = ad.Estimand(os.path.join(CACHE, cache), a.n_boot, SIM)
        if tag == "t1995_s1":
            est0 = est
        sv = np.load(os.path.join(JOBS, job, "preds_b3_dev.npz"))
        P = {k.replace("_PLUS_", "+"): np.asarray(sv[k], "float64") for k in sv.files}
        o = {}
        for sname in ("all", "sedentary", "resident", "migratory"):
            m = est.strata[sname]
            D = {k: est.draws(P[k], "dev", m) for k in ("vcz", "vcc_h50", "vcc_h50+vcz", "comm_lin_h50_esk64_vcread",
                                                       "desk_split_level")}
            oo = {"n": int(m.sum()), "arms": {k: arm(v) for k, v in D.items()}, "diffs": {}}
            for x, y in (("vcc_h50+vcz", "vcz"), ("vcc_h50+vcz", "vcc_h50"), ("vcc_h50", "vcz"),
                         ("comm_lin_h50_esk64_vcread", "vcz")):
                dd = diff(D[x], D[y])
                dd.update(species_detail(est, P[x], P[y], m))
                oo["diffs"][f"{x} - {y}"] = dd
            o[sname] = oo
        res["saved"][tag] = o
        log(f"C1 saved {tag} [pooled corr, mse, eqw corr, capped corr, drop-top10 corr, jk min, top5 share]: "
            + json.dumps({s: {k: (round(v['corr'], 3), round(v['mse'], 3),
                                                       round(v['equal_weight']['corr'], 3),
                                                       round(v['capped_weight']['corr'], 3),
                                                       round(v['drop_top10_vt']['corr'], 3),
                                                       round(v['jackknife']['corr_min'], 3),
                                                       v['top5_share_of_total'])
                                                   for k, v in o[s]['diffs'].items()} for s in ('all', 'sedentary')}))
    save("C1", res)
    # (2) refits on t1995_s1: capacity-matched covariate delta (24 PCs of W50 -> 120 features = vcz's 120)
    cache = os.path.join(CACHE, "desk_tempho_1995_s1")
    est, cov, z, zbar, trn, Yw, ctx = setup(cache, a.n_boot)
    zw = z - zbar
    W = cov.w(50, "no_clock")
    Wc = W - W[trn].mean(0)
    _, sv_, Vt = np.linalg.svd(Wc[trn], full_matrices=False)
    W24 = Wc @ Vt[:24].T
    W24 = W24 / W24[trn].std(0)
    res["refit"]["W50_pca24_var_share"] = float((sv_[:24] ** 2).sum() / (sv_ ** 2).sum())
    P = {}

    def within(name, blocks):
        t = time.time()
        mm = ad.fit_blocks(blocks, Yw, trn)
        P[name] = np.hstack([est.delta(B) for B in blocks]) @ mm["coef"].T
        log(f"C1 fit {name}: {sum(B.shape[1] for B in blocks)} features, {time.time() - t:.0f}s")
    within("vcz", ad.vc_blocks(zw, ctx))
    sv = np.load(os.path.join(JOBS, "B3-t1995-s1-dd417580ddd9", "preds_b3_dev.npz"))
    saved = {k.replace("_PLUS_", "+"): np.asarray(sv[k], "float64") for k in sv.files}
    res["refit"]["vcz_reproduction_rel_maxdiff"] = float(np.abs(P["vcz"] - saved["vcz"]).max()
                                                         / np.abs(saved["vcz"]).max())
    within("vcc24", ad.vc_blocks(W24, ctx))
    within("vcc24+vcz", ad.vc_blocks(W24, ctx) + ad.vc_blocks(zw, ctx))
    # comm_lin + vcz (is DESK's change contained in the buildable form too?)
    Zfull = np.asarray(np.load(os.path.join(cache, "esk_annual.npy"), mmap_mode="r"), "float64")
    Zw64 = Zfull - ad.cell_mean(est.keys, Zfull, est.ty & cov.fin)
    G = W[trn].T @ W[trn]
    Bl = np.linalg.solve(G + 1e-1 * np.trace(G) / W.shape[1] * np.eye(W.shape[1]), W[trn].T @ Zw64[trn])
    dl = W @ Bl
    within("comm_lin+vcz", ad.vc_blocks(dl, ctx) + ad.vc_blocks(zw, ctx))
    for k in ("vcc_h50", "vcc_h50+vcz", "comm_lin_h50_esk64_vcread"):
        P[k] = saved[k]
    out = {}
    for sname in ("all", "sedentary", "migratory"):
        m = est.strata[sname]
        D = {k: est.draws(v, "dev", m) for k, v in P.items()}
        oo = {"arms": {k: arm(v) for k, v in D.items()}, "diffs": {}}
        for x, y in (("vcc24", "vcz"), ("vcc24+vcz", "vcz"), ("vcc24+vcz", "vcc24"), ("vcc24", "vcc_h50"),
                     ("comm_lin+vcz", "comm_lin_h50_esk64_vcread"), ("comm_lin+vcz", "vcz"),
                     ("vcc_h50+vcz", "vcz"), ("vcc_h50+vcz", "vcc_h50")):
            dd = diff(D[x], D[y])
            dd.update(species_detail(est, P[x], P[y], m))
            oo["diffs"][f"{x} - {y}"] = dd
        out[sname] = oo
        log(f"C1 refit {sname}: " + "  ".join(f"{k} {v['corr']:.3f}/k{v['k']:.2f}/mse{v['mse']:+.3f}"
                                               for k, v in oo["arms"].items()))
        for k, v in oo["diffs"].items():
            log(f"    {k:45s} corr {v['corr']:+.3f} [{v['corr_ci'][0]:+.3f},{v['corr_ci'][1]:+.3f}] mse {v['mse']:+.3f} "
                f"[{v['mse_ci'][0]:+.3f},{v['mse_ci'][1]:+.3f}] eqw {v['equal_weight']['corr']:+.3f} "
                f"cap {v['capped_weight']['corr']:+.3f} -top10 {v['drop_top10_vt']['corr']:+.3f} "
                f"jkmin {v['jackknife']['corr_min']:+.3f} sp+ {v['per_species_corr_diff']['share_pos']:.2f}")
    res["refit"]["scores"] = out
    save("C1", res)


# ------------------------------------------------------------------------------------------------ C2
def run_c2(a):
    cache = os.path.join(CACHE, a.cache or "desk_tempho_1995_s1")
    key = f"C2_{a.cache or 'desk_tempho_1995_s1'}"
    est, cov, z, zbar, trn, Yw, ctx = setup(cache, a.n_boot)
    keys, fin = est.keys, cov.fin
    blocks = ad.vc_blocks(cov.w(50, "no_clock"), ctx)
    X = np.hstack(blocks)
    dX = est.delta(X)
    mults = (1.0, 0.3, 0.1, 0.03, 0.01)
    t = time.time()
    m = ad.fit_blocks(blocks, Yw, trn)
    log(f"C2 full fit {time.time() - t:.0f}s")
    Pm = {mu: dX @ ad.scaled_coef(X[trn], Yw[trn], m, mu).T for mu in mults}
    res = {"mults": mults}
    # in-era change machinery (as arms_b4)
    yrs = keys[:, 2]
    ty_years = np.unique(yrs[est.ty])
    y0, y1 = int(ty_years.min()), int(ty_years.max())
    span = (y1 - y0 + 1) // 3
    early_m = est.ty & (yrs <= y0 + span - 1) & fin
    late_m = est.ty & (yrs >= y1 - span + 1) & fin
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    blk = (keys[:, 0] // 6).astype(np.int64) * 100000 + keys[:, 1] // 6
    rng = np.random.default_rng(13)
    ub = np.unique(blk[trn])
    fold_of_blk = dict(zip(ub, rng.integers(5, size=len(ub))))
    fold = np.array([fold_of_blk.get(b, -1) for b in blk])

    def in_era_change(F, rows_mask):
        cells = np.unique(cid[rows_mask])
        e, _ = ad.group_mean_rows(keys, F, early_m & np.isin(cid, cells), cid)
        l_, _ = ad.group_mean_rows(keys, F, late_m & np.isin(cid, cells), cid)
        ce = np.unique(cid[early_m & np.isin(cid, cells)])
        cl = np.unique(cid[late_m & np.isin(cid, cells)])
        both = np.intersect1d(ce, cl)
        return l_[np.searchsorted(cl, both)] - e[np.searchsorted(ce, both)]

    S = est.Y.shape[1]
    loss_ann = np.zeros((len(mults), S))
    loss_era = np.zeros((len(mults), S))
    kfold = {"ann_num": np.zeros(len(mults)), "ann_den": np.zeros(len(mults)), "era_num": np.zeros(len(mults)),
             "era_den": np.zeros(len(mults))}
    for f in range(5):
        tr_f = trn[fold[trn] != f]
        te_f = trn[fold[trn] == f]
        t = time.time()
        mf = ad.fit_blocks(blocks, Yw, tr_f)
        te_mask = np.zeros(len(keys), bool)
        te_mask[te_f] = True
        dXf = in_era_change(X, te_mask)
        dYf = in_era_change(est.Y, te_mask)
        dYc = dYf - dYf.mean(0)
        xm, ym = X[tr_f].mean(0), Yw[tr_f].mean(0)
        for i, mu in enumerate(mults):
            c = ad.scaled_coef(X[tr_f], Yw[tr_f], mf, mu)
            Pa = (X[te_f] - xm) @ c.T + ym
            loss_ann[i] += ((Yw[te_f] - Pa) ** 2).sum(0)
            Pac = Pa - Pa.mean(0)
            Yac = Yw[te_f] - Yw[te_f].mean(0)
            kfold["ann_num"][i] += (Pac * Yac).sum()
            kfold["ann_den"][i] += (Pac * Pac).sum()
            Pe = dXf @ c.T
            Pe = Pe - Pe.mean(0)
            loss_era[i] += ((Pe - dYc) ** 2).sum(0)
            kfold["era_num"][i] += (Pe * dYc).sum()
            kfold["era_den"][i] += (Pe * Pe).sum()
        log(f"C2 fold {f}: {time.time() - t:.0f}s")
    res["cv_k_annual"] = (kfold["ann_num"] / kfold["ann_den"]).tolist()
    res["cv_k_inera"] = (kfold["era_num"] / kfold["era_den"]).tolist()
    # in-sample k (full fit applied to its own training cells): overfitting vs scale
    trm = np.zeros(len(keys), bool)
    trm[trn] = True
    dXi, dYi = in_era_change(X, trm), in_era_change(est.Y, trm)
    Pi = dXi @ m["coef"].T
    Pi, dYi = Pi - Pi.mean(0), dYi - dYi.mean(0)
    res["insample_k_inera"] = float((Pi * dYi).sum() / (Pi * Pi).sum())
    Pa = (X[trn] - m["xbar"]) @ m["coef"].T + m["ybar"]
    Pac = Pa - Pa.mean(0)
    res["insample_k_annual"] = float((Pac * (Yw[trn] - Yw[trn].mean(0))).sum() / (Pac * Pac).sum())
    log(f"C2 k: cv annual {res['cv_k_annual']}, cv in-era {res['cv_k_inera']}, in-sample in-era "
        f"{res['insample_k_inera']:.3f}, in-sample annual {res['insample_k_annual']:.3f}")
    # chosen multipliers (training cells only)
    res["scores"] = {}
    for sname in ("sedentary", "all"):
        msk = est.strata[sname]
        D = {f"x{mu:g}": est.draws(P_, "dev", msk) for mu, P_ in Pm.items()}
        o = {"grid": {k: arm(v) for k, v in D.items()}, "chosen": {}}
        for crit, L in (("annual", loss_ann), ("inera", loss_era)):
            i_sh = int(np.argmin(L[:, msk].sum(1)))
            i_sp = np.argmin(L, 0)
            Psp = np.stack([Pm[mults[i_sp[j]]][:, j] for j in range(S)], 1)
            Dsp = est.draws(Psp, "dev", msk)
            o["chosen"][crit] = {"shared_mult": mults[i_sh], "shared": arm(D[f"x{mults[i_sh]:g}"]),
                                 "shared_vs_ML": diff(D[f"x{mults[i_sh]:g}"], D["x1"]),
                                 "perspecies_mult_counts": {f"{mu:g}": int((i_sp[msk] == i).sum())
                                                            for i, mu in enumerate(mults)},
                                 "perspecies": arm(Dsp), "perspecies_vs_ML": diff(Dsp, D["x1"])}
        # species halves: pick the multiplier on half A's TEST MSE, evaluate the gain on half B
        idx = np.where(msk)[0]
        gains, picks = [], []
        rs = np.random.default_rng(5)
        for _ in range(50):
            perm = rs.permutation(idx)
            A_, B_ = np.zeros(S, bool), np.zeros(S, bool)
            A_[perm[: len(idx) // 2]] = True
            B_[perm[len(idx) // 2:]] = True
            sa = [pooled(terms(est, Pm[mu], A_))[2] for mu in mults]
            best = mults[int(np.argmax(sa))]
            sb = pooled(terms(est, Pm[best], B_))[2] - pooled(terms(est, Pm[1.0], B_))[2]
            gains.append(sb)
            picks.append(best)
        o["species_halves"] = {"mean_gain_B": float(np.mean(gains)), "min_gain_B": float(np.min(gains)),
                               "share_pos": float((np.array(gains) > 0).mean()),
                               "picks": {f"{mu:g}": int(np.sum(np.array(picks) == mu)) for mu in mults}}
        o["oracle_scalar_mse_ML"] = float(D["x1"][0, 0] ** 2)
        res["scores"][sname] = o
        log(f"C2 {sname}: grid " + "  ".join(f"{k} {v['corr']:.3f}/k{v['k']:.2f}/mse{v['mse']:+.3f}"
                                              for k, v in o["grid"].items()))
        log(f"C2 {sname}: chosen {json.dumps({c: (v['shared_mult'], round(v['shared_vs_ML']['mse'], 3), v['shared_vs_ML']['mse_ci'], round(v['perspecies_vs_ML']['mse'], 3)) for c, v in o['chosen'].items()})}")
        log(f"C2 {sname}: species halves {o['species_halves']}; oracle scalar mse {o['oracle_scalar_mse_ML']:.3f}")
    save(key, res)
    # data-size: half of the training blocks (two disjoint halves); 1/5-year row subsample (win5's row count)
    hb = dict(zip(ub, np.random.default_rng(21).permutation(len(ub)) % 2))
    half = np.array([hb.get(b, -1) for b in blk])
    res["datasize"] = {}
    for tag, rows in (("half0", trn[half[trn] == 0]), ("half1", trn[half[trn] == 1]),
                      ("yr_sub5", trn[(yrs[trn] - 1966) % 5 == 2])):
        t = time.time()
        mh = ad.fit_blocks(blocks, Yw, rows)
        Ph = dX @ mh["coef"].T
        res["datasize"][tag] = {"n_rows": int(len(rows))}
        for sname in ("sedentary", "all"):
            msk = est.strata[sname]
            Dh = est.draws(Ph, "dev", msk)
            res["datasize"][tag][sname] = arm(Dh)
            res["datasize"][tag][sname]["vs_full"] = diff(Dh, est.draws(Pm[1.0], "dev", msk))
        log(f"C2 datasize {tag} ({len(rows)} rows, {time.time() - t:.0f}s): "
            f"{json.dumps({s: (round(res['datasize'][tag][s]['corr'], 3), round(res['datasize'][tag][s]['k'], 2), round(res['datasize'][tag][s]['mse'], 3)) for s in ('sedentary', 'all')})}")
    save(key, res)


# ------------------------------------------------------------------------------------------------ C3
def run_c3(a):
    cache = os.path.join(CACHE, "desk_tempho_1995_s1_ctrl073")
    est, cov, z, zbar, trn, Yw, ctx = setup(cache, a.n_boot)
    keys = est.keys
    uc, slope, thr, trn_cells, inv = ad.cover_trend_groups(est, cov)
    keep_cell = ~(trn_cells & (np.abs(slope) > thr))
    trn_rest = trn[keep_cell[inv[trn]]]
    dm_rand = np.load(os.path.join(JOBS, "c3b_train_drop_mask_random.npy")).astype(bool)
    trn_rand = trn[~dm_rand[keys[trn, 0], keys[trn, 1]]]
    ev_cid = est.cells_ev[:, 0].astype(np.int64) * 100000 + est.cells_ev[:, 1]
    sev = slope[np.searchsorted(uc, ev_cid)]
    groups = {"beyond": np.where(np.abs(sev) > thr)[0], "inside": np.where(np.abs(sev) <= thr)[0]}
    res = {"threshold": thr, "n_eval": {g: int(len(c)) for g, c in groups.items()},
           "n_rows": {"full": int(len(trn)), "cover": int(len(trn_rest)), "rand6": int(len(trn_rand))}}
    # training cells (row, col) and the cover-dropped ones
    tc_id = uc[trn_cells]
    tc_rc = np.stack([tc_id // 100000, tc_id % 100000], 1)
    cover_drop = np.abs(slope[trn_cells]) > thr
    n_target = int(cover_drop.sum())
    masks = {"cover": cover_drop}
    rd = dm_rand[tc_rc[:, 0], tc_rc[:, 1]]
    masks["rand6"] = rd
    # regional random controls: whole 18x18 super-blocks (random partition offset), same number of cells
    for s in range(8):
        rs = np.random.default_rng(300 + s)
        off = rs.integers(18, size=2)
        sb = ((tc_rc[:, 0] + off[0]) // 18) * 100000 + (tc_rc[:, 1] + off[1]) // 18
        msk = np.zeros(len(tc_rc), bool)
        for b in rs.permutation(np.unique(sb)):
            msk |= sb == b
            if msk.sum() >= n_target:
                break
        masks[f"region18_s{s}"] = msk
    # geographic twins: for each cover-dropped cell, the nearest NOT cover-dropped training cell (greedy, no reuse)
    rs = np.random.default_rng(400)
    avail = ~cover_drop
    twin = np.zeros(len(tc_rc), bool)
    for i in rs.permutation(np.where(cover_drop)[0]):
        cand = np.where(avail & ~twin)[0]
        d2 = ((tc_rc[cand] - tc_rc[i]) ** 2).sum(1)
        twin[cand[np.argmin(d2)]] = True
    masks["twin"] = twin
    # mask diagnostics: size, |cover slope| of dropped, distance from beyond / inside held-out cells to kept cells
    ev_rc = est.cells_ev
    res["masks"] = {}
    for k, msk in masks.items():
        kept = tc_rc[~msk]
        dd = {}
        for g, cells in groups.items():
            d = np.sqrt(((ev_rc[cells][:, None, :] - kept[None, :, :]) ** 2).sum(-1)).min(1)
            dd[g] = {"median": float(np.median(d)), "mean": float(d.mean()), "share_gt6": float((d > 6).mean())}
        res["masks"][k] = {"n_dropped": int(msk.sum()),
                           "dropped_abs_slope_median": float(np.median(np.abs(slope[trn_cells][msk]))),
                           "kept_share_beyond_thr": float((np.abs(slope[trn_cells][~msk]) > thr).mean()),
                           "dist_to_kept": dd}
    log(f"C3 masks: {json.dumps(res['masks'])}")
    # truth variance and noise: beyond vs inside
    Ta, Tb = est.T["dev"]
    res["truth"] = {}
    for g, cells in groups.items():
        for sname in ("sedentary", "all"):
            msk = est.strata[sname]
            A_, B_ = Ta[cells][:, msk], Tb[cells][:, msk]
            Ac, Bc = A_ - A_.mean(0), B_ - B_.mean(0)
            sig = float((Ac * Bc).mean())
            noise = float((((Ac - Bc) / 2) ** 2).mean())        # variance of the half-mean noise ~ /2 of a half
            res["truth"][f"{g}|{sname}"] = {"signal_var": sig, "half_noise_var": float((((Ac - Bc)) ** 2).mean() / 2),
                                            "mean_noise_var": noise,
                                            "reliability_mean": sig / (sig + noise) if sig + noise > 0 else None}
    log(f"C3 truth: {json.dumps(res['truth'])}")
    # readouts: vcc_h50 and vcz[ctrl] restricted to each mask's kept training cells
    zw = z - zbar
    arms = {"vcc_h50": ad.vc_blocks(cov.w(50, "no_clock"), ctx), "vcz": ad.vc_blocks(zw, ctx)}
    tc_set = {int(c): i for i, c in enumerate(tc_id)}
    row_tc = np.array([tc_set[int(c)] for c in (keys[trn, 0].astype(np.int64) * 100000 + keys[trn, 1])])
    P = {}
    for name, blocks in arms.items():
        Fe = np.hstack([est.delta(B) for B in blocks])
        for mk in ("full",) + tuple(masks):
            rows = trn if mk == "full" else trn[~masks[mk][row_tc]]
            t = time.time()
            P[f"{name}|{mk}"] = Fe @ ad.fit_blocks(blocks, Yw, rows)["coef"].T
            log(f"C3 fit {name}|{mk}: {len(rows)} rows, {time.time() - t:.0f}s")
        save("C3_partial", {"done": list(P)})
    # sanity: the cover/rand6 restriction rows equal holdout3's
    assert len(trn[~masks["cover"][row_tc]]) == len(trn_rest) and len(trn[~masks["rand6"][row_tc]]) == len(trn_rand)
    res["scores"] = {}
    for g, cells in groups.items():
        for sname in ("sedentary", "all"):
            msk = est.strata[sname]
            D = {k: est.draws(v, "dev", msk, cells) for k, v in P.items()}
            o = {"arms": {k: arm(v) for k, v in D.items()}, "penalty_vs_full": {}}
            for k in P:
                nm, mk = k.split("|")
                if mk != "full":
                    dd = diff(D[k], D[f"{nm}|full"])
                    if mk in ("cover", "rand6", "twin", "region18_s0"):
                        dd.update(species_detail(est, P[k], P[f"{nm}|full"], msk, cells))
                        # leave-one-6x6-block-out over the evaluated cells
                        bl = (est.cells_ev[cells, 0] // 6) * 100000 + est.cells_ev[cells, 1] // 6
                        vals = []
                        for b in np.unique(bl):
                            keep = cells[bl != b]
                            vals.append(pooled(terms(est, P[k], msk, keep))[0]
                                        - pooled(terms(est, P[f"{nm}|full"], msk, keep))[0])
                        dd["block_jackknife_corr"] = {"min": float(np.min(vals)), "max": float(np.max(vals)),
                                                      "n_blocks": int(len(vals))}
                    o["penalty_vs_full"][k] = dd
            # cover-specific vs each control
            for nm in arms:
                for ctl in ("rand6", "twin") + tuple(k for k in masks if k.startswith("region18")):
                    if f"{nm}|{ctl}" in D:
                        o["penalty_vs_full"][f"{nm}|cover - {nm}|{ctl}"] = diff(D[f"{nm}|cover"], D[f"{nm}|{ctl}"])
            res["scores"][f"{g}|{sname}"] = o
            log(f"== C3 {g}|{sname}: " + "  ".join(f"{k} {v['corr']:.3f}/k{v['k']:.2f}" for k, v in o["arms"].items()))
            for k, v in o["penalty_vs_full"].items():
                extra = ""
                if "jackknife" in v:
                    extra = (f" sp-jk [{v['jackknife']['corr_min']:+.3f},{v['jackknife']['corr_max']:+.3f}] blk-jk "
                             f"[{v['block_jackknife_corr']['min']:+.3f},{v['block_jackknife_corr']['max']:+.3f}] "
                             f"eqw {v['equal_weight']['corr']:+.3f}")
                log(f"    {k:40s} corr {v['corr']:+.3f} [{v['corr_ci'][0]:+.3f},{v['corr_ci'][1]:+.3f}] mse {v['mse']:+.3f}"
                    + extra)
    save("C3", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("c1", "c2", "c3"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--safe-inv", action="store_true")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    torch.set_num_threads(4)
    if a.safe_inv:
        install_safe_inv()
    {"c1": run_c1, "c2": run_c2, "c3": run_c3}[a.part](a)
    log("done")


if __name__ == "__main__":
    main()
