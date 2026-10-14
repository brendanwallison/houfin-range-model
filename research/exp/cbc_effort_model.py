"""E064: circle-level effort model for raw CBC counts (negative binomial, penalized), in the Link & Sauer tradition.

    python research/exp/cbc_effort_model.py --data cbc_circles.npz --effort ph|counters --out fit.npz [--holdout 0.1]

Per species s (all species fitted jointly, parameters separate), for circle c in stratum r and winter t:
    log mu = a[c,s] + g[r,t,s] + p[s] * log(E / mean E) + f[s] * log1p(feeder_hrs) + q[s] * log1p(nocturnal_hrs)
    count ~ NegBin2(mu, theta[s])
E = party-hours (--effort ph; circle-years without party-hours are dropped -- nearly all before 1940) or field counters
(--effort counters; available for ~100% of circle-years since 1901). The power function of effort, species-specific,
follows Link & Sauer 1999 (diminishing returns when p < 1). Penalties stand in for the usual random effects: a ridge on
the circle effects a (sd ~3), a first-order random walk on g over years within each stratum (sd ~0.3 per year, so
years without counts borrow from their neighbours), and a weak ridge on g. Fitted by Adam on the full data.
The STRATUM INDEX of a species is exp(g[r,t,s]) x mean over the stratum's circles of exp(a[c,s]): the expected count of
a typical circle in that stratum at mean effort, no feeder or nocturnal effort -- Audubon's definition of its index.
A random --holdout share of circle-years is left out of the fit and scored afterwards (predictive check by era).
E065 adds: --completeness listlen|bench (+ k[s] * a leave-one-out list-completeness covariate, measured against the
circle's own mean over its modern counts, --ref-from), --family bern (detection model, cloglog link) and --half 0|1
(fit one random half of the circles, for split-half reliability).
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--effort", choices=["ph", "counters"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--sd-a", type=float, default=3.0)
    ap.add_argument("--sd-rw", type=float, default=0.3)
    ap.add_argument("--sd-g", type=float, default=5.0)
    ap.add_argument("--thin-post1966-to-early", action="store_true")
    # E065 options
    ap.add_argument("--completeness", choices=["none", "listlen", "bench"], default="none",
                    help="sampling-completeness covariate, leave-one-out per species (needs --completeness-file)")
    ap.add_argument("--completeness-file", default=None)
    ap.add_argument("--ref-from", type=int, default=1966,
                    help="completeness is measured against each circle's own mean over its counts from this year on")
    ap.add_argument("--family", choices=["nb", "bern"], default="nb",
                    help="nb: counts (negative binomial); bern: detection (presence/absence, cloglog link)")
    ap.add_argument("--p-fixed", type=float, default=None,
                    help="E068: one effort exponent for every species, not fitted")
    ap.add_argument("--persistent-windows", type=int, nargs=4, default=[1910, 1939, 1950, 1965],
                    help="E070: the two windows (y0 y1 y2 y3) a persistent circle must have counts in")
    ap.add_argument("--effort2", choices=["none", "distance"], default="none",
                    help="E070: + q2[s] * centred log party-distance where recorded, + m2[s] where it is missing")
    ap.add_argument("--circle-prior", choices=["ridge", "neighbour"], default="ridge",
                    help="E070: circle effects shrink toward 0 (ridge, sd --sd-a) or toward the mean of their --k-nbr "
                         "nearest circles within --nbr-km (sd --sd-nbr; plus the ridge)")
    ap.add_argument("--k-nbr", type=int, default=5); ap.add_argument("--nbr-km", type=float, default=150.0)
    ap.add_argument("--sd-nbr", type=float, default=1.0)
    ap.add_argument("--persistent", type=int, default=0,
                    help="E068c: fit only circles with >= N counts in 1910-39 and >= N in 1950-65")
    ap.add_argument("--half", type=int, choices=[-1, 0, 1], default=-1,
                    help="fit only circles in a random half (split-half reliability); -1 = all circles")
    a = ap.parse_args()
    import torch
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() and os.environ.get("CBC_DEVICE") == "cuda" else "cpu"
    D = np.load(a.data, allow_pickle=False)
    Y = D["Y"]
    E = D["party_hours"] if a.effort == "ph" else D["field_counters"]
    ok = np.isfinite(E) & (E > 0)
    rng = np.random.default_rng(a.seed)
    hold = rng.random(len(Y)) < a.holdout
    fit = ok & ~hold
    yr = D["year_end"]
    if a.thin_post1966_to_early:
        # E064b: keep pre-1966 rows; thin post-1966 FIT rows so each stratum has, per decade, as many circle-years as
        # it had per decade before 1940 (strata with none before 1940 keep none after 1966). Held-out rows untouched.
        sr = D["stratum_idx"]
        early = ok & (yr < 1940)
        per_dec = np.bincount(sr[early], minlength=len(D["strata"])) / 4.0           # 1901-1939: ~4 decades
        late = np.where(fit & (yr >= 1966))[0]
        keep = np.zeros(len(Y), bool)
        dec = yr // 10
        for (r, d), rows in __import__("pandas").Series(late).groupby([sr[late], dec[late]]):
            k = int(rng.binomial(1, per_dec[r] % 1)) + int(per_dec[r])
            if k > 0:
                keep[rng.choice(rows.to_numpy(), size=min(k, len(rows)), replace=False)] = True
        fit = fit & ((yr < 1966) | keep)
        print(f"[E064b] thinned post-1966 fit rows: {int((fit & (yr >= 1966)).sum()):,} kept of {len(late):,}", flush=True)
    nC = len(D["circles"])
    if a.persistent > 0:
        # E068c: only circles with >= N counts in 1910-1939 AND >= N in 1950-1965 (the windows of E065's C check)
        ci_ = D["circle_idx"]
        w0, w1, w2, w3 = a.persistent_windows
        n1 = np.bincount(ci_[ok & (yr >= w0) & (yr <= w1)], minlength=nC)
        n2 = np.bincount(ci_[ok & (yr >= w2) & (yr <= w3)], minlength=nC)
        keep_c = (n1 >= a.persistent) & (n2 >= a.persistent)
        fit = fit & keep_c[ci_]
        print(f"[E068c] persistent circles (>= {a.persistent} counts in both windows): {int(keep_c.sum())}; "
              f"{int(fit.sum()):,} fit rows", flush=True)
    if a.half >= 0:
        in_half = np.random.default_rng(a.seed + 1).random(nC) < 0.5
        fit = fit & (in_half[D["circle_idx"]] == bool(a.half))
        print(f"[E065] half {a.half}: {int(fit.sum()):,} fit rows", flush=True)
    years = np.arange(yr.min(), yr.max() + 1)
    T = len(years)
    nR, S = len(D["strata"]), Y.shape[1]
    # completeness covariate, leave-one-out per species (the target species' own detection removed)
    Zc = np.zeros(Y.shape, dtype=np.float32)
    if a.completeness != "none":
        Cf = np.load(a.completeness_file, allow_pickle=False)
        det = (Y > 0).astype(np.float32)
        if a.completeness == "listlen":
            Zc = np.log(np.maximum(Cf["list_len"][:, None] - det, 1.0))
        else:
            inb = Cf["in_bench"][D["circle_idx"]].astype(np.float32)
            num = Cf["bench_hit"][:, None] - det * inb
            den = Cf["bench_n"][:, None] - inb
            b = np.where(den > 0, num / np.maximum(den, 1), np.nan)
            b = np.clip(b, 0.02, 0.98)
            Zc = np.log(b / (1 - b))
        Zc = np.where(np.isfinite(Zc), Zc, np.nan)
        # RELATIVE TO THE SITE: subtract the circle's own mean over its modern (>= --ref-from) fit rows, so a richer
        # site does not read as a more complete one. Circles with < 3 modern rows borrow the nearest circle's
        # reference within ~1 degree, else their stratum's median reference.
        ci_ = D["circle_idx"]
        mod = fit & (yr >= a.ref_from)
        ssum = np.zeros((nC, S)); scnt = np.zeros((nC, S))
        fin = np.isfinite(Zc) & mod[:, None]
        np.add.at(ssum, ci_, np.where(fin, Zc, 0.0)); np.add.at(scnt, ci_, fin)
        ref = np.where(scnt >= 3, ssum / np.maximum(scnt, 1), np.nan)
        has = ~np.isnan(ref).all(1)
        from scipy.spatial import cKDTree
        clat = np.zeros(nC); clon = np.zeros(nC); clat[ci_] = D["lat"]; clon[ci_] = D["lon"]
        cstr = np.zeros(nC, int); cstr[ci_] = D["stratum_idx"]
        own = int(has.sum()); hv = np.where(has)[0]
        d, j = cKDTree(np.column_stack([clat[hv], clon[hv]])).query(np.column_stack([clat, clon]))
        borrow = ~has & (d <= 1.0)
        ref[borrow] = ref[hv[j[borrow]]]
        for r in range(nR):
            m = (cstr == r) & has
            fill = (cstr == r) & np.isnan(ref).all(1)
            if m.any() and fill.any():
                ref[fill] = np.nanmedian(ref[m], axis=0)
        ref = np.where(np.isnan(ref), np.nanmedian(ref, axis=0), ref)
        Zc = np.where(np.isfinite(Zc), Zc - ref[ci_], 0.0).astype(np.float32)  # missing -> site-typical (no adjustment)
        print(f"[E065] completeness '{a.completeness}' relative to modern site mean: {own} circles own reference, "
              f"{int(borrow.sum())} borrowed, rest stratum; covariate sd {float(Zc[fit].std()):.2f}; "
              f"median by era " + ", ".join(f"{lab} {float(np.median(Zc[fit & (yr >= y0) & (yr < y1)])):+.2f}"
                                             for lab, y0, y1 in [("<1940", 0, 1940), ("1940-65", 1940, 1966),
                                                                 ("1966+", 1966, 9999)]), flush=True)
    logE = np.log(np.where(ok, E, 1.0)); logE = logE - logE[fit].mean()
    feed = np.log1p(np.clip(np.nan_to_num(D["feeder_hrs"]), 0, None))     # one record is negative (entry error)
    if a.effort2 == "distance":
        km = D["distance_km"]; has = np.isfinite(km) & (km > 0)
        logD = np.log(np.where(has, km, 1.0)); logD = np.where(has, logD - logD[fit & has].mean(), 0.0)
        missD = (~has).astype(float)
        print(f"[E070] distance recorded for {has[fit].mean():.2f} of fit rows", flush=True)
    else:
        logD = np.zeros(len(Y)); missD = np.zeros(len(Y))
    noct = np.log1p(np.clip(np.nan_to_num(D["nocturnal_hrs"]), 0, None))
    tt = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    idx = np.where(fit)[0]
    c_, r_, t_ = tt(D["circle_idx"][idx], torch.long), tt(D["stratum_idx"][idx], torch.long), tt(yr[idx] - years[0], torch.long)
    x_e, x_f, x_n, y = tt(logE[idx]), tt(feed[idx]), tt(noct[idx]), tt(Y[idx])
    x_d, x_m = tt(logD[idx]), tt(missD[idx])
    z_c = tt(Zc[idx])
    yb = (y > 0).float()
    # circle-years by stratum-year presence: random-walk penalty only links years, data or not
    A = torch.zeros(nC, S, device=dev, requires_grad=True)
    nbr = None
    if a.circle_prior == "neighbour":
        from scipy.spatial import cKDTree
        clat = np.zeros(nC); clon = np.zeros(nC)
        clat[D["circle_idx"]] = D["lat"]; clon[D["circle_idx"]] = D["lon"]
        xy = np.column_stack([clat * 111.0, clon * 111.0 * np.cos(np.radians(clat))])
        dd, jj = cKDTree(xy).query(xy, k=a.k_nbr + 1)
        dd, jj = dd[:, 1:], jj[:, 1:]
        wm = (dd <= a.nbr_km).astype(np.float32)
        has_n = wm.sum(1) > 0
        nbr = (tt(jj, torch.long), tt(wm / np.maximum(wm.sum(1, keepdims=True), 1)), tt(has_n.astype(np.float32)))
        print(f"[E070] neighbour prior: {int(has_n.sum())} of {nC} circles have a neighbour within {a.nbr_km} km",
              flush=True)
    G = torch.zeros(nR, T, S, device=dev)
    if a.family == "nb":
        mean_log = torch.log(y.mean(0) + 0.1)
    else:                                                         # cloglog of the mean detection rate
        pbar = yb.mean(0).clamp(0.01, 0.99)
        mean_log = torch.log(-torch.log1p(-pbar))
    G += mean_log
    G.requires_grad_(True)
    P = torch.full((S,), 0.5 if a.p_fixed is None else a.p_fixed, device=dev, requires_grad=a.p_fixed is None)
    F = torch.zeros(S, device=dev, requires_grad=True)
    Q = torch.zeros(S, device=dev, requires_grad=True)
    K = torch.zeros(S, device=dev, requires_grad=True)            # completeness coefficient
    LT = torch.zeros(S, device=dev, requires_grad=True)           # log theta
    Q2 = torch.zeros(S, device=dev, requires_grad=True)           # distance exponent
    M2 = torch.zeros(S, device=dev, requires_grad=True)           # distance-missing offset
    opt = torch.optim.Adam([A, G, F, Q, K, LT, Q2, M2] + ([P] if a.p_fixed is None else []), lr=a.lr)
    n = float(len(idx))
    t0 = time.time()
    for step in range(a.steps):
        opt.zero_grad()
        eta = (A[c_] + G[r_, t_] + P * x_e[:, None] + F * x_f[:, None] + Q * x_n[:, None] + K * z_c
               + Q2 * x_d[:, None] + M2 * x_m[:, None])
        if a.family == "nb":
            th = torch.exp(LT)
            # NB2 log-likelihood (theta = size)
            ll = (torch.lgamma(y + th) - torch.lgamma(th) - torch.lgamma(y + 1) + th * (LT - torch.logaddexp(LT, eta))
                  + y * (eta - torch.logaddexp(LT, eta)))
        else:
            lam = torch.exp(eta.clamp(max=15.0))                  # cloglog: P(detect) = 1 - exp(-lam)
            ll = yb * torch.log(-torch.expm1(-lam) + 1e-12) - (1 - yb) * lam
        nll = -ll.sum() / n
        if nbr is not None:
            jn, wn, hn = nbr
            nb_mean = (A[jn] * wn[:, :, None]).sum(1)
            pen_nb = (((A - nb_mean) ** 2).sum(1) * hn).sum() / (2 * a.sd_nbr ** 2)
        else:
            pen_nb = 0.0
        pen = (pen_nb + (A ** 2).sum() / (2 * a.sd_a ** 2) + ((G[:, 1:] - G[:, :-1]) ** 2).sum() / (2 * a.sd_rw ** 2)
               + ((G - mean_log) ** 2).sum() / (2 * a.sd_g ** 2)) / n
        loss = nll + pen
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"[E064 {a.effort}] step {step} nll/row {float(nll) / S:.4f} pen {float(pen):.4f} "
                  f"p median {float(P.median()):.2f} ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        Ah, Gh = A.cpu().numpy(), G.cpu().numpy()
        Pn, Fn, Qn, th = P.cpu().numpy(), F.cpu().numpy(), Q.cpu().numpy(), np.exp(LT.cpu().numpy())
        Kn = K.cpu().numpy(); Q2n, M2n = Q2.cpu().numpy(), M2.cpu().numpy()
    # stratum index at mean effort and mean completeness: NB -> expected count of a typical circle,
    # i.e. exp(g) x mean over the stratum's circles of exp(a); bern -> mean detection probability over its circles
    circ_str = np.full(nC, -1)
    circ_str[D["circle_idx"]] = D["stratum_idx"]
    index = np.zeros((nR, T, S))
    # only circles the fit saw (E068c): an unfitted circle's effect stays ~0, which only rescales the count index
    # (log changes unaffected) but shifts the detection index nonlinearly (E065's V3 half-fits included them)
    seen = np.bincount(D["circle_idx"][fit], minlength=nC) > 0
    for r in range(nR):
        m = (circ_str == r) & seen
        if not m.any():
            continue
        if a.family == "nb":
            index[r] = np.exp(Gh[r]) * np.exp(Ah[m]).mean(0)[None, :]
        else:
            index[r] = (1 - np.exp(-np.exp(np.clip(Gh[r][None] + Ah[m][:, None, :], None, 15)))).mean(0)
    # support: circle-years used in the fit per stratum-year
    support = np.zeros((nR, T), dtype=np.int32)
    np.add.at(support, (D["stratum_idx"][fit], yr[fit] - years[0]), 1)
    # held-out predictions (rows with effort only)
    hidx = np.where(ok & hold)[0]
    if a.half >= 0:
        hidx = hidx[in_half[D["circle_idx"][hidx]] == bool(a.half)]
    eta_h = (Ah[D["circle_idx"][hidx]] + Gh[D["stratum_idx"][hidx], yr[hidx] - years[0]] + Pn * logE[hidx, None]
             + Fn * feed[hidx, None] + Qn * noct[hidx, None] + Kn * Zc[hidx] + Q2n * logD[hidx, None]
             + M2n * missD[hidx, None])
    if a.family == "nb":
        hold_pred = np.exp(eta_h)
        hold_pdet = 1 - (th / (th + hold_pred)) ** th               # P(count > 0) under NB2
    else:
        hold_pred = hold_pdet = 1 - np.exp(-np.exp(np.clip(eta_h, None, 15)))
    np.savez_compressed(a.out, index=index.astype(np.float32), support=support, years=years, strata=D["strata"],
                        species=D["species"], p=Pn, f=Fn, q=Qn, k=Kn, theta=th, hold_rows=hidx,
                        hold_pred=hold_pred.astype(np.float32), hold_pdet=hold_pdet.astype(np.float32),
                        effort=a.effort, family=a.family, completeness=a.completeness, half=a.half)
    json.dump({"effort": a.effort, "family": a.family, "completeness": a.completeness, "half": a.half,
               "k_median": float(np.median(Kn)),
               "rows_fit": int(fit.sum()), "rows_holdout": int(len(hidx)), "species": int(S),
               "p_median": float(np.median(Pn)), "p_q10_q90": [float(np.quantile(Pn, 0.1)), float(np.quantile(Pn, 0.9))],
               "seconds": time.time() - t0}, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(f"[E064 {a.effort}] done: {int(fit.sum()):,} rows fitted, {len(hidx):,} held out; effort exponent p median "
          f"{np.median(Pn):.2f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
