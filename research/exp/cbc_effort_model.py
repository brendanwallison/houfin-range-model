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
    years = np.arange(yr.min(), yr.max() + 1)
    T = len(years)
    nC, nR, S = len(D["circles"]), len(D["strata"]), Y.shape[1]
    logE = np.log(np.where(ok, E, 1.0)); logE = logE - logE[fit].mean()
    feed = np.log1p(np.clip(np.nan_to_num(D["feeder_hrs"]), 0, None))     # one record is negative (entry error)
    noct = np.log1p(np.clip(np.nan_to_num(D["nocturnal_hrs"]), 0, None))
    tt = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    idx = np.where(fit)[0]
    c_, r_, t_ = tt(D["circle_idx"][idx], torch.long), tt(D["stratum_idx"][idx], torch.long), tt(yr[idx] - years[0], torch.long)
    x_e, x_f, x_n, y = tt(logE[idx]), tt(feed[idx]), tt(noct[idx]), tt(Y[idx])
    # circle-years by stratum-year presence: random-walk penalty only links years, data or not
    A = torch.zeros(nC, S, device=dev, requires_grad=True)
    G = torch.zeros(nR, T, S, device=dev)
    mean_log = torch.log(y.mean(0) + 0.1)
    G += mean_log
    G.requires_grad_(True)
    P = torch.full((S,), 0.5, device=dev, requires_grad=True)
    F = torch.zeros(S, device=dev, requires_grad=True)
    Q = torch.zeros(S, device=dev, requires_grad=True)
    LT = torch.zeros(S, device=dev, requires_grad=True)           # log theta
    opt = torch.optim.Adam([A, G, P, F, Q, LT], lr=a.lr)
    n = float(len(idx))
    t0 = time.time()
    for step in range(a.steps):
        opt.zero_grad()
        eta = A[c_] + G[r_, t_] + P * x_e[:, None] + F * x_f[:, None] + Q * x_n[:, None]
        th = torch.exp(LT)
        # NB2 log-likelihood (theta = size)
        ll = (torch.lgamma(y + th) - torch.lgamma(th) - torch.lgamma(y + 1) + th * (LT - torch.logaddexp(LT, eta))
              + y * (eta - torch.logaddexp(LT, eta)))
        nll = -ll.sum() / n
        pen = ((A ** 2).sum() / (2 * a.sd_a ** 2) + ((G[:, 1:] - G[:, :-1]) ** 2).sum() / (2 * a.sd_rw ** 2)
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
    # stratum index: exp(g) x mean over the stratum's circles of exp(a)
    circ_str = np.full(nC, -1)
    circ_str[D["circle_idx"]] = D["stratum_idx"]
    ea = np.zeros((nR, S))
    for r in range(nR):
        m = (circ_str == r)
        if m.any():
            ea[r] = np.exp(Ah[m]).mean(0)
    index = np.exp(Gh) * ea[:, None, :]
    # support: circle-years used in the fit per stratum-year
    support = np.zeros((nR, T), dtype=np.int32)
    np.add.at(support, (D["stratum_idx"][fit], yr[fit] - years[0]), 1)
    # held-out predictions (rows with effort only)
    hidx = np.where(ok & hold)[0]
    eta_h = (Ah[D["circle_idx"][hidx]] + Gh[D["stratum_idx"][hidx], yr[hidx] - years[0]] + Pn * logE[hidx, None]
             + Fn * feed[hidx, None] + Qn * noct[hidx, None])
    np.savez_compressed(a.out, index=index.astype(np.float32), support=support, years=years, strata=D["strata"],
                        species=D["species"], p=Pn, f=Fn, q=Qn, theta=th, hold_rows=hidx, hold_pred=np.exp(eta_h).astype(np.float32),
                        effort=a.effort)
    json.dump({"effort": a.effort, "rows_fit": int(fit.sum()), "rows_holdout": int(len(hidx)), "species": int(S),
               "p_median": float(np.median(Pn)), "p_q10_q90": [float(np.quantile(Pn, 0.1)), float(np.quantile(Pn, 0.9))],
               "seconds": time.time() - t0}, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(f"[E064 {a.effort}] done: {int(fit.sum()):,} rows fitted, {len(hidx):,} held out; effort exponent p median "
          f"{np.median(Pn):.2f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
