"""SDM baseline plan D2: the ONE scorer for every model on the age model's held-out sets.

NB2 log density per observation (map_diagnostics conventions), plug-in / marginal / conditional over an
observer x route lognormal effect, and the strata of SDM_BASELINE_B0_2026-10-30.md. Pure NumPy/SciPy.

    python sdm_b0_score.py --design <dir>        # parity gate: re-score step 3 from its own predictive
"""
import argparse
import json
import os

import numpy as np
from scipy.special import gammaln, logsumexp

REGIONS = (("west", -180, -105), ("plains", -105, -95), ("midwest", -95, -85), ("east", -85, 0))
ERAS = (("1966-79", 1966, 1979), ("1980-93", 1980, 1993), ("1994-07", 1994, 2007), ("2008-20", 2008, 2020))
TARGET = {"T_plug": -1.8435745163552493, "T_marg": -1.8439029402496867, "T_cond": -1.7311423008466358,
          "S_plug": -1.492528352992277, "S_marg": -1.2194000843297628}


def nb_lp(y, mu, phi, gate=None):
    """NB2 log pmf; optional route-occupancy gate (occupied routes NB with mean mu / (1 - gate))."""
    mu = np.maximum(mu, 1e-12)
    mu_nb = mu / np.maximum(1.0 - gate, 1e-300) if gate is not None else mu
    lp = (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1.0)
          + phi * np.log(phi / (phi + mu_nb)) + y * np.log(mu_nb / (phi + mu_nb)))
    if gate is not None:
        lp = np.where(y == 0, np.logaddexp(np.log(np.maximum(gate, 1e-300)), np.log1p(-gate) + lp), np.log1p(-gate) + lp)
    return lp


def integrated_lp(y, mu0, phi, sigma, nodes=24, gate_fn=None):
    """log of E_x[NB(y | mu0 exp(sigma x), phi)], x ~ N(0,1), Gauss-Hermite (as map_diagnostics)."""
    t, w = np.polynomial.hermite.hermgauss(nodes)
    x, logw = np.sqrt(2.0) * t, np.log(w / np.sqrt(np.pi))
    lps = []
    for xk in x:
        mk = np.maximum(mu0 * np.exp(sigma * xk), 1e-12)
        lps.append(nb_lp(y, mk, phi, None if gate_fn is None else gate_fn(mk)))
    return logsumexp(np.stack(lps) + logw[:, None], axis=0)


def per_obs_lp(design, mu, phi, mu_marg=None, sigma=0.0, gate=None, gate_fn=None):
    """Per-observation plug-in, marginal and conditional log densities (all rows).

    mu: the model's plug-in predictive mean (with whatever observer knowledge it has).
    mu_marg, sigma: the observer-free mean and the observer-route effect sd to integrate; None -> marginal = plug-in.
    Conditional = marginal for real observer-routes never fitted, plug-in otherwise (map_diagnostics definition).
    """
    y = design["y"]
    phi = np.broadcast_to(np.asarray(phi, float), y.shape)
    plug = nb_lp(y, mu, phi, gate)
    if mu_marg is None or sigma <= 0:
        marg = plug.copy()
    else:
        marg = plug.copy()
        real = design["obsroute"] >= 0
        hm = real & design["holdout"]              # only held-out rows are needed; fitted rows keep plug-in
        marg[hm] = integrated_lp(y[hm], mu_marg[hm], phi[hm], sigma, gate_fn=gate_fn)
    cond = np.where(unseen_mask(design), marg, plug)
    return {"plug": plug, "marg": marg, "cond": cond}


def unseen_mask(design):
    idx = design["obsroute"]
    fitted = ~design["holdout"]
    seen = np.zeros(int(idx.max()) + 2, bool)
    seen[idx[fitted & (idx >= 0)]] = True
    return (idx >= 0) & ~seen[np.maximum(idx, 0)]


def sets(design):
    T = design["temporal"]
    S = design["spatial"]
    return {"T": T, "S_all": S, "S_real": S & ~design["pseudo"]}


def summarise(design, lp, mu, strata=True):
    """Mean log density per set (+ strata), log1p RMSE / correlation of mu."""
    y = design["y"]
    out = {}
    unseen = unseen_mask(design)
    for sname, m in sets(design).items():
        r = {"n": int(m.sum())}
        for kind, v in lp.items():
            r[kind] = float(v[m].mean())
        r["log1p_rmse"] = float(np.sqrt(np.mean((np.log1p(y[m]) - np.log1p(mu[m])) ** 2)))
        r["log1p_corr"] = float(np.corrcoef(np.log1p(y[m]), np.log1p(mu[m]))[0, 1]) if m.sum() > 2 else None
        if strata:
            reg = {}
            for rn, lo, hi in REGIONS:
                mm = m & (design["lon"] >= lo) & (design["lon"] < hi)
                if mm.any():
                    reg[rn] = {"n": int(mm.sum()), **{k: float(v[mm].mean()) for k, v in lp.items()}}
            r["by_region"] = reg
            if sname == "T":
                r["seen_obsroute"] = {k: float(v[m & ~unseen].mean()) for k, v in lp.items()}
                r["unseen_obsroute"] = {k: float(v[m & unseen].mean()) for k, v in lp.items()}
                r["n_unseen"] = int((m & unseen).sum())
            if sname == "S_real":
                er = {}
                for rn, lo, hi in REGIONS:
                    for en, a, b in ERAS:
                        mm = m & (design["lon"] >= lo) & (design["lon"] < hi) & (design["year"] >= a) & (design["year"] <= b)
                        if mm.sum() >= 50:
                            er[f"{rn}|{en}"] = {"n": int(mm.sum()), **{k: float(v[mm].mean()) for k, v in lp.items()}}
                r["by_region_era"] = er
        out[sname] = r
    return out


def block_ids(design, block_cells=4):
    return (design["row"] // block_cells) * 100000 + design["col"] // block_cells


def paired_bootstrap(design, diff, mask, n=1000, seed=0):
    """90% interval of the mean paired difference, resampling 216 km blocks (4 x 4 cells at 54 km)."""
    b = block_ids(design)[mask]
    d = diff[mask]
    ub, inv = np.unique(b, return_inverse=True)
    s = np.bincount(inv, d, minlength=ub.size)
    c = np.bincount(inv, minlength=ub.size).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, ub.size, size=(n, ub.size))
    est = s[draws].sum(1) / c[draws].sum(1)
    return {"mean": float(d.mean()), "lo90": float(np.percentile(est, 5)), "hi90": float(np.percentile(est, 95)),
            "n_blocks": int(ub.size)}


def load_design(d):
    z = np.load(os.path.join(d, "design.npz"))
    return {k: z[k] for k in z.files}


def step3_lp(design, pred):
    mu = pred["mu"]
    gate = pred["gate"] if bool(pred["has_gate"]) else None
    gate_fn = None
    if gate is not None:
        raise SystemExit("step 3 has a route-occupancy gate: integrate it via route_occupancy_parts (not implemented)")
    mu0 = mu * np.exp(-pred["eps_part"])            # observer effect removed, first-year term kept
    return per_obs_lp(design, mu, pred["phi"], mu_marg=mu0, sigma=float(pred["sigma_u"]), gate=gate, gate_fn=gate_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    a = ap.parse_args()
    design = load_design(a.design)
    pr = np.load(os.path.join(a.design, "step3_pred.npz"))
    pred = {k: pr[k] for k in pr.files}
    lp = step3_lp(design, pred)
    T, S = design["temporal"], design["spatial"]
    got = {"T_plug": lp["plug"][T].mean(), "T_marg": lp["marg"][T].mean(), "T_cond": lp["cond"][T].mean(),
           "S_plug": lp["plug"][S].mean(), "S_marg": lp["marg"][S].mean()}
    diff = {k: float(got[k] - TARGET[k]) for k in TARGET}
    ok = all(abs(v) < 1e-3 for v in diff.values())
    res = {"gate_pass": ok, "got": {k: float(v) for k, v in got.items()}, "target": TARGET, "diff": diff,
           "n_T": int(T.sum()), "n_S": int(S.sum()),
           "step3_summary": summarise(design, lp, pred["mu"])}
    json.dump(res, open(os.path.join(a.design, "parity_gate.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("gate_pass", "got", "diff", "n_T", "n_S")}, indent=1))
    if not ok:
        raise SystemExit("PARITY GATE FAILED")


if __name__ == "__main__":
    main()
