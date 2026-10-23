"""E069b data step: CBC lists of the community's RESIDENT species, with a fixed z -> species readout, for DESK's
presence-only term (desk.aux_presence).

    python research/exp/build_cbc_presence.py --cache <research cache> --cbc cbc_circles_comm.npz
        --similarity species_similarity.csv --out cbc_presence.npz

Species: the reference (community) species CBC carries after name harmonization that are RESIDENT (AVONET migration
1-2) -- the sedentary focus; dev species are never included (they are the evaluation targets), nor sealed species or
the House Finch. Lists: every circle-year from 1940 (the DESK window start) to 2024 on the model grid with >= 2 listed
species. Readout: ridge from the cache's ESK z (esk_annual, the observed communities' projection, i.e. the space DESK's
stabilizing term targets) to log1p BBS abundance of those species, on training cell-years from trained years (>= 1996),
lambda by 5-fold CV over spatial blocks. A LEVEL readout, as the age model's. Saved: year, row, col, circle, P (lists x S),
W (L x S), b (S), species. Held-out, buffer and dropped cells are removed by the trainer, which knows them.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e069_presence_score import ridge_cv  # noqa: E402


def winter_operator(a, keys, esk, spl, D, P, yr, rel):
    """E086x: the WINTER observation operator z -> CBC presence, fitted where both seasons are observed and then
    frozen: eta = z W + b (+ comp * gamma), loss = the trainer's -sum(P * log_softmax(eta)) / sum(P), on lists at
    TRAINING cells in --fit-years whose cell-year has observed ESK coordinates; L2 on W chosen by held-out likelihood
    over 6x6 spatial blocks (80 / 20 split). gamma is fitted alongside (so W does not absorb completeness) but not
    saved -- the trainer learns its own. Returns W (L x S), b (S), lambda."""
    import torch
    kidx = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(keys) if spl["is_train"][i]}
    m = np.array([kidx.get((int(r), int(c), int(y)), -1) for r, c, y in zip(D["row"], D["col"], yr)])
    use = np.where((m >= 0) & (yr >= a.fit_years[0]) & (yr <= a.fit_years[1]) & (P.sum(1) >= 2))[0]
    Z = torch.as_tensor(esk[m[use]], dtype=torch.float64); Pt = torch.as_tensor(P[use], dtype=torch.float64)
    cp = torch.as_tensor(np.nan_to_num(rel[use]) if rel is not None else np.zeros(len(use)), dtype=torch.float64)
    blk = (D["row"][use] // 6) * 1000 + D["col"][use] // 6
    ub = np.unique(blk); rng = np.random.default_rng(86)
    val = np.isin(blk, rng.choice(ub, max(1, len(ub) // 5), replace=False))
    L, S = Z.shape[1], Pt.shape[1]
    def fit(idx, lam):
        W = torch.zeros(L, S, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(S, dtype=torch.float64, requires_grad=True); g = torch.zeros(S, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([W, b, g], max_iter=500, line_search_fn="strong_wolfe")
        def nll(ii, W, b, g):
            eta = Z[ii] @ W + b + cp[ii, None] * g
            return -(torch.log_softmax(eta, -1) * Pt[ii]).sum() / Pt[ii].sum()
        def closure():
            opt.zero_grad(); loss = nll(idx, W, b, g) + lam * (W ** 2).sum(); loss.backward(); return loss
        opt.step(closure)
        return W.detach(), b.detach(), g.detach(), nll
    tr_i, va_i = torch.as_tensor(np.where(~val)[0]), torch.as_tensor(np.where(val)[0])
    best = None
    for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        W, b, g, nll = fit(tr_i, lam)
        v = float(nll(va_i, W, b, g))
        print(f"[E086x] winter operator lambda {lam:g}: held-out list nll {v:.5f}", flush=True)
        if best is None or v < best[0]:
            best = (v, lam)
    W, b, _, nll = fit(torch.arange(len(use)), best[1])
    base = -(torch.log_softmax(torch.zeros(S, dtype=torch.float64), -1) * Pt).sum() / Pt.sum()
    print(f"[E086x] winter operator: {len(use):,} lists ({a.fit_years[0]}-{a.fit_years[1]}, training cells with ESK z), "
          f"lambda {best[1]:g}, nll {float(nll(torch.arange(len(use)), W, b, torch.zeros(S, dtype=torch.float64))):.4f} "
          f"(uniform {float(base):.4f})", flush=True)
    return W.numpy(), b.numpy(), best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--cbc", required=True)
    ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--completeness", default=None,
                    help="E078: completeness npz aligned to --cbc rows (cbc_completeness.py); adds 'comp' per list = "
                         "logit of benchmark completeness relative to the circle's own mean over its 1966+ counts, "
                         "centred (0 where unavailable)")
    ap.add_argument("--years", type=int, nargs=2, default=[1940, 2024], help="E086v: list years kept (winter end year)")
    ap.add_argument("--comp-ref-from", type=int, default=1966,
                    help="E086v: the completeness reference is the circle's mean over counts from this year on")
    ap.add_argument("--readout-target", choices=("bbs", "cbc"), default="bbs",
                    help="E086x: bbs = the E069b ridge to BBS log1p abundance (a BREEDING readout); cbc = a WINTER "
                         "observation operator: W, b fitted by the presence term's own within-list softmax likelihood on "
                         "CBC lists of --fit-years at training cells, z = the cell-year's observed ESK coordinates")
    ap.add_argument("--fit-years", type=int, nargs=2, default=[1996, 2024],
                    help="E086x: years whose lists fit the winter operator (the era where both seasons are observed)")
    ap.add_argument("--comp-center-ref", action="store_true",
                    help="E086v: centre the covariate on reference-era lists, not on the kept lists")
    a = ap.parse_args()
    import pandas as pd
    C = a.cache
    meta = json.load(open(os.path.join(C, "meta.json")))
    keys = np.load(os.path.join(C, "keys.npy")); Xc = np.load(os.path.join(C, "X_comm.npy"))
    esk = np.load(os.path.join(C, "esk_annual.npy")).astype(float)
    sp_ = np.load(os.path.join(C, "split.npz"))
    D = np.load(a.cbc, allow_pickle=False)
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    comm, cs = list(meta["community"]), list(D["species"])
    sp = [s for s in comm if s in cs and mig.get(s) in (1.0, 2.0)]
    assert "houfin" not in sp and not set(sp) & set(meta["dev_species"]) and not set(sp) & set(meta["sealed_species"])
    tr = sp_["is_train"] & (keys[:, 2] >= 1996)
    T = np.log1p(Xc[:, [comm.index(s) for s in sp]])
    Z = esk[tr]
    _, lam = ridge_cv(Z, T[tr], sp_["block_id"][tr], [1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    mu_z, mu_t = Z.mean(0), T[tr].mean(0)
    A = (Z - mu_z).T @ (Z - mu_z)
    B = np.linalg.solve(A + lam * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), (Z - mu_z).T @ (T[tr] - mu_t))
    b = mu_t - mu_z @ B
    P = D["Y"][:, [cs.index(s) for s in sp]] > 0
    yr = D["year_end"]
    ok = (D["row"] >= 0) & (yr >= a.years[0]) & (yr <= a.years[1]) & (P.sum(1) >= 2)
    extra = {}
    if a.completeness:
        Cf = np.load(a.completeness)
        comp = np.where(Cf["bench_n"] > 0, Cf["bench_hit"] / np.maximum(Cf["bench_n"], 1), np.nan)
        comp = np.clip(comp, 0.02, 0.98); lc = np.log(comp / (1 - comp))
        ci_ = D["circle_idx"]; late = (yr >= a.comp_ref_from) & np.isfinite(lc)
        ref = np.bincount(ci_[late], np.where(late, lc, 0)[late], minlength=len(D["circles"])) / np.maximum(
            np.bincount(ci_[late], minlength=len(D["circles"])), 1)
        has = np.bincount(ci_[late], minlength=len(D["circles"])) >= 3
        rel = np.where(np.isfinite(lc) & has[ci_], lc - ref[ci_], np.nan)
        # E086v: with --years restricting the lists, centre on the REFERENCE-era lists (yr >= --comp-ref-from), as the
        # 1940-2024 build effectively did (mostly modern lists); centring on a degraded window would zero its deficit
        cen = ok if not a.comp_center_ref else ((D["row"] >= 0) & (yr >= a.comp_ref_from) & (yr <= 2024) & (P.sum(1) >= 2))
        rel = rel - np.nanmean(rel[cen])
        extra["comp"] = np.where(np.isfinite(rel), rel, 0.0)[ok].astype(np.float32)
        print(f"[E078] completeness covariate: available for {np.isfinite(rel[ok]).mean():.2f} of lists; median by decade "
              + ", ".join(f"{d}s {np.nanmedian(rel[ok & (yr // 10 * 10 == d)]):+.2f}" for d in (1940, 1960, 1980, 2000, 2020)),
              flush=True)
    if a.readout_target == "cbc":
        B, b, lam = winter_operator(a, keys, esk, sp_, D, P, yr, rel if a.completeness else None)
    np.savez_compressed(a.out, **extra, year=yr[ok].astype(np.int32), row=D["row"][ok].astype(np.int32),
                        col=D["col"][ok].astype(np.int32), circle=D["circle_idx"][ok].astype(np.int32),
                        P=P[ok], W=B.astype(np.float32), b=b.astype(np.float32), species=np.array(sp))
    dec = (yr[ok] // 10) * 10
    print(f"[E069b] {len(sp)} resident reference species; ridge lambda {lam}; {int(ok.sum()):,} lists "
          f"({int(P[ok].sum()):,} records); by decade: "
          + ", ".join(f"{d}s {int((dec == d).sum())}" for d in np.unique(dec)), flush=True)


if __name__ == "__main__":
    main()
