"""E018 (S1): does the readout know when it is guessing?

    python research/exp/readout_honesty.py --cache <dir> --out <dir> [--population withheld] [--ranks 12 24 64]
        [--species-regex RE]
    python research/exp/readout_honesty.py --summarize <out>

In the age model's readout (H = z.beta, beta pinned by modern data) a site-time's interval widens only with its
LEVERAGE -- how far its z lies outside the cloud of training z's -- so a wrong z is a confident wrong habitat unless
it leaves the cloud (the refinement of S1). Three checks:
  1. novelty: Mahalanobis distance (per dim) of DESK z from the training rows' z, and of the covariates (64 PCs) from
     the training rows' covariates, for row sets: training / held-out cells in trained years / held-out cells in
     withheld years (the backcast) / training cells in withheld years. Does covariate novelty survive into z?
  2. does leverage track error? per held-out cell, the noise-free backcast error energy of DESK's z (halves A, B)
     against the Mahalanobis distance of its backcast z (Spearman), and against covariate novelty.
  3. interval honesty for species: a level BLR (research/lib/blr; cell-epoch log1p means of dev species on DESK z,
     training cells, trained epochs) predicts each held-out cell's change with posterior variance V = dz' Ainv dz;
     R_s = sum[(pred - d_B)^2 - noise_B] / sum V over cells, per resolvable species. R ~ 1: the readout's habitat
     intervals know their error; R >> 1: confidently wrong (sqrt(R) = how much too narrow).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from lib import blr, covfeat  # noqa: E402


def mahal(X, ref):
    """Per-row Mahalanobis distance squared / dim of X against the reference rows' mean and covariance."""
    mu = ref.mean(0)
    C = np.cov(ref - mu, rowvar=False) + 1e-9 * np.eye(ref.shape[1])
    Ci = np.linalg.inv(C)
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, Ci, d) / X.shape[1]


def spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() > 5 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="withheld", choices=("trained", "withheld"))
    ap.add_argument("--ranks", type=int, nargs="+", default=(12, 24, 64))
    ap.add_argument("--species-regex", default=None)
    ap.add_argument("--lvl-epochs", default="modern", choices=("modern", "both"),
                    help="levels the readout is fitted on (training cells); modern keeps n equal across populations")
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from change_oracle import group_rows
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import change_noise
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    E, full, half_a, half_b = group_rows(a.cache, keys, withheld)
    is_train = split["is_train"].astype(bool)
    wh_year = np.isin(keys[:, 2], withheld)
    held_cell = split["holdout"][keys[:, 0], keys[:, 1]]
    row_sets = {"training": is_train, "heldout_trained_years": held_cell & ~wh_year,
                "heldout_withheld_years": held_cell & wh_year, "training_cells_withheld_years": ~held_cell & wh_year
                & ~split["buffer"][keys[:, 0], keys[:, 1]]}
    zr = np.asarray(np.load(c("z_raw.npy"), mmap_mode="r"), "float64")
    P = covfeat.cov_pcs(np.load(c("F_cov.npy"), mmap_mode="r"), is_train)
    sub = np.where(is_train)[0][:: max(1, is_train.sum() // 40000)]
    res = {"cache": a.cache, "population": a.population, "novelty": {}, "leverage_vs_error": {},
           "interval_honesty": {}}
    cov_nov = mahal(P, P[sub])
    res["novelty"]["covariates"] = {k: _q(cov_nov[m]) for k, m in row_sets.items() if m.any()}
    for r in a.ranks:
        zn = mahal(zr[:, :r], zr[sub, :r])
        res["novelty"][f"z_r{r}"] = {k: _q(zn[m]) for k, m in row_sets.items() if m.any()}

    # held-out cell pairs
    want_wh = a.population == "withheld"
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh) == want_wh:
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]]
    buf = split["buffer"][cells[:, 0], cells[:, 1]]
    tr, ev = np.where(~held & ~buf)[0], np.where(held)[0]
    gm = lambda A: np.stack([np.asarray(A[np.asarray(g)], "float64").mean(0) for g in full])
    D = gm(zr)
    Pg = gm(P)
    cov_nov_g = mahal(Pg, P[sub])
    T = {h: E[f"z_{h}"].astype("float64") for h in ("a", "b")}
    for r in a.ranks:
        e_ix = pairs[ev, 0]
        err = ((D[e_ix, :r] - T["a"][e_ix, :r]) * (D[e_ix, :r] - T["b"][e_ix, :r])).sum(1)
        lev = mahal(D[e_ix, :r], zr[sub, :r])
        res["leverage_vs_error"][str(r)] = {"spearman_error_vs_z_leverage": spearman(err, lev),
                                            "spearman_error_vs_cov_novelty": spearman(err, cov_nov_g[e_ix]),
                                            "spearman_z_leverage_vs_cov_novelty": spearman(lev, cov_nov_g[e_ix]),
                                            "median_error": float(np.median(err))}

    # interval honesty for species
    Xd = np.load(c("X_dev.npy")).astype("float64")
    Y = {h: epoch_mean_observed(Xd, g).astype("float64") for h, g in (("full", full), ("a", half_a),
                                                                       ("b", half_b))}
    keep = (Y["full"] > 0).any(0)
    if a.species_regex:
        import re
        meta = json.load(open(c("meta.json"), encoding="utf-8"))
        keep &= np.array([bool(re.search(a.species_regex, s)) for s in meta["dev_species"]])
    Y = {h: v[:, keep] for h, v in Y.items()}
    d = {h: v[pairs[:, 1]] - v[pairs[:, 0]] for h, v in Y.items()}
    nz = change_noise(d["full"][ev], d["a"][ev], d["b"][ev])
    resm = nz["resolvable"]
    noise_b = (d["a"][ev] - d["b"][ev]) ** 2 / 2.0
    # the level fit uses the MODERN epoch of training cells by default in both populations: R scales with 1/n, so
    # designs must match n to be compared (skeptic, 2026-10-09: "trained 261 vs backcast 126" was n 2838 vs 1419)
    both = a.lvl_epochs == "both" and not want_wh
    lvl_groups = np.unique(np.concatenate([pairs[tr, 1]] + ([pairs[tr, 0]] if both else [])))
    res["level_fit_rows"] = int(len(lvl_groups))
    rngb = np.random.default_rng(0)
    for r in a.ranks:
        m = blr.fit(D[lvl_groups, :r], Y["b"][lvl_groups], [(0, r)])
        dz = D[pairs[ev, 1], :r] - D[pairs[ev, 0], :r]
        pred = dz @ m["coef"].T
        out_r = {"n_resolvable": int(resm.sum())}
        # total: the species' mean change (trend) is outside the readout's model and counts as error; place: trend
        # removed from prediction, truth AND the posterior variance (V from centered dz) alike
        dzc = dz - dz.mean(0)
        for kind, (p_, t_, dz_) in (("total", (pred, d["b"][ev], dz)),
                                    ("place", (pred - pred.mean(0), d["b"][ev] - d["b"][ev].mean(0), dzc))):
            V = np.einsum("cp,spq,cq->cs", dz_, m["Ainv"], dz_)
            err2 = (p_ - t_) ** 2 - noise_b
            with np.errstate(invalid="ignore", divide="ignore"):
                Rs = err2.sum(0) / V.sum(0)
            Rr = Rs[resm & np.isfinite(Rs)]
            boot = [np.median(rngb.choice(Rr, len(Rr))) for _ in range(1000)] if len(Rr) > 2 else []
            out_r[kind] = {"median_R": float(np.median(Rr)) if len(Rr) else None,
                           "median_R_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
                           if len(boot) else None,
                           "share_R_above_4": float((Rr > 4).mean()) if len(Rr) else None,
                           "R_quartiles": [float(np.percentile(Rr, q)) for q in (25, 75)] if len(Rr) else None}
        res["interval_honesty"][str(r)] = out_r
    with open(os.path.join(a.out, "readout_honesty.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def _q(v):
    return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99))}


def summarize(out):
    r = json.load(open(os.path.join(out, "readout_honesty.json"), encoding="utf-8"))
    print(f"readout honesty [{r['population']}]")
    print("  novelty (Mahalanobis^2 per dim vs training rows; median / p90):")
    for space, sets in r["novelty"].items():
        print(f"    {space:11s} " + "   ".join(f"{k} {v['median']:.2f}/{v['p90']:.2f}" for k, v in sets.items()))
    for rk, v in r["leverage_vs_error"].items():
        f = lambda x: "-" if x is None else f"{x:+.2f}"
        print(f"  r{rk}: Spearman(backcast error, z leverage) {f(v['spearman_error_vs_z_leverage'])}, "
              f"(error, covariate novelty) {f(v['spearman_error_vs_cov_novelty'])}, "
              f"(z leverage, covariate novelty) {f(v['spearman_z_leverage_vs_cov_novelty'])}")
    for rk, v in r["interval_honesty"].items():
        for kind in ("total", "place"):
            w = v.get(kind) or {}
            if w.get("median_R") is not None:
                ci = w.get("median_R_ci") or [float("nan"), float("nan")]
                print(f"  r{rk} {kind:5s}: species change intervals: median R {w['median_R']:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] (IQR "
                      f"{w['R_quartiles'][0]:.1f}-{w['R_quartiles'][1]:.1f}); {w['share_R_above_4']:.0%} of "
                      f"{v['n_resolvable']} resolvable species have errors > 2x their interval's sd")


if __name__ == "__main__":
    main()
