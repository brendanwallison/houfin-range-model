"""The age-model-matched readout (Phase 1.3): block-prior BLR of dev species' log1p abundance on DESK z.

    python research/exp/fast_arm.py --cache <dir> --out <dir> [--features raw|ema] [--rank 24]
        [--kernel iso|split|ard] [--ref-window 1996 2025] [--time-basis 0] [--lag-hl 0] [--tag name]
    python research/exp/fast_arm.py --summarize <out dir>

Mirrors the downstream contract by default: RAW z (what the cube exports), the first 24 of 64 dims by
position, an intercept, and one iid prior on the coefficients (``--kernel iso``). Variants are the
kernel experiments the age model could adopt with an iid-per-block prior:

    split   s_b^2 zbar.zbar' + s_w^2 dz.dz', zbar = the cell's mean z over --ref-window (a FIXED window,
            so dz is defined for any year, 1902 included), dz = z - zbar            (B1)
    ard     one amplitude per dimension                                              (B1 diagnostic)
    --time-basis k   k continental cosine terms over 1902-2025, own amplitude        (B4; K trend)
    --lag-hl H       raw z re-smoothed by a causal EMA of half-life H years          (L1)

Reported per held-out group (space / time / space_time; tempho_1995's time group is the long-reach
proxy for 1902-1939), on the SAME rows for every predictor:
    level   per-species RMSE skill vs intercept, vs no_change (the cell's modern-epoch z) and vs
            persistence (the cell's observed modern mean; time group), plus 50/90% coverage
    change  epoch-mean change (validation_core estimand, noise-only back-transform), on RESOLVABLE
            species: RMSE skill vs no_change, captured share of available change, and the
            noise-corrected ATTENUATION SLOPE cov(pred, obs) / (var(obs) - noise) -- a "modern map"
            backcast has slope ~0 however good its level is
In-sample rows (the time set's modern epoch) are predicted leave-one-out. Every dev evaluation is
logged to research/ledger.csv. Sealed species are never read here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from lib import blr, ledger  # noqa: E402

GROUPS = {1: "space", 2: "time", 3: "space_time"}


def load_cache(cache):
    j = lambda f: os.path.join(cache, f)
    c = {k: np.load(j(f"{k}.npy"), mmap_mode="r") for k in
         ("keys", "X_dev", "cells", "years", "z_raw_all", "z_ema_all", "key_cell_index")}
    c["split"] = dict(np.load(j("split.npz")))
    c["meta"] = json.load(open(j("meta.json"), encoding="utf-8"))
    return c


def causal_ema(z_all, half_life):
    """Causal EMA along the year axis of dense (cells, years, L) z, as the DESK output EMA does."""
    a = 1.0 - 2.0 ** (-1.0 / float(half_life))
    out = np.empty_like(z_all)
    out[:, 0] = z_all[:, 0]
    for t in range(1, z_all.shape[1]):
        out[:, t] = a * z_all[:, t] + (1 - a) * out[:, t - 1]
    return out


def features(c, args):
    """Row features at every key, and at each key's cell in the modern epoch (the no-change z)."""
    z_all = np.asarray(c["z_ema_all"] if args.features == "ema" else c["z_raw_all"], "float64")
    if args.lag_hl > 0:
        z_all = causal_ema(z_all, args.lag_hl)
    z_all = z_all[:, :, : args.rank]
    years = np.asarray(c["years"])
    keys = np.asarray(c["keys"])
    ci = np.asarray(c["key_cell_index"])
    yi = keys[:, 2] - years[0]
    z = z_all[ci, yi]
    mod = (years >= 2005) & (years <= 2025)
    z_nc = z_all[:, mod].mean(1)[ci]                               # cell's modern-epoch mean z
    ref = (years >= args.ref_window[0]) & (years <= args.ref_window[1])
    zbar = z_all[:, ref].mean(1)[ci]
    blocks_of = {"iso": lambda Z, Zb: (Z, [(0, Z.shape[1])]),
                 "split": lambda Z, Zb: (np.hstack([Zb, Z - Zb]),
                                         [(0, Z.shape[1]), (Z.shape[1], 2 * Z.shape[1])]),
                 "ard": lambda Z, Zb: (Z, [(j, j + 1) for j in range(Z.shape[1])])}
    X, blocks = blocks_of[args.kernel](z, zbar)
    Xnc, _ = blocks_of[args.kernel](z_nc, zbar)
    if args.time_basis > 0:
        T = time_basis(keys[:, 2], args.time_basis)
        p = X.shape[1]
        X = np.hstack([X, T])
        # no-change holds z at the modern epoch but keeps the row's own year in the time basis:
        # the continental trend is not a property of the place
        Xnc = np.hstack([Xnc, T])
        blocks = blocks + [(p, p + T.shape[1])]
    return X, Xnc, blocks


def time_basis(years, k, lo=1902, hi=2025):
    u = (np.asarray(years, "float64") - lo) / (hi - lo)
    return np.column_stack([np.cos(np.pi * (j + 1) * u) for j in range(k)])


def attenuation(d_pred, d_obs, noise):
    """Per-species noise-corrected slope of predicted on observed change; NaN where unresolvable."""
    dp, do = np.asarray(d_pred), np.asarray(d_obs)
    ok = np.isfinite(dp).all(1)
    dp, do = dp[ok] - dp[ok].mean(0), do[ok] - do[ok].mean(0)
    cov = (dp * do).mean(0)
    sig = do.var(0) - noise["noise"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(noise["resolvable"] & (sig > 0), cov / sig, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--features", default="raw", choices=("raw", "ema"))
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--kernel", default="iso", choices=("iso", "split", "ard"))
    ap.add_argument("--ref-window", type=int, nargs=2, default=(1996, 2025))
    ap.add_argument("--time-basis", type=int, default=0)
    ap.add_argument("--lag-hl", type=float, default=0.0)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--tag", default="")
    ap.add_argument("--exp", default=os.environ.get("RESEARCH_EXP", "adhoc"))
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)

    from src.community_encoder.train_DESK.validate_bbs_routes import (EPOCH_EARLY, EPOCH_MODERN,
                                                                       MIN_EPOCH_YEARS, epoch_gate)
    from src.community_encoder.train_DESK.validate_gp_species import (
        block_sums, change_sets, persistence_prediction, pooled_skill, predicted_raw,
        interval_coverage, self_index)
    from src.community_encoder.train_DESK.validation_core import (captured_share, change_noise,
                                                                  epoch_values, split_half_change)
    t0 = time.perf_counter()
    os.makedirs(a.out, exist_ok=True)
    c = load_cache(a.cache)
    keys = np.asarray(c["keys"])
    Y = np.log1p(np.asarray(c["X_dev"], "float64"))
    sp = c["split"]
    is_train, group = sp["is_train"].astype(bool), sp["group"]
    withheld = [int(y) for y in sp["withheld"]]
    yr = keys[:, 2]
    time_cells = {(int(r), int(q)) for r, q in keys[group == 2, :2]}
    in_tc = np.array([(int(r), int(q)) in time_cells for r, q in keys[:, :2]])
    insample = is_train & in_tc & (yr >= EPOCH_MODERN[0]) & (yr <= EPOCH_MODERN[1])
    tr = np.where(is_train)[0]
    te = np.where((group > 0) | insample)[0]
    keep = (Y[tr] > 0).any(0)                                     # gradable: a training detection
    Y = Y[:, keep]
    species = np.asarray(c["meta"]["dev_species"])[keep]

    X, Xnc, blocks = features(c, a)
    model = blr.fit(X[tr], Y[tr], blocks)
    sidx = self_index(tr, te)
    y_self = np.where((sidx >= 0)[:, None], Y[te], np.nan)
    preds = {"model": blr.predict(model, X[te], y_self=y_self),
             "no_change": blr.predict(model, Xnc[te])}
    mu0 = Y[tr].mean(0)
    v0 = Y[tr].var(0)
    preds["intercept"] = (np.broadcast_to(mu0, (len(te), Y.shape[1])),
                          np.broadcast_to(v0, (len(te), Y.shape[1])))
    preds["persistence"], pnoise = persistence_prediction(keys, Y, tr, te)
    noise = {"model": model["v"], "no_change": model["v"], "intercept": v0,
             "persistence": pnoise}
    Yte, g_te, bid = Y[te], group[te], sp["block_id"][te]
    res = {"cache": a.cache, "tag": a.tag, "features": a.features, "rank": a.rank,
           "kernel": a.kernel, "ref_window": list(a.ref_window), "time_basis": a.time_basis,
           "lag_hl": a.lag_hl, "n_species": int(Y.shape[1]), "withheld": withheld,
           "block_amplitude_medians": [float(np.median(model["a"][model["ok"], b]))
                                       for b in range(model["a"].shape[1])],
           "noise_median": float(np.median(model["v"][model["ok"]])), "level": {}, "change": {}}
    per = {"species": species}

    for g, gname in GROUPS.items():
        sel = np.where(g_te == g)[0]
        if not len(sel):
            continue
        out = {"n_rows": int(len(sel))}
        err = {k: (Yte[sel] - np.asarray(v[0])[sel]) ** 2 for k, v in preds.items()}
        ok = {k: np.isfinite(e).all(1) for k, e in err.items()}
        sig = (Yte[sel] > 0).astype("float64")
        for ref in ("intercept", "no_change", "persistence"):
            m = ok["model"] & ok[ref]
            if not m.any():
                out[f"vs_{ref}"] = None
                continue
            _, sa = block_sums(err["model"][m], bid[sel][m])
            _, sb = block_sums(err[ref][m], bid[sel][m])
            _, sg = block_sums(sig[m], bid[sel][m])
            sb[:, sg.sum(0) == 0] = 0.0
            pooled, sk = pooled_skill(sa, sb, a.n_boot, 0, block_signal=sg)
            out[f"vs_{ref}"] = {k: pooled.get(k) for k in ("median", "median_ci", "share_above_zero",
                                                          "n_species_defined")}
            per[f"level_{gname}_vs_{ref}"] = sk
        mu, var = preds["model"]
        out["coverage50"] = float(interval_coverage(Yte[sel], mu[sel], var[sel], 0.5).mean())
        out["coverage90"] = float(interval_coverage(Yte[sel], mu[sel], var[sel], 0.9).mean())
        res["level"][gname] = out

    tk = keys[te]
    row_group = np.where(np.isin(te, np.where(insample)[0]), 0, g_te)
    raw_te = np.expm1(Yte)
    for sname, rows in change_sets(tk, row_group, withheld).items():
        cells, e_loc, m_loc, gate = epoch_gate(tk[rows], EPOCH_EARLY, EPOCH_MODERN, MIN_EPOCH_YEARS)
        if not len(cells):
            res["change"][sname] = {"note": "no cell passes the epoch gate"}
            continue
        e_rows = [rows[np.asarray(r)] for r in e_loc]
        m_rows = [rows[np.asarray(r)] for r in m_loc]
        d_full, d_a, d_b = split_half_change(raw_te, e_rows, m_rows, tk[:, 2])
        nz = change_noise(d_full, d_a, d_b)
        cell_block = bid[[int(r[0]) for r in e_rows]]
        out = {"n_cells": int(len(cells)), "n_resolvable": int(nz["resolvable"].sum())}
        dp = {}
        for k in ("model", "no_change"):
            pe, pm = epoch_values(predicted_raw(preds[k][0], noise[k][None, :]), e_rows, m_rows)
            dp[k] = pm - pe
        _, sa = block_sums((dp["model"] - d_full) ** 2, cell_block)
        _, sb = block_sums((dp["no_change"] - d_full) ** 2, cell_block)
        _, sg = block_sums((np.abs(d_full) > 0).astype("float64"), cell_block)
        sa[:, ~nz["resolvable"]], sb[:, ~nz["resolvable"]] = 0.0, 0.0
        pooled, sk = pooled_skill(sa, sb, a.n_boot, 0, block_signal=sg)
        out["skill_vs_no_change"] = {k: pooled.get(k) for k in ("median", "median_ci",
                                                                "share_above_zero",
                                                                "n_species_defined")}
        _, cap = captured_share(d_full, dp["model"], nz)
        out["captured_pooled"] = cap
        att = attenuation(dp["model"], d_full, nz)
        fin = np.isfinite(att)
        out["attenuation_median"] = float(np.median(att[fin])) if fin.any() else None
        out["attenuation_iqr"] = ([float(np.percentile(att[fin], 25)),
                                   float(np.percentile(att[fin], 75))] if fin.any() else None)
        res["change"][sname] = out
        per[f"change_skill_{sname}"] = sk
        per[f"attenuation_{sname}"] = att
        per[f"resolvable_{sname}"] = nz["resolvable"]
        ledger.log_eval(a.exp, f"fast_arm/{a.kernel}/{a.features}/r{a.rank}", sname, "dev",
                        int(nz["resolvable"].sum()), a.tag)

    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    import pandas as pd
    pd.DataFrame(per).to_csv(os.path.join(a.out, "per_species.csv"), index=False)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "results.json"), encoding="utf-8"))
    f = lambda v: "-" if v is None else (f"{v:+.3f}" if isinstance(v, float) else str(v))
    print(f"fast arm [{r['tag']}] features={r['features']} r={r['rank']} kernel={r['kernel']} "
          f"tb={r['time_basis']} lag={r['lag_hl']} species={r['n_species']} "
          f"amps={[round(x, 4) for x in r['block_amplitude_medians']][:6]} "
          f"noise={r['noise_median']:.3f} ({r['elapsed_s']} s)")
    for g, o in r["level"].items():
        parts = [f"{ref}: {f((o.get('vs_' + ref) or {}).get('median'))}" for ref in
                 ("intercept", "no_change", "persistence")]
        print(f"  level {g:10s} n={o['n_rows']:6d} " + "  ".join(parts)
              + f"  cov50 {o['coverage50']:.2f} cov90 {o['coverage90']:.2f}")
    for s, o in r["change"].items():
        if "note" in o:
            print(f"  change {s:10s} {o['note']}")
            continue
        sk = o["skill_vs_no_change"]
        print(f"  change {s:10s} cells={o['n_cells']} resolvable={o['n_resolvable']} skill "
              f"{f(sk.get('median'))} CI {[round(x, 3) for x in (sk.get('median_ci') or [])]} captured {f(o['captured_pooled'])} "
              f"attenuation {f(o['attenuation_median'])} IQR {[round(x, 3) for x in (o['attenuation_iqr'] or [])]}")


if __name__ == "__main__":
    main()
