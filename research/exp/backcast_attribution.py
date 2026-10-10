"""E035b: which part of DESK's backcast change -- land-use-driven or climate-driven -- tracks the REAL change, and which
carries the lean toward climate/land-use analogs? (the skeptic's test after E035)

    python research/exp/backcast_attribution.py --cache <dir> --run-dir <desk run dir> --overlay <run overlay> --out <dir>
    python research/exp/backcast_attribution.py --summarize <out>

On a model with withheld decades (tempho1995: 1966-1995 withheld), DESK's change on HELD-OUT cells between the early
withheld epoch (1966-86) and the modern epoch (2005-25) is re-encoded with one covariate family swapped, year by year,
for its value 39 years later (1966 -> 2005, ..., 1986 -> 2025): land use (LUH, HYDE, BUI) or climate. Then
    land-use part = change - change(land use swapped)        climate part = change - change(climate swapped)
Each part (and the full change) is graded against the observed community change (ESK r24, noise-free cross-half
covariances, centred over cells): correlation, slope of truth on prediction (k), share of the full change's size;
and its alignment with the analog direction (E017 v2: modern DESK z of the 10 cells whose MODERN covariates are nearest
the cell's early covariates minus that of the 10 nearest its modern covariates, > 200 km away), with a random-rotation
null (each cell's covariate change rotated at random in PC space, M19).
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
from lib import covfeat  # noqa: E402

LAND = ("landuse", "hyde", "bui")
SHIFT = 39


def ccov(x, y):
    return ((x - x.mean(0)) * (y - y.mean(0))).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--run-dir")
    ap.add_argument("--overlay")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-null", type=int, default=40)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    os.makedirs(a.out, exist_ok=True)
    ov = json.load(open(a.overlay, encoding="utf-8"))
    ov.setdefault("paths", {})["desk_output_dir"] = a.run_dir
    ovp = os.path.join(a.out, "overlay_run.json")
    json.dump(ov, open(ovp, "w", encoding="utf-8"), indent=1)
    os.environ["ESK_DESK_CONFIG"] = ovp
    from change_oracle import desk_epoch_z, group_rows
    from src.config_utils import load_config
    from src.community_encoder.train_DESK import covariate_io as cio
    from src.community_encoder.train_DESK.validate_spacetime import encode_points
    cfg = load_config()
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    r = a.rank
    E, full, half_a, half_b = group_rows(a.cache, keys, [int(y) for y in split["withheld"]])
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]]
    ev = np.where(held)[0]
    # truth: observed community change, halves A/B (noise-free cross-covariances)
    Za, Zb = E["z_a"][:, :r].astype("float64"), E["z_b"][:, :r].astype("float64")
    Ta = (Za[pairs[:, 1]] - Za[pairs[:, 0]])[ev]
    Tb = (Zb[pairs[:, 1]] - Zb[pairs[:, 0]])[ev]
    vt = ccov(Ta, Tb)
    # DESK modern epoch from the cache (unswapped)
    zr = np.load(c("z_raw_all.npy"), mmap_mode="r")
    ci, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    Zmod = desk_epoch_z(zr, ci, y0, keys, [full[pairs[j, 1]] for j in ev], r)
    # early epoch re-encoded: actual, land use swapped, climate swapped
    early_rows = [np.asarray(full[pairs[j, 0]]) for j in ev]
    pts = np.concatenate(early_rows)
    pidx = keys[pts].astype("int32")
    schema = cio.load_schema(os.path.join(cfg["paths"]["hist_dir"], "yearly_states"))
    fam = {"land": [], "climate": []}
    for s_ in schema["streams"]:
        cols = list(range(int(s_["start"]), int(s_["end"])))
        if s_["name"] == "climate":
            fam["climate"] += cols
        elif s_["name"] in LAND:
            fam["land"] += cols
    orig = cio.load_state_stack
    mode = {"which": None}

    def patched(year, states_dir, schema_):
        x = orig(year, states_dir, schema_)
        if mode["which"] is not None and 1966 <= int(year) <= 1986:
            y2 = orig(int(year) + SHIFT, states_dir, schema_)
            x = x.copy()
            x[..., fam[mode["which"]]] = y2[..., fam[mode["which"]]]
        return x
    cio.load_state_stack = patched
    early = {}
    for which in (None, "land", "climate"):
        mode["which"] = which
        Zp, ok = encode_points(cfg, pidx)
        Zp = Zp[:, :r].astype("float64")
        off = np.cumsum([0] + [len(rl) for rl in early_rows])
        early[str(which)] = np.stack([np.nanmean(Zp[off[i]:off[i + 1]], 0) for i in range(len(early_rows))])
    cio.load_state_stack = orig
    D_full = Zmod - early["None"]
    D_noland = Zmod - early["land"]
    D_noclim = Zmod - early["climate"]
    parts = {"full": D_full, "land_part": D_full - D_noland, "climate_part": D_full - D_noclim,
             "land_swapped(= non-land change)": D_noland, "climate_swapped(= non-climate change)": D_noclim}
    # analog direction (E017 v2, DESK modern z), covariates from the cache's F_cov at keys
    F = np.load(c("F_cov.npy"), mmap_mode="r")
    allpairs_modern = [np.asarray(full[pairs[j, 1]]) for j in range(len(pairs))]
    cov_early = np.stack([np.asarray(F[np.asarray(full[pairs[j, 0]])], "float64").mean(0) for j in range(len(pairs))])
    cov_mod = np.stack([np.asarray(F[rl], "float64").mean(0) for rl in allpairs_modern])
    good = np.isfinite(cov_early).all(1) & np.isfinite(cov_mod).all(1)
    P = covfeat.cov_pcs(np.vstack([cov_early, cov_mod]), np.concatenate([good, good]), n_pcs=64)
    Pe, Pm = P[: len(pairs)], P[len(pairs):]
    zmod_all = desk_epoch_z(zr, ci, y0, keys, allpairs_modern, r)
    xy = cells.astype("float64") * 27.0
    cand = np.where(good)[0]

    def analog_mean(q, i):
        far = np.sqrt(((xy[cand] - xy[i]) ** 2).sum(1)) > 200.0
        cc_ = cand[far]
        nn = cc_[np.argsort(((Pm[cc_] - q) ** 2).sum(1))[:10]]
        return zmod_all[nn].mean(0)
    Dir = np.stack([analog_mean(Pe[i], i) - analog_mean(Pm[i], i) for i in ev])
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(a.n_null):
        Dn = []
        for i in ev:
            dv = Pe[i] - Pm[i]
            Q, _ = np.linalg.qr(rng.normal(size=(len(dv), len(dv))))
            Dn.append(analog_mean(Pm[i] + Q @ dv, i) - analog_mean(Pm[i], i))
        nulls.append(np.stack(Dn))

    def align(B, D):
        Bc, Dc = B - B.mean(0), D - D.mean(0)
        return float((Bc * Dc).sum() / np.sqrt((Bc ** 2).sum() * (Dc ** 2).sum()))
    # species readout of each part: level coefficients from training cells' modern levels (E013/E030), noise-free
    from lib import blr
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import change_noise
    Xd = np.load(c("X_dev.npy")).astype("float64")
    Ys = {h: epoch_mean_observed(Xd, g).astype("float64") for h, g in (("full", full), ("a", half_a), ("b", half_b))}
    keep = (Ys["full"] > 0).any(0)
    Ys = {h: v[:, keep] for h, v in Ys.items()}
    dS = {h: (v[pairs[:, 1]] - v[pairs[:, 0]])[ev] for h, v in Ys.items()}
    resm = change_noise(dS["full"], dS["a"], dS["b"])["resolvable"]
    buf = split["buffer"][cells[:, 0], cells[:, 1]]
    trc = np.where(~held & ~buf)[0]
    lvl = np.unique(pairs[trc, 1])
    Fd = desk_epoch_z(zr, ci, y0, keys, full, r)
    beta = blr.fit(Fd[lvl], Ys["b"][lvl], [(0, r)])["coef"].T
    vts = ccov(dS["a"][:, resm], dS["b"][:, resm])

    # change is modern - early; the backcast change (early - modern) aligns with D, so use -part
    res = {"cache": a.cache, "run_dir": a.run_dir, "n_cells": int(len(ev)), "parts": {}}
    nfull = float(np.linalg.norm(D_full, axis=1).mean())
    for name, Dp in parts.items():
        cov_t = 0.5 * (ccov(Dp, Ta) + ccov(Dp, Tb))
        vp = ccov(Dp, Dp)
        al = align(-Dp, Dir)
        nul = [align(-Dp, Dn) for Dn in nulls]
        ps = (Dp @ beta)[:, resm]
        cs, vps = ccov(ps, dS["full"][:, resm]), ccov(ps, ps)
        res["parts"][name] = {"corr_with_truth": float(cov_t / np.sqrt(vp * vt)) if vp > 0 and vt > 0 else None,
                              "k": float(cov_t / vp) if vp > 0 else None,
                              "size_vs_full": float(np.linalg.norm(Dp, axis=1).mean() / nfull),
                              "analog_alignment": al, "analog_null_p95": float(np.percentile(nul, 95)),
                              "analog_null_mean": float(np.mean(nul)),
                              "species_corr": float(cs / np.sqrt(vps * vts)) if vps > 0 and vts > 0 else None,
                              "species_k": float(cs / vps) if vps > 0 else None}
    with open(os.path.join(a.out, "backcast_attribution.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "backcast_attribution.json"), encoding="utf-8"))
    print(f"backcast attribution [{os.path.basename(os.path.dirname(r['run_dir']))}]: {r['n_cells']} held-out cells, "
          f"withheld 1966-86 vs 2005-25; noise-free centred corr with the observed community change, k, size, and "
          f"alignment of the backcast with the analog direction (random-rotation null):")
    for name, v in r["parts"].items():
        c = v["corr_with_truth"]
        sp = (f" | species corr {v['species_corr']:+.3f} k {v['species_k']:.2f}" if v.get("species_corr") is not None
              else "")
        print(f"  {name:38s} corr {c:+.3f}  k {v['k']:.2f}  size {v['size_vs_full']:.2f}  analog {v['analog_alignment']:+.3f}"
              f" (null mean {v['analog_null_mean']:+.3f}, p95 {v['analog_null_p95']:+.3f})" + sp)


if __name__ == "__main__":
    main()
