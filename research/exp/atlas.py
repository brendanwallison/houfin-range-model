"""Turnover attenuation atlas (Phase 1.5): how much NOISE-FREE temporal vs spatial community signal survives
each stage -- exact Ruzicka -> ESK rank r -> DESK.

    python research/exp/atlas.py --cache <dir> --out <dir> [--population trained|withheld]
    python research/exp/atlas.py --summarize <out>

NOISE-FREE SIGNAL FROM SPLIT HALVES. Every cell-epoch's surveyed years are split year-balanced (ABBA)
into halves A and B, whose communities differ only by measurement noise (esk_project.py). For a pair of
cell-epochs (1, 2) the half-differences D_A = phi(1_A) - phi(2_A) and D_B = phi(1_B) - phi(2_B) carry
independent noise, so <D_A, D_B> estimates the squared norm of the TRUE difference. In the exact
kernel's feature space that is K(1A,1B) - K(1A,2B) - K(2A,1B) + K(2A,2B); in ESK coordinates a dot.
    TEMPORAL pair  the same cell, early (1966-1986) vs modern (2005-2025) epoch
    SPATIAL pair   two cells at Chebyshev lag d in the modern epoch, d in --lags

Per stage:
    retention(r)  = sum <P_r D_A, P_r D_B> / sum <D_A, D_B>_K     ESK rank r (<= 1 up to Nystrom error)
    DESK slope    = sum <D_desk, (D_A + D_B)/2> / sum <D_A, D_B>  in ESK rank-r coordinates; 1 = DESK
                    reproduces the true difference's size along it (noise-corrected: errors in variables)
    DESK corr     = sum <D_desk, Dbar> / sqrt(sum |D_desk|^2 sum <D_A, D_B>)   disattenuated direction
DESK is graded on HELD-OUT cells only: at a training cell its prediction was fitted to that cell's own
observation noise, which the half-differences share. ESK stages use every cell (the basis saw them all;
recorded, mild).

The R1 reading is MATCHED: spatial retention is a function of the spatial signal's size (it falls as
pairs get closer and differences smaller), so temporal retention is compared with the spatial curve
interpolated at the TEMPORAL signal's size. Below the curve = truncation drops temporal DIRECTIONS; on
it = temporal change is simply small. The temporal signal is also expressed as the spatial lag with the
same exact-kernel signal (the "km of turnover" a cell's decades amount to). Per ESK component, the
temporal and spatial shares of signal energy show where in the spectrum the temporal signal lives.
"""
import argparse
import json
import os
import sys

import numpy as np

CELL_KM = 27.0


def ruzicka_rows(X, Y):
    """Row-paired uncentered Ruzicka (generalized Jaccard) of non-negative vectors."""
    X, Y = np.asarray(X, "float64"), np.asarray(Y, "float64")
    mx = np.maximum(X, Y).sum(1)
    return np.where(mx > 0, np.minimum(X, Y).sum(1) / np.maximum(mx, 1e-300), 0.0)


KERNEL = ruzicka_rows          # tests swap in a linear kernel, where the answer is known exactly


def cross_K(x1a, x1b, x2a, x2b):
    return KERNEL(x1a, x1b) - KERNEL(x1a, x2b) - KERNEL(x2a, x1b) + KERNEL(x2a, x2b)


def cross_z(z1a, z1b, z2a, z2b):
    return ((np.asarray(z1a) - z2a) * (np.asarray(z1b) - z2b)).sum(1)


def pairs_temporal(E, population):
    """``(i_early, i_modern)`` group indices per cell; early groups withheld iff population == withheld."""
    want_wh = population == "withheld"
    early = {tuple(c): j for j, (c, e, w) in enumerate(zip(E["cells"], E["epoch"], E["withheld"]))
             if e == 0 and bool(w) == want_wh}
    modern = {tuple(c): j for j, (c, e, w) in enumerate(zip(E["cells"], E["epoch"], E["withheld"]))
              if e == 1 and not w}
    both = sorted(set(early) & set(modern))
    return np.array([early[c] for c in both], int), np.array([modern[c] for c in both], int)


def pairs_spatial(E, lag, rng):
    """DISJOINT modern-epoch neighbour pairs at Chebyshev distance ``lag``: each cell is used at most
    once. Overlapping pairs share a cell's half-noise across pairs, which a resampling scheme that
    treats pairs as independent cannot see -- measured on a planted case, its CI then excluded the
    truth."""
    modern = {tuple(c): j for j, (c, e, w) in enumerate(zip(E["cells"], E["epoch"], E["withheld"]))
              if e == 1 and not w}
    ring = [(dr, dc) for dr in range(-lag, lag + 1) for dc in range(-lag, lag + 1)
            if max(abs(dr), abs(dc)) == lag]
    used, a, b = set(), [], []
    cells = list(modern)
    for i in rng.permutation(len(cells)):
        r, c = cells[i]
        if (r, c) in used:
            continue
        for k in rng.permutation(len(ring)):
            nb = (r + ring[k][0], c + ring[k][1])
            if nb in modern and nb not in used:
                a.append(modern[(r, c)])
                b.append(modern[nb])
                used.update({(r, c), nb})
                break
    return np.array(a, int), np.array(b, int)


def pair_blocks(E, i1, block=6):
    """The 6x6-cell spatial block of each pair's first cell: the resampling unit, as in the suite."""
    c = E["cells"][i1]
    return (c[:, 0] // block) * 100000 + c[:, 1] // block


N_BOOT = 200


def _ratio_ci(num, den, blocks, rng, n_boot=N_BOOT):
    """95% BLOCK-bootstrap CI of sum(num)/sum(den), resampling spatial blocks of pairs. The estimators
    are unbiased but, where true differences are small next to the halves' noise (close spatial
    pairs), one ratio can be off by a factor of two -- so no ratio is reported without its CI."""
    ub, inv = np.unique(blocks, return_inverse=True)
    if len(ub) < 5:
        return None
    nb = np.bincount(inv, weights=num, minlength=len(ub))
    db = np.bincount(inv, weights=den, minlength=len(ub))
    W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=n_boot)
    d = W @ db
    v = np.where(d > 0, (W @ nb) / np.where(d > 0, d, 1.0), np.nan)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None


def stage_table(E, i1, i2, ranks, zd1=None, zd2=None, seed=0):
    """Exact-kernel signal, ESK retention per rank, and DESK slope/corr per rank for pairs (i1, i2)."""
    rng = np.random.default_rng(seed)
    blk = pair_blocks(E, i1)
    sk = cross_K(E["x_a"][i1], E["x_b"][i1], E["x_a"][i2], E["x_b"][i2])
    out = {"n_pairs": int(len(i1)), "signal_exact_mean": float(sk.mean()),
           "signal_exact_se": float(sk.std(ddof=1) / np.sqrt(max(len(sk), 2))), "esk": {},
           "desk": {}}
    for r in ranks:
        za, zb = E["z_a"][:, :r], E["z_b"][:, :r]
        sz = cross_z(za[i1], zb[i1], za[i2], zb[i2])
        out["esk"][str(r)] = {"retention": float(sz.sum() / sk.sum()) if sk.sum() > 0 else None,
                              "retention_ci": _ratio_ci(sz, sk, blk, rng),
                              "signal_mean": float(sz.mean())}
        if zd1 is not None:
            dd = zd1[:, :r] - zd2[:, :r]
            dbar = 0.5 * ((za[i1] - za[i2]) + (zb[i1] - zb[i2]))
            per_num = (dd * dbar).sum(1)
            num, den = float(per_num.sum()), float(sz.sum())
            out["desk"][str(r)] = {
                "slope": num / den if den > 0 else None,
                "slope_ci": _ratio_ci(per_num, sz, blk, rng),
                "corr": (num / np.sqrt((dd * dd).sum() * den)) if den > 0 else None,
                "desk_energy_over_signal": float((dd * dd).sum() / den) if den > 0 else None}
    return out


def component_t0(E, i1, i2, zd1, zd2, r):
    """T0, per ESK component: split-half reliability of the observed change (corr of the two halves'
    differences across pairs) and DESK's noise-corrected slope on it. If DESK's under-movement were
    just MSE-optimal shrinkage toward unreliable targets, slope would track reliability component by
    component; slopes far below reliability mean DESK shrinks more than the noise justifies."""
    za, zb = E["z_a"][:, :r], E["z_b"][:, :r]
    da, db = za[i1] - za[i2], zb[i1] - zb[i2]
    cross = (da * db).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = cross / np.sqrt((da * da).sum(0) * (db * db).sum(0))
        dd = zd1[:, :r] - zd2[:, :r]
        slope = (dd * 0.5 * (da + db)).sum(0) / cross
    return {"reliability": rel.tolist(), "desk_slope": slope.tolist()}


def component_energy(E, i1, i2, r):
    za, zb = E["z_a"][:, :r], E["z_b"][:, :r]
    e = ((za[i1] - za[i2]) * (zb[i1] - zb[i2])).sum(0)
    return e / max(e.sum(), 1e-300)


def desk_epoch_z(cache, E, which):
    z = np.load(os.path.join(cache, f"z_{which}.npy"), mmap_mode="r")
    ptr, flat = E["rows_ptr"], E["rows_flat"]
    return np.stack([np.asarray(z[flat[ptr[j]:ptr[j + 1]]], "float64").mean(0)
                     for j in range(len(ptr) - 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="trained", choices=("trained", "withheld"))
    ap.add_argument("--ranks", type=int, nargs="+", default=(6, 24, 64))
    ap.add_argument("--lags", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    os.makedirs(a.out, exist_ok=True)
    E = dict(np.load(os.path.join(a.cache, "esk_epochs.npz")))
    split = np.load(os.path.join(a.cache, "split.npz"))
    ho = split["holdout"]
    rng = np.random.default_rng(a.seed)
    ranks = [r for r in a.ranks if r <= E["z_a"].shape[1]]
    desk = {w: desk_epoch_z(a.cache, E, w) for w in ("ema", "raw")}
    held = ho[E["cells"][:, 0], E["cells"][:, 1]]

    res = {"cache": a.cache, "population": a.population, "ranks": ranks, "temporal": {},
           "spatial": {}, "components": {}}
    e_i, m_i = pairs_temporal(E, a.population)
    res["temporal"]["all_cells"] = stage_table(E, e_i, m_i, ranks)
    h = held[e_i]
    for w in ("ema", "raw"):
        # same orientation as the observed half-differences: phi(early) - phi(modern)
        res["temporal"][f"heldout_desk_{w}"] = stage_table(E, e_i[h], m_i[h], ranks,
                                                           desk[w][e_i[h]], desk[w][m_i[h]])
    for lag in a.lags:
        s1, s2 = pairs_spatial(E, lag, rng)
        if not len(s1):
            continue
        res["spatial"][str(lag)] = stage_table(E, s1, s2, ranks)
        hs = held[s1] & held[s2]
        if hs.sum() >= 10:
            res["spatial"][f"{lag}_heldout_desk_raw"] = stage_table(
                E, s1[hs], s2[hs], ranks, desk["raw"][s1[hs]], desk["raw"][s2[hs]])
        if lag == min(a.lags):
            res["components"]["spatial_share"] = component_energy(E, s1, s2, max(ranks)).tolist()
    res["components"]["temporal_share"] = component_energy(E, m_i, e_i, max(ranks)).tolist()
    for w in ("ema", "raw"):
        res["components"][f"t0_desk_{w}"] = component_t0(E, e_i[h], m_i[h], desk[w][e_i[h]],
                                                         desk[w][m_i[h]], max(ranks))

    # matched reading: spatial retention as a function of spatial signal size, at the temporal size
    lags = sorted(int(k) for k in res["spatial"] if k.isdigit())
    sig = np.array([res["spatial"][str(l)]["signal_exact_mean"] for l in lags])
    t_sig = res["temporal"]["all_cells"]["signal_exact_mean"]
    order = np.argsort(sig)
    # np.interp clamps: a temporal change SMALLER than the adjacent-cell difference would read as
    # "27 km" with lag 1's retention. Flag it, and extrapolate linearly from the two smallest sizes.
    below = bool(t_sig < sig[order][0])
    res["matched"] = {"temporal_signal": t_sig, "below_smallest_lag": below,
                      "km_equivalent": float(np.interp(t_sig, sig[order],
                                                       np.array(lags)[order] * CELL_KM))}

    def at_size(y):
        if below and len(order) > 1:
            s0, s1 = sig[order][:2]
            return float(y[order][0] + (t_sig - s0) * (y[order][1] - y[order][0]) / (s1 - s0))
        return float(np.interp(t_sig, sig[order], y[order]))
    if below:
        res["matched"]["km_equivalent_extrapolated"] = at_size(np.array(lags, float) * CELL_KM)
    for r in ranks:
        ret = np.array([res["spatial"][str(l)]["esk"][str(r)]["retention"] for l in lags])
        res["matched"][str(r)] = {
            "temporal_retention": res["temporal"]["all_cells"]["esk"][str(r)]["retention"],
            "spatial_retention_at_temporal_size": at_size(ret)}
    with open(os.path.join(a.out, "atlas.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "atlas.json"), encoding="utf-8"))
    f = lambda v: "-" if v is None else f"{v:.3f}"
    t = r["temporal"]["all_cells"]
    m0 = r["matched"]
    km = (f"less than one cell ({CELL_KM:.0f} km) of spatial turnover, ~"
          f"{m0['km_equivalent_extrapolated']:.0f} km extrapolated" if m0.get("below_smallest_lag")
          else f"~{m0['km_equivalent']:.0f} km of spatial turnover")
    print(f"atlas [{r['population']}] temporal pairs {t['n_pairs']}, exact signal "
          f"{t['signal_exact_mean']:.4f}+-{t['signal_exact_se']:.4f}; {km}")
    for rk in r["ranks"]:
        m = r["matched"][str(rk)]
        print(f"  ESK r{rk:<3d} retention: temporal {f(m['temporal_retention'])}  spatial-at-same-size "
              f"{f(m['spatial_retention_at_temporal_size'])}")
    for lag, s in r["spatial"].items():
        if lag.isdigit():
            print(f"  spatial lag {lag:>2s} ({s['n_pairs']} pairs) signal {s['signal_exact_mean']:.4f} "
                  "ret " + " ".join(f"r{k}:{f(v['retention'])}" for k, v in s["esk"].items()))
    for w in ("ema", "raw"):
        d = r["temporal"].get(f"heldout_desk_{w}", {})
        if d.get("desk"):
            print(f"  DESK {w} held-out temporal ({d['n_pairs']} cells): " + "  ".join(
                f"r{k} slope {f(v['slope'])} corr {f(v['corr'])}" for k, v in d["desk"].items()))
    for lag, s in r["spatial"].items():
        if lag.endswith("heldout_desk_raw") and s.get("desk"):
            print(f"  DESK raw held-out spatial lag {lag.split('_')[0]} ({s['n_pairs']}): " + "  ".join(
                f"r{k} slope {f(v['slope'])} corr {f(v['corr'])}" for k, v in s["desk"].items()))


if __name__ == "__main__":
    main()
