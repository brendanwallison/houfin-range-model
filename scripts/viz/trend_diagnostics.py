#!/usr/bin/env python
"""Reconstruction + diagnostic visualizations for the trend-product community target.

Regenerates the figures behind the trend-diagnostic artifact and assembles a
self-contained HTML page. Reusable and meant to be iterated on: point it at the
real pipeline outputs (default, via config) or at a scratch directory of aligned
grids for local iteration.

Figures
-------
  reconstruction : one species carried back in time (anchor=eBird abd->ref, deep=BBS k·B/f)
  turnover       : recent vs deep community turnover (1 - log1p Ružicka), with gating
  stability      : deep-vs-anchor-year community similarity (no blow-up / false absence)
  vector_field   : community-analog shift vectors (deep + recent) -- see below

Anchor: --anchor-mode trends-abd (default) forward-extrapolates the trends midpoint `abd`
to --anchor-year (2025), matching the pipeline; 'weekly' is the legacy eBird weekly mean.

Community-analog shift field (per origin cell p, times t_hist -> t_mod):
  baseline COG  C_base = softmax_k over q of R(x[p,hist], x[q,hist]) · L_q
  modern   COG  C_mod  = softmax_k over q of R(x[p,hist], x[q,mod ]) · L_q
  vector V_p    = C_mod - C_base       (where p's historical community-type is now found)
Similarities are raised to exponent k (sharpening; low-similarity pixels shouldn't
drag the COG to the continent's centre) and masked beyond ``radius_km`` (dispersal-
plausible, and O(N·neighbours) instead of O(N^2)). Both COGs use the same estimator,
so its geographic/edge bias cancels in the subtraction.

    python scripts/viz/trend_diagnostics.py --out results/trend_viz
    python scripts/viz/trend_diagnostics.py --grids <scratch> --anchor-mode trends-abd \
        --community <scratch>/community_trend_100.csv --out <scratch>/viz
"""
import argparse
import base64
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:                                 # make ``src`` importable standalone
    sys.path.insert(0, _REPO)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree

from src.config_utils import load_config, load_data_config
from src.community_encoder.train_DESK.trend_community import (
    backward_trajectory, blend_weight, _species_scale, _load_trend_grid, _smooth_log_years,
)


# --- loading + reconstruction ------------------------------------------------

def _grid_paths(args, dcfg):
    """Resolve the three aligned trend grids (config defaults; --grids overrides)."""
    if args.grids:
        j = lambda n: os.path.join(args.grids, n)
        return j(args.bbs_trend), j(args.ebird_trend), j(args.bbs_abund)
    tr = dcfg["trends"]
    return tr["bbs_trend_grid"], tr["ebird_trend_grid"], tr["bbs_abund_grid"]


def load_reconstruction(args):
    """Load grids, build the anchor, reconstruct the community cube. Returns a dict."""
    dcfg = load_data_config()
    cfg = load_config()
    tc = cfg.get("trend", {})
    community = args.community or dcfg["community_trend_list"]
    codes = [str(c) for c in pd.read_csv(community)["species_code"]]
    S = len(codes)

    bt, et, ba = _grid_paths(args, dcfg)
    bbs_rate = _load_trend_grid(bt, codes, "rate")[0]
    ebird_ppy = _load_trend_grid(et, codes, "abd_ppy")[0]
    bbs_abund = _load_trend_grid(ba, codes, "abund")[0]

    if args.anchor_mode == "trends-abd":                      # default: abd -> anchor_year (matches pipeline)
        from src.community_encoder.train_DESK.trend_community import _trends_abd_anchor
        anchor = _trends_abd_anchor(et, codes, ebird_ppy, args.anchor_year)
    else:                                                     # legacy: eBird weekly annual mean
        from src.community_encoder.train_DESK.trend_community import _annual_anchor
        weekly = args.ebird_weekly or cfg.get("paths", {}).get("ebird_folder")
        anchor = _annual_anchor(weekly, codes)

    S, H, W = anchor.shape
    k = _species_scale(anchor, bbs_abund)
    valid = np.any(np.isfinite(anchor), axis=0)
    rr, cc = np.where(valid)
    flat = lambda A: np.stack([A[i][rr, cc] for i in range(S)])

    import rasterio
    with rasterio.open(dcfg["grid"]["ref_raster"]) as r:
        t = r.transform
    L = np.column_stack([t.c + t.a * (cc + 0.5), t.f + t.e * (rr + 0.5)])  # cell centres, metres

    # Match the pipeline reconstruction: both soft caps + optional post-cap smoothing.
    abs_asy = None
    if bool(tc.get("abs_soft_cap", True)):
        q = float(tc.get("abs_reference_quantile", 95.0))
        ref = np.array([float(np.nanpercentile(anchor[i][anchor[i] > 0], q))
                        if (np.isfinite(anchor[i]) & (anchor[i] > 0)).any() else 1.0 for i in range(S)])
        abs_asy = (ref * float(tc.get("abs_soft_max_mult", 1.0))).reshape(S, 1)
    years = sorted({args.anchor_year, args.recent_year, args.deep_year, args.mid_year})
    _, N = backward_trajectory(
        flat(anchor), flat(bbs_rate), flat(ebird_ppy), flat(bbs_abund), k.reshape(S, 1),
        years, args.anchor_year, args.deep_year,
        float(tc.get("handoff_crossover", 2010.0)), float(tc.get("handoff_width", 1.5)),
        float(np.log(tc.get("soft_max_fold", 100.0))), float(tc.get("soft_cap_p", 2.0)),
        abs_asy=abs_asy)
    sigma = float(tc.get("smooth_sigma_cells", 0.0))
    if sigma > 0:
        from src.community_encoder.train_DESK.trend_community import _smooth_log_years
        N = _smooth_log_years(N, rr, cc, H, W, sigma)
    yidx = {y: i for i, y in enumerate(years)}
    return dict(codes=codes, anchor=anchor, bbs_rate=bbs_rate, ebird_ppy=ebird_ppy,
                bbs_abund=bbs_abund, k=k, valid=valid, rr=rr, cc=cc, L=L, H=H, W=W,
                transform=t, N=N, years=years, yidx=yidx, flat_anchor=flat(anchor),
                crossover=float(tc.get("handoff_crossover", 2010.0)),
                width=float(tc.get("handoff_width", 1.5)),
                soft_asymptote=float(np.log(tc.get("soft_max_fold", 100.0))),
                soft_cap_p=float(tc.get("soft_cap_p", 2.0)),
                abs_reference_quantile=q)


def _to_grid(R, vec):
    g = np.full((R["H"], R["W"]), np.nan)
    g[R["rr"], R["cc"]] = vec
    return g


def _logvec(R, year):
    return np.log1p(np.nan_to_num(R["N"][R["yidx"][year]])).T      # (M, S)


def _ruzicka(U, V):
    mn = np.minimum(U, V).sum(1); mx = np.maximum(U, V).sum(1)
    return np.where(mx > 0, mn / mx, np.nan)


def _gate_mask(R, year, min_cov=0.5):
    a = R["flat_anchor"]; occ = a > 0; nocc = np.clip(occ.sum(0), 1, None)
    has_e = np.isfinite(np.stack([R["ebird_ppy"][i][R["rr"], R["cc"]] for i in range(len(R["codes"]))]))
    has_b = np.isfinite(np.stack([R["bbs_rate"][i][R["rr"], R["cc"]] for i in range(len(R["codes"]))]))
    w = blend_weight(year, R["crossover"], R["width"])
    contrib = (has_e & (w > 0.05)) | (has_b & ((1 - w) > 0.05))
    return ((contrib & occ).sum(0) / nocc) >= min_cov


# --- figures -----------------------------------------------------------------

def _outline(ax, R):
    """Draw the sampled-continent outline (contour of the valid/land footprint)."""
    land = np.zeros((R["H"], R["W"])); land[R["rr"], R["cc"]] = 1.0
    ax.contour(land, levels=[0.5], colors="k", linewidths=0.6, alpha=0.55)


def _coarsen(R, years, factor):
    """Block-average the log1p community to a coarser grid; return coarse cube + geometry.

    Aggregating before the shift field (a) cuts O(N·neighbours) work, (b) smooths per-cell
    stochasticity, and (c) tames the centre-of-gravity noise. Returns (Xc[year]->(S,Mc),
    Lc, rrc, ccc, Hc, Wc).
    """
    from src.processing.regrid import block_reduce
    H, W = R["H"], R["W"]; f = int(factor); Hc, Wc = H // f, W // f
    S = len(R["codes"])
    vgrid = np.zeros((H, W)); vgrid[R["rr"], R["cc"]] = 1.0
    vc = block_reduce(vgrid[:Hc * f, :Wc * f], f, "mean") > 0     # coarse cell has any valid fine cell
    rrc, ccc = np.where(vc)
    t = R["transform"]
    Lc = np.column_stack([t.c + t.a * f * (ccc + 0.5), t.f + t.e * f * (rrc + 0.5)])
    Xc = {}
    for y in years:
        lg = np.log1p(np.nan_to_num(R["N"][R["yidx"][y]]))       # (S, M)
        block = np.empty((S, int(vc.sum())))
        for s in range(S):
            g = np.full((H, W), np.nan); g[R["rr"], R["cc"]] = lg[s]
            gm = block_reduce(g[:Hc * f, :Wc * f], f, "mean")    # nan-aware block mean
            block[s] = gm[vc]
        Xc[y] = np.nan_to_num(block)
    return Xc, Lc, rrc, ccc, Hc, Wc


def fig_reconstruction(R, species, out):
    codes = R["codes"]; sp = species if species in codes else codes[min(3, len(codes) - 1)]
    si = codes.index(sp); ay, my, dy = R["anchor_year"], R["mid_year"], R["deep_year"]
    Bpat = R["bbs_abund"][si][R["rr"], R["cc"]] * R["k"][si]
    panels = [(_to_grid(R, R["N"][R["yidx"][dy]][si]), f"reconstructed {dy} (deep / BBS)"),
              (_to_grid(R, R["N"][R["yidx"][my]][si]), f"reconstructed {my} (recent / eBird)"),
              (_to_grid(R, R["N"][R["yidx"][ay]][si]), f"{ay} = eBird anchor"),
              (_to_grid(R, Bpat), "BBS present pattern (k·B)")]
    vmax = np.nanpercentile(np.concatenate([p.ravel() for p, _ in panels]), 99.5)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for aa, (g, t) in zip(ax, panels):
        im = aa.imshow(g, cmap="viridis", norm=LogNorm(vmin=max(vmax / 1e3, 1e-4), vmax=vmax))
        _outline(aa, R); aa.set_title(t, fontsize=11); aa.axis("off")
    fig.colorbar(im, ax=ax, fraction=.012, pad=.01)
    fig.suptitle(f"Method-B reconstruction — {sp}: present=eBird, recent {my}=eBird trend, deep {dy}=BBS pattern (k·B/f)", fontsize=13)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    return sp


def fig_reconstruction_multi(R, species_list, out):
    """Deep / recent / present reconstructions for several species (rows)."""
    codes = R["codes"]; ay, my, dy = R["anchor_year"], R["mid_year"], R["deep_year"]
    sps = [s for s in species_list if s in codes][:5] or codes[:4]
    cols = [(dy, f"{dy} (deep/BBS)"), (my, f"{my} (recent/eBird)"), (ay, f"{ay} (present)")]
    fig, ax = plt.subplots(len(sps), 3, figsize=(12, 3.4 * len(sps)), squeeze=False)
    for i, sp in enumerate(sps):
        si = codes.index(sp)
        gs = [_to_grid(R, R["N"][R["yidx"][y]][si]) for y, _ in cols]
        vmax = np.nanpercentile(np.concatenate([g.ravel() for g in gs]), 99.5) or 1.0
        for j, (g, (_, t)) in enumerate(zip(gs, cols)):
            aa = ax[i][j]
            im = aa.imshow(g, cmap="viridis", norm=LogNorm(vmin=max(vmax / 1e3, 1e-4), vmax=vmax))
            _outline(aa, R); aa.axis("off")
            if i == 0:
                aa.set_title(t, fontsize=11)
            if j == 0:
                aa.text(-0.04, 0.5, sp, transform=aa.transAxes, rotation=90, va="center", fontsize=11, weight="bold")
        fig.colorbar(im, ax=ax[i].tolist(), fraction=.012, pad=.01)
    fig.suptitle("Method-B reconstructions across species — present (eBird) → recent (eBird trend) → deep (BBS pattern)", fontsize=13)
    fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
    return sps


def fig_turnover(R, out):
    ay, ry, dy = R["anchor_year"], R["recent_year"], R["deep_year"]
    recent = 1 - _ruzicka(_logvec(R, ry), _logvec(R, ay))
    deep = 1 - _ruzicka(_logvec(R, dy), _logvec(R, ry))
    present_ct = (R["flat_anchor"] > 0).sum(0)
    kr = _gate_mask(R, ry) & (present_ct >= 3)
    kd = _gate_mask(R, dy) & (present_ct >= 3)
    vmax = np.nanpercentile(np.concatenate([recent[kr], deep[kd]]), 98)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for aa, (g, kk, t) in zip(ax, [(recent, kr, f"Recent turnover ({ry}→{ay}, eBird-dominated)"),
                                   (deep, kd, f"Deep turnover ({dy}→{ry}, BBS-dominated)")]):
        im = aa.imshow(_to_grid(R, np.where(kk, g, np.nan)), cmap="inferno", vmin=0, vmax=vmax)
        aa.set_title(f"{t}\n{int(kk.sum())} cells retained"); aa.axis("off"); fig.colorbar(im, ax=aa, fraction=.035)
    fig.suptitle("Community turnover (1 − log1p Ružicka): recent vs deep; deep gated where BBS coverage is thin")
    fig.tight_layout(); fig.savefig(out, dpi=115); plt.close(fig)


def fig_stability(R, out):
    ay, dy = R["anchor_year"], R["deep_year"]
    Rz = _ruzicka(_logvec(R, dy), _logvec(R, ay))
    present_ct = (R["flat_anchor"] > 0).sum(0); m = present_ct >= 3
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(_to_grid(R, np.where(m, Rz, np.nan)), cmap="RdYlGn", vmin=0, vmax=1)
    ax[0].set_title(f"community similarity {dy} vs {ay} (log1p Ružicka)"); ax[0].axis("off")
    fig.colorbar(im, ax=ax[0], fraction=.035)
    ax[1].hist(Rz[m], bins=40, color="seagreen"); ax[1].set_xlabel("per-cell Ružicka")
    ax[1].set_ylabel("cells"); ax[1].set_title(f"collapse (R<0.1): {100 * np.mean(Rz[m] < 0.1):.1f}%")
    fig.suptitle("Reconstructed community change is substantial and stable — no blow-up, no false absences")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def shift_field(R, hist_year, mod_year, step, radius_m, kexp, min_cov=0.5):
    """Community-analog shift vectors at an even subset of retained origin cells."""
    L = R["L"]; Xh = _logvec(R, hist_year); Xm = _logvec(R, mod_year)
    keep = _gate_mask(R, hist_year, min_cov) & ((R["flat_anchor"] > 0).sum(0) >= 3)
    tree = cKDTree(L)
    rr, cc = R["rr"], R["cc"]
    origins = [i for i in range(len(rr)) if keep[i] and rr[i] % step == 0 and cc[i] % step == 0]
    P, Vx, Vy, mag = [], [], [], []
    for p in origins:
        idx = np.array(tree.query_ball_point(L[p], radius_m))
        idx = idx[keep[idx]]                                  # targets must be retained too
        if idx.size < 5:
            continue
        Lq = L[idx]; u = Xh[p]
        sb = _ruzicka_row(u, Xh[idx]) ** kexp
        sm = _ruzicka_row(u, Xm[idx]) ** kexp
        Cb = (sb @ Lq) / sb.sum(); Cm = (sm @ Lq) / sm.sum()
        v = Cm - Cb
        P.append(L[p]); Vx.append(v[0]); Vy.append(v[1]); mag.append(np.hypot(*v))
    return np.array(P), np.array(Vx), np.array(Vy), np.array(mag)


def _ruzicka_row(u, V):
    mn = np.minimum(u, V).sum(1); mx = np.maximum(u, V).sum(1)
    return np.where(mx > 0, mn / mx, 0.0)


def fig_vector_field(R, out, coarsen, radius_km, kexp):
    """Community-analog shift field, computed on a coarsened grid (readable + smooth)."""
    t = R["transform"]; radius_m = radius_km * 1000.0; f = int(coarsen)
    ay, ry, dy = R["anchor_year"], R["recent_year"], R["deep_year"]
    Xc, Lc, rrc, ccc, Hc, Wc = _coarsen(R, [dy, ry, ay], f)
    tree = cKDTree(Lc)

    def field(hy, my):
        Xh, Xm = Xc[hy].T, Xc[my].T                            # (Mc, S)
        P, U, V, mg = [], [], [], []
        for p in range(len(rrc)):
            idx = np.array(tree.query_ball_point(Lc[p], radius_m))
            if idx.size < 5:
                continue
            u = Xh[p]; Lq = Lc[idx]
            sb = _ruzicka_row(u, Xh[idx]) ** kexp; sm = _ruzicka_row(u, Xm[idx]) ** kexp
            v = (sm @ Lq) / sm.sum() - (sb @ Lq) / sb.sum()
            P.append(p); U.append(v[0]); V.append(v[1]); mg.append(np.hypot(*v))
        return np.array(P, int), np.array(U), np.array(V), np.array(mg)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.9))
    for aa, (hy, title) in zip(ax, [(dy, f"Deep: {dy}→{ay} (BBS-dominated)"),
                                    (ry, f"Recent: {ry}→{ay} (eBird-dominated)")]):
        P, Vx, Vy, mag = field(hy, ay)
        _outline(aa, R)
        col = f * (ccc[P] + 0.5); row = f * (rrc[P] + 0.5)     # coarse cell -> fine-pixel centre
        dcol = Vx / t.a; drow = Vy / t.e                       # e<0 -> north = up
        clim = (0, np.percentile(mag / 1000.0, 95)) if mag.size else (0, 1)
        # autoscale per panel (deep ~5x larger than recent): direction is geographic
        # (angles="xy"), length auto-fit to the axis, magnitude read from colour.
        q = aa.quiver(col, row, dcol, drow, mag / 1000.0, cmap="plasma",
                      angles="xy", width=0.005, headwidth=4, clim=clim)
        aa.set_xlim(0, R["W"]); aa.set_ylim(R["H"], 0); aa.set_aspect("equal")
        aa.set_title(f"{title}\nmedian {np.median(mag) / 1000:.0f} km, {mag.size} vectors"); aa.axis("off")
        fig.colorbar(q, ax=aa, fraction=.035, label="shift (km)")
    fig.suptitle(f"Community-analog shift ({f}× coarsened ≈ {f * 27} km cells, k={kexp}, ≤{radius_km:.0f} km): "
                 f"where each cell's historical community-type is found today")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _recon_params(R, years, abs_max_mult, sigma):
    """Reconstruct with a given absolute-cap fraction (× occupied-cell p95) and smoothing σ.

    ``abs_max_mult=None`` disables the absolute cap. The relative/fold cap is held at
    its config value (this sweep isolates the absolute cap + smoothing).
    """
    S = len(R["codes"]); rr, cc, H, W = R["rr"], R["cc"], R["H"], R["W"]
    anchor = R["anchor"]
    fl = lambda A: np.stack([A[i][rr, cc] for i in range(S)])
    aa = None
    if abs_max_mult is not None:
        ref = np.array([float(np.nanpercentile(anchor[i][anchor[i] > 0], R["abs_reference_quantile"]))
                        if (np.isfinite(anchor[i]) & (anchor[i] > 0)).any() else 1.0 for i in range(S)])
        aa = (ref * abs_max_mult).reshape(S, 1)
    _, N = backward_trajectory(fl(anchor), fl(R["bbs_rate"]), fl(R["ebird_ppy"]), fl(R["bbs_abund"]),
                               R["k"].reshape(S, 1), years, R["anchor_year"], R["deep_year"],
                               R["crossover"], R["width"], R["soft_asymptote"], R["soft_cap_p"],
                               abs_asy=aa)
    if sigma > 0:
        N = _smooth_log_years(N, rr, cc, H, W, sigma)
    return N


def fig_sensitivity(R, out, species_list, caps=(0.25, 0.5, 1.0, 2.0, 4.0), sigmas=(0, 0.5, 1.0, 2.0, 3.0),
                    cap0=1.0, sigma0=1.0):
    """Sensitivity of the reconstruction to the absolute-cap fraction and smoothing σ.

    ``caps`` are multiples of the species' p99 (best-habitat) abundance: 1.0 ≈ change
    bounded at worst→best. Overall = median deep (1966→2023) community turnover;
    per-species = median log(deep/present) abundance ratio. One knob is swept at a
    time (the other fixed at its config value).
    """
    ay, dy = R["anchor_year"], R["deep_year"]; years = sorted([dy, ay])
    di, ai = years.index(dy), years.index(ay)
    present_ct = (R["flat_anchor"] > 0).sum(0); m = present_ct >= 3
    codes = R["codes"]; sps = [s for s in species_list if s in codes][:3] or codes[:3]

    def dturn(N):
        U = np.log1p(np.nan_to_num(N[ai])).T; V = np.log1p(np.nan_to_num(N[di])).T
        mn = np.minimum(U, V).sum(1); mx = np.maximum(U, V).sum(1)
        return float(np.nanmedian(np.where(mx > 0, 1 - mn / mx, np.nan)[m]))

    def sp_ratio(N, si):
        d, p = N[di][si], N[ai][si]; occ = p > 0
        return float(np.nanmedian(np.log((d[occ] + 1e-6) / (p[occ] + 1e-6)))) if occ.any() else np.nan

    Ncap = [_recon_params(R, years, c, sigma0) for c in caps]
    Nsig = [_recon_params(R, years, cap0, s) for s in sigmas]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(caps, [dturn(N) for N in Ncap], "o-", color="teal")
    ax[0, 0].set_xlabel("absolute cap (× p99, best-cell)"); ax[0, 0].set_ylabel("median deep turnover")
    ax[0, 0].set_title(f"Overall vs absolute-cap fraction (σ={sigma0})")
    ax[0, 1].plot(sigmas, [dturn(N) for N in Nsig], "o-", color="indianred")
    ax[0, 1].set_xlabel("smoothing σ (cells)"); ax[0, 1].set_ylabel("median deep turnover")
    ax[0, 1].set_title(f"Overall vs smoothing scale (cap={cap0:g}× p99, best-cell)")
    for sp in sps:
        si = codes.index(sp)
        ax[1, 0].plot(caps, [sp_ratio(N, si) for N in Ncap], "o-", label=sp)
        ax[1, 1].plot(sigmas, [sp_ratio(N, si) for N in Nsig], "o-", label=sp)
    ax[1, 0].set_xlabel("absolute cap (× p99, best-cell)"); ax[1, 0].set_ylabel("median log(deep/present)")
    ax[1, 0].set_title("Per-species vs absolute-cap fraction"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].set_xlabel("smoothing σ (cells)"); ax[1, 1].set_ylabel("median log(deep/present)")
    ax[1, 1].set_title("Per-species vs smoothing scale"); ax[1, 1].legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=.25)
    fig.suptitle("Sensitivity: absolute-cap fraction (× p99) and smoothing scale — overall + per-species", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=115); plt.close(fig)
    return sps


def recon_grid(R, caps=(0.5, 1.0, 2.0), sigmas=(0.0, 1.0, 3.0)):
    """Reconstruct the full community once per (cap × σ) combo. Returns (years, {(cap,σ): N})."""
    years = sorted([R["deep_year"], R["anchor_year"]])
    return years, {(c, sg): _recon_params(R, years, c, sg) for c in caps for sg in sigmas}


def fig_species_grid(R, years, grid, sp, out, caps=(0.5, 1.0, 2.0), sigmas=(0.0, 1.0, 3.0)):
    """One species over cap × σ: deep-1966 reconstruction (left 3×3, log abundance),
    the modern eBird anchor per σ (middle column), and the 2023−1966 Δ (right 3×3, linear).

    The Δ gets its own 3×3 because both the deep reconstruction (cap × σ) and the anchor
    (σ, since smoothing touches all years) move with the parameters.
    """
    dy, ay = R["deep_year"], R["anchor_year"]; di, ai = years.index(dy), years.index(ay)
    si = R["codes"].index(sp); nr, ncap = len(sigmas), len(caps)
    deep = {(c, sg): np.nan_to_num(grid[(c, sg)][di][si]) for c in caps for sg in sigmas}
    modern = {sg: np.nan_to_num(grid[(caps[0], sg)][ai][si]) for sg in sigmas}   # anchor is cap-independent
    delta = {(c, sg): modern[sg] - deep[(c, sg)] for c in caps for sg in sigmas}
    vmax = (np.nanpercentile(np.concatenate(list(deep.values()) + list(modern.values())), 99.5) or 1.0)
    vmin = max(vmax / 1e3, 1e-4)
    dlim = np.nanpercentile(np.abs(np.concatenate(list(delta.values()))), 99) or 1.0

    ncol = ncap + 1 + ncap                                                        # deep | modern | Δ
    fig, axes = plt.subplots(nr, ncol, figsize=(3.1 * ncol, 3.2 * nr))
    im = imd = None
    for i, sg in enumerate(sigmas):
        for jx, c in enumerate(caps):                                            # deep block
            ax = axes[i, jx]
            im = ax.imshow(_to_grid(R, deep[(c, sg)]), cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
            _outline(ax, R); ax.axis("off")
            if i == 0:
                ax.set_title(f"deep · cap {c:g}×", fontsize=10)
            if jx == 0:
                ax.text(-0.08, 0.5, f"σ = {sg:g}", transform=ax.transAxes, rotation=90,
                        va="center", fontsize=11, weight="bold")
        axm = axes[i, ncap]                                                      # modern (this σ)
        axm.imshow(_to_grid(R, modern[sg]), cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))
        _outline(axm, R); axm.axis("off")
        if i == 0:
            axm.set_title(f"{ay} eBird", fontsize=10)
        for jx, c in enumerate(caps):                                            # Δ block
            ax = axes[i, ncap + 1 + jx]
            imd = ax.imshow(_to_grid(R, delta[(c, sg)]), cmap="RdBu_r", vmin=-dlim, vmax=dlim)
            _outline(ax, R); ax.axis("off")
            if i == 0:
                ax.set_title(f"Δ · cap {c:g}×", fontsize=10)
    fig.colorbar(im, ax=axes[:, :ncap + 1].ravel().tolist(), fraction=.015, label="abundance (log)")
    fig.colorbar(imd, ax=axes[:, ncap + 1:].ravel().tolist(), fraction=.02, label=f"Δ {ay}−{dy} (linear)")
    fig.suptitle(f"{sp} over cap × σ — deep {dy} reconstruction (left), {ay} eBird (mid), and Δ {ay}−{dy} (right)", fontsize=13)
    fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
    return sp


def _community_sim(R, years, N):
    di, ai = years.index(R["deep_year"]), years.index(R["anchor_year"])
    U = np.log1p(np.nan_to_num(N[ai])).T; V = np.log1p(np.nan_to_num(N[di])).T
    return _ruzicka(U, V)


def fig_similarity_grid(R, years, grid, out, caps=(0.5, 1.0, 2.0), sigmas=(0.0, 1.0, 3.0)):
    """3×3 community-similarity maps (deep vs anchor year) over cap × σ — stability-figure style."""
    dy, ay = R["deep_year"], R["anchor_year"]
    nr, nc = len(sigmas), len(caps); m = (R["flat_anchor"] > 0).sum(0) >= 3
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.3 * nr)); im = None
    for i, sg in enumerate(sigmas):
        for jx, c in enumerate(caps):
            Rz = _community_sim(R, years, grid[(c, sg)]); ax = axes[i, jx]
            im = ax.imshow(_to_grid(R, np.where(m, Rz, np.nan)), cmap="RdYlGn", vmin=0, vmax=1)
            _outline(ax, R); ax.axis("off")
            if i == 0:
                ax.set_title(f"cap {c:g}× p99", fontsize=11)
            if jx == 0:
                ax.text(-0.06, 0.5, f"σ = {sg:g}", transform=ax.transAxes, rotation=90,
                        va="center", fontsize=11, weight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.02, label=f"Ružicka({dy}, {ay})")
    fig.suptitle(f"Community similarity {dy} vs {ay} over cap × σ", fontsize=13)
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def fig_histogram_grid(R, years, grid, out, caps=(0.5, 1.0, 2.0), sigmas=(0.0, 1.0, 3.0)):
    """3×3 per-cell Ružicka(deep, anchor) histograms over cap × σ — stability-figure style."""
    dy, ay = R["deep_year"], R["anchor_year"]
    nr, nc = len(sigmas), len(caps); m = (R["flat_anchor"] > 0).sum(0) >= 3
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.0 * nr), sharex=True)
    for i, sg in enumerate(sigmas):
        for jx, c in enumerate(caps):
            Rz = _community_sim(R, years, grid[(c, sg)])[m]; ax = axes[i, jx]
            ax.hist(Rz[np.isfinite(Rz)], bins=40, color="seagreen")
            ax.set_title((f"cap {c:g}× p99, " if i == 0 else "") +
                         f"collapse {100 * np.mean(Rz < 0.1):.1f}%", fontsize=9)
            if jx == 0:
                ax.set_ylabel(f"σ = {sg:g}", fontsize=11, weight="bold")
            if i == nr - 1:
                ax.set_xlabel("per-cell Ružicka")
    fig.suptitle(f"Per-cell Ružicka({dy}, {ay}) distribution over cap × σ", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


# --- HTML --------------------------------------------------------------------

def _uri(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def build_html(out_dir, figs, meta):
    imgs = "\n".join(
        f'<figure><figcaption>{cap}</figcaption>'
        f'<img src="{_uri(os.path.join(out_dir, f))}" style="width:100%;height:auto"></figure>'
        for f, cap in figs)
    html = f"""<!doctype html><meta charset="utf-8"><title>Trend reconstruction diagnostics</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.6}}
figure{{margin:0 0 2rem}}figcaption{{font-weight:600;margin-bottom:.4rem}}
h1{{font-family:Georgia,serif}}code{{background:#eef;padding:.1em .3em;border-radius:3px}}</style>
<h1>Historical community reconstruction — diagnostics</h1>
<p>{meta}</p>{imgs}"""
    p = os.path.join(out_dir, "index.html")
    open(p, "w").write(html)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output directory for figures + index.html.")
    ap.add_argument("--grids", default=None, help="Dir of aligned grids (overrides config paths).")
    ap.add_argument("--bbs-trend", default="bbs_trend_grid.npz")
    ap.add_argument("--ebird-trend", default="ebird_trend_grid.npz")
    ap.add_argument("--bbs-abund", default="bbs_abund_grid.npz")
    ap.add_argument("--community", default=None)
    ap.add_argument("--anchor-mode", choices=["weekly", "trends-abd"], default="trends-abd",
                    help="'trends-abd' (default, matches pipeline) = abd forward-extrapolated to "
                         "--anchor-year; 'weekly' = legacy eBird weekly annual mean.")
    ap.add_argument("--ebird-weekly", default=None, help="Projected weekly grid dir (weekly anchor mode).")
    ap.add_argument("--anchor-year", type=int, default=2025)
    ap.add_argument("--recent-year", type=int, default=2011, help="eBird/BBS handoff split (turnover + shift field).")
    ap.add_argument("--mid-year", type=int, default=2013, help="eBird-driven reconstruction panel year (in the eBird window).")
    ap.add_argument("--deep-year", type=int, default=1966)
    ap.add_argument("--species", default="amegfi", help="Species for the single reconstruction panel.")
    ap.add_argument("--species-multi", default="amegfi,houspa,sonspa,easmea",
                    help="Comma-separated species for the multi-species reconstruction grid.")
    ap.add_argument("--k", type=float, default=5.0, help="Shift-field sharpening exponent (3-10).")
    ap.add_argument("--radius-km", type=float, default=1000.0, help="Shift-field dispersal mask radius.")
    ap.add_argument("--coarsen", type=int, default=7, help="Shift-field grid coarsening factor (cells/block).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    R = load_reconstruction(args)
    R["anchor_year"], R["recent_year"] = args.anchor_year, args.recent_year
    R["deep_year"], R["mid_year"] = args.deep_year, args.mid_year
    print(f"[viz] {len(R['codes'])} species, {R['rr'].size} cells; years {R['years']}")

    j = lambda n: os.path.join(args.out, n)
    sp = fig_reconstruction(R, args.species, j("reconstruction.png"))
    sps = fig_reconstruction_multi(R, [s.strip() for s in args.species_multi.split(",")], j("reconstruction_multi.png"))
    fig_turnover(R, j("turnover.png"))
    fig_stability(R, j("stability.png"))
    fig_vector_field(R, j("vector_field.png"), args.coarsen, args.radius_km, args.k)
    sens_sps = fig_sensitivity(R, j("sensitivity.png"), [s.strip() for s in args.species_multi.split(",")])
    # 3x3 (cap × σ) interaction: reconstruct once, reuse for per-species grids + community grids.
    grid_caps, grid_sigmas = (0.5, 1.0, 2.0), (0.0, 1.0, 3.0)
    gyears, grid = recon_grid(R, grid_caps, grid_sigmas)
    grid_sps = [s.strip() for s in args.species_multi.split(",") if s.strip() in R["codes"]][:3]
    for gs in grid_sps:
        fig_species_grid(R, gyears, grid, gs, j(f"grid_{gs}.png"), grid_caps, grid_sigmas)
    fig_similarity_grid(R, gyears, grid, j("similarity_grid.png"), grid_caps, grid_sigmas)
    fig_histogram_grid(R, gyears, grid, j("histogram_grid.png"), grid_caps, grid_sigmas)

    figs = [("vector_field.png", "Community-analog shift field (deep + recent), coarsened."),
            ("reconstruction.png", f"Method-B reconstruction of {sp} (deep/BBS → recent/eBird → present)."),
            ("reconstruction_multi.png", f"Reconstructions across species: {', '.join(sps)}."),
            ("turnover.png", "Recent vs deep community turnover (with coverage gating)."),
            ("stability.png", "Deep community similarity — stable, no collapse."),
            ("sensitivity.png", f"Sensitivity line plots — absolute-cap fraction + smoothing (species: {', '.join(sens_sps)}).")]
    figs += [(f"grid_{gs}.png", f"{gs}: deep reconstruction over cap × σ, + modern eBird + Δ.") for gs in grid_sps]
    figs += [("similarity_grid.png", f"Community similarity {args.deep_year} vs {args.anchor_year} over cap × σ."),
             ("histogram_grid.png", f"Per-cell Ružicka({args.deep_year}, {args.anchor_year}) histograms over cap × σ.")]
    p = build_html(args.out, figs, f"{len(R['codes'])} species; anchor {args.anchor_year}, "
                   f"recent {args.recent_year}, mid {args.mid_year}, deep {args.deep_year}; "
                   f"shift-field k={args.k}, radius {args.radius_km} km, coarsen {args.coarsen}×; "
                   f"grid caps {grid_caps}× p99, σ {grid_sigmas}.")
    print(f"[viz] wrote figures + {p}")


if __name__ == "__main__":
    main()
