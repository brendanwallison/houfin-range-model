"""The validation results as maps of North America.

The chart suite (`validation_report.py`) made the numbers legible. It cannot answer where. And the
questions this project is stuck on are all "where": is the error at the expansion front, is it
inside the Great Plains barrier where `Z_disp` feeds `Q` and nothing downstream is fitted to
correct it, is the coastal pattern ecology or sampling. A median answers none of them.

THE DESIGN CONSTRAINT. A naive map throws away every device the metric suite uses to stay honest:
it shows one predictor's one number in one colour, with no floor, no ceiling, no ladder, and no
way to decline a cell it cannot resolve. Each map here carries at least one of those devices over:

  floor + ceiling            -> per-cell room, faded toward the background as room -> 0
  resolving_room's REFUSAL   -> unresolvable cells are hatched with the reason, never coloured
  the six-rung ladder        -> a winner-take-all map, saturation = margin
  the exact mag/ang split    -> direction and magnitude drawn together, never one alone
  structural n/a             -> hatched, never zero
  win rate on the finite     -> a common-support mask across the compared predictors
    intersection
  cells not pairs (the       -> held-out results aggregate to BLOCK; within-block texture is not
    bootstrap's rule)           evidence, since 217 held-out cells sit in ~87 blocks

Two things only a map can say, and neither has a scalar ancestor: where the experiment has any
power at all, and whether the error lines up with the ecology or with the sampling.

    python scripts/viz/validation_maps.py --run-dir <dir> [--out DIR]
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _geo                                    # noqa: E402
import _validation_load as L                   # noqa: E402
import _validation_style as S                  # noqa: E402
import matplotlib                             # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

#: Held-out blocks are 6x6 cells. Any held-out map aggregates to this, because cells inside one
#: block are not independent -- the same reason `bootstrap_skill_ci` resamples focal cells and
#: never pairs. 217 held-out cells are ~87 independent observations, not 217.
BLOCK = 6

#: A cell needs this many surveys before its per-cell statistic is drawn. Gate hard to NaN and
#: print the retained count, following `trend_diagnostics._gate_mask`; there is no house style for
#: alpha-by-n or stippling and a faded cell still invites reading its colour.
MIN_SUPPORT = 3


def _panel(geo, ncols=1, nrows=1, w=4.6):
    h = w * (geo.box_bounds[3] - geo.box_bounds[1]) / (geo.box_bounds[2] - geo.box_bounds[0])
    # +1.3 not +0.9: panel titles run to three lines (name, median, scale and n) and a tighter
    # allowance clips the first of them, which is the one that says what the panel is.
    fig, axs = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 1.3), squeeze=False)
    return fig, axs


def _draw(geo, ax, grid, title, cmap="magma", diverging=False, vmax=None, unit=None,
          vmin=None, cbar=True, fmt="{:+.3f}", center=0.0):
    """One map panel, per-panel scaled with its median in the title.

    The scale is annotated because a shared vmax across panels of different magnitude once
    rendered a predicted field at 20% of its range and read as "no change predicted" when it meant
    "much less change predicted" -- the rule is recorded on `validation_report.fig12_maps` and
    costs nothing to keep.
    """
    geo.basemap(ax)
    fin = np.isfinite(grid)
    if not fin.any():
        ax.set_title(f"{title}\n(no cells)", fontsize=8.5)
        geo.coastline(ax)
        return None
    if diverging:
        # `center` is the value the colour is neutral at, and it is not always zero: a magnitude
        # RATIO is neutral at 1.0 (predicted change the same size as observed), so centring it on
        # 0 would paint every cell that moved at all as "high" and hide the over/under-move split
        # the panel exists to show.
        v = vmax if vmax is not None else float(
            np.nanpercentile(np.abs(grid[fin] - center), 98))
        lo, hi = center - v, center + v
    else:
        # vmin is explicit for quantities whose zero is not meaningful -- a YEAR field stretched
        # from 0 puts every value in the top 3% of the ramp and shows nothing.
        lo = 0.0 if vmin is None else vmin
        hi = vmax if vmax is not None else float(np.nanpercentile(grid[fin], 98))
    im = geo.imshow(ax, grid, cmap=cmap, vmin=lo, vmax=hi)
    geo.coastline(ax)
    # The unit lives on the colorbar and nowhere else, per the house rule -- and a map without one
    # is a picture of a pattern with no way to say how big it is.
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.038, pad=0.02, shrink=0.72)
        cb.ax.tick_params(labelsize=6.5)
        if unit:
            cb.set_label(unit, fontsize=7)
    med = float(np.nanmedian(grid[fin]))
    scale = (f"{fmt.format(center).lstrip('+')} ± {fmt.format(v).lstrip('+')}" if diverging
             else f"{fmt.format(lo).lstrip('+')}–{fmt.format(hi).lstrip('+')}")
    ax.set_title(f"{title}\nmedian {fmt.format(med)}   scale {scale}   n={int(fin.sum()):,}",
                 fontsize=8.5)
    return im


def _blockify(grid, block=BLOCK):
    """Block means on the holdout lattice. The honest resolution of an out-of-sample map."""
    H, W = grid.shape
    ph, pw = (-H) % block, (-W) % block
    g = np.pad(grid, ((0, ph), (0, pw)), constant_values=np.nan)
    with np.errstate(invalid="ignore"):
        bm = np.nanmean(g.reshape(g.shape[0] // block, block, g.shape[1] // block, block),
                        axis=(1, 3))
    return np.kron(bm, np.ones((block, block)))[:H, :W]


# =================================================================================================

def map01_design(run, maps, geo, out_dir):
    """The experiment as geography: what was trained on, what was held out, and how thickly.

    The split is a config value until it is a picture. `block_cells=6` is 162 km and the buffer
    ring is the reason a held-out cell's receptive field never touched a training cell -- both are
    claims about distance on the ground, and neither is checkable from a number.
    """
    ho, bf = maps.get("holdout"), maps.get("buffer")
    if ho is None:
        return None
    sup = np.zeros(geo.shape, "float64")
    ep = maps.get("epoch")
    if ep is not None:
        sup[ep["rows"].astype(int), ep["cols"].astype(int)] = 1.0

    split = np.full(geo.shape, np.nan)
    if geo.land is not None:
        split[geo.land] = 0.0                                   # land, unsupervised
    split[sup > 0] = 1.0                                        # training
    if bf is not None:
        split[bf.astype(bool)] = 2.0
    split[ho.astype(bool)] = 3.0

    # Buffer width is MEASURED off the mask, not derived from BLOCK: it comes from the latent
    # conv kernel (kernel//2), not from the block size, and the two are different numbers that a
    # label computed from the wrong one would quietly misstate as a distance on the ground.
    buf_cells = 0
    if bf is not None and bf.any() and ho.any():
        # CHEBYSHEV, matching blocked_holdout's square structuring element. A Euclidean
        # transform reports the diagonal (2 cells dilated -> 2.83) and would round the label up
        # to a distance the split never used.
        from scipy.ndimage import distance_transform_cdt
        buf_cells = int(distance_transform_cdt(
            ~ho.astype(bool), metric="chessboard")[bf.astype(bool)].max())
    n_ho = int(ho.sum())
    n_blocks = int(np.ceil(n_ho / BLOCK ** 2))

    fig, axs = _panel(geo, ncols=2, w=5.0)
    a = axs[0][0]
    geo.basemap(a)
    cmap = ListedColormap(["#e6e2db", "#8fb4d6", "#d9c48a", "#c1440e"])
    geo.imshow(a, split, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4),
               mask_to_land=False)
    geo.coastline(a)
    geo.great_plains(a)
    geo.scalebar(a)
    a.set_title(f"the split on the ground\n{n_ho:,} held-out cells in ~{n_blocks} blocks",
                fontsize=8.5)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="none", label=l) for c, l in
                        (("#e6e2db", "land, no supervision"), ("#8fb4d6", "training cells"),
                         ("#d9c48a", f"buffer ring ({buf_cells} cells = "
                                     f"{buf_cells * geo.res_m / 1000:.0f} km)"),
                         ("#c1440e", f"held out ({BLOCK}x{BLOCK} = "
                                     f"{BLOCK * geo.res_m / 1000:.0f} km blocks)"))],
               fontsize=7, ncol=4, loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 1.005))

    # How MANY observations each cell contributes, which is what a per-cell statistic's precision
    # actually rests on -- and it is the confound to check before reading any spatial pattern.
    b = axs[0][1]
    st = maps.get("spacetime")
    if st is not None and "recon_rows" in st:
        n_cy = _geo.to_grid(st["recon_rows"].astype(int), st["recon_cols"].astype(int),
                            np.ones(len(st["recon_rows"])), geo.shape, reduce="count")
        n_cy = np.where(n_cy > 0, n_cy, np.nan)
        _draw(geo, b, n_cy, "graded cell-years per cell", cmap="Blues")
        geo.great_plains(b)
    else:
        geo.basemap(b)
        b.set_title("cell-year counts unavailable", fontsize=8.5)

    return S.finish(fig, os.path.join(out_dir, "m01_design.png"),
                    f"Held-out blocks are contiguous by construction, so the independent sample "
                    f"is ~{n_blocks} blocks and not {n_ho:,} cells — every out-of-sample panel "
                    f"below is drawn at block resolution for that reason. The dashed outline is "
                    f"the Great Plains barrier corridor, where Z_disp feeds the dispersal "
                    f"survival term and nothing downstream is fitted to correct it.",
                    tight_rect=(0, 0.09, 1, 0.93))


def map02_ceiling(run, maps, geo, out_dir):
    """Where can we tell anything at all?

    The per-cell noise floor: the similarity between two disjoint halves of the SAME cell in the
    SAME era, where no real turnover is possible, so everything below 1.0 is measurement noise. A
    cell surveyed twice has almost no resolving power and a cell surveyed twenty times has a lot,
    and that varies across the continent by more than the model's skill does. Read every other map
    through this one.
    """
    ep = maps.get("epoch")
    if ep is None or "floor_early" not in ep:
        return None
    r, c = ep["rows"].astype(int), ep["cols"].astype(int)
    ok = ep["split_ok"].astype(bool)
    fig, axs = _panel(geo, ncols=3, w=4.4)
    for ax, key, title in ((axs[0][0], "floor_early", "early era (1966–1986)"),
                           (axs[0][1], "floor_modern", "modern era (2005–2025)")):
        g = np.full(geo.shape, np.nan)
        g[r[ok], c[ok]] = ep[key][ok]
        _draw(geo, ax, g, f"noise floor — {title}", cmap="viridis", vmax=1.0)
        geo.great_plains(ax)
    room = np.full(geo.shape, np.nan)
    room[r[ok], c[ok]] = 1.0 - np.asarray(ep["floor_early"])[ok]
    _draw(geo, axs[0][2], room, "resolvable room  (1 − floor)", cmap="magma")
    geo.great_plains(axs[0][2])
    # REFUSED cells get their own flat colour and a legend entry rather than a hatch: at 27 km a
    # hatch over scattered single cells is illegible, and an illegible mark that means "we cannot
    # answer here" is worse than no mark, because the reader reads the colour underneath instead.
    unres = np.zeros(geo.shape, bool)
    unres[r[~ok], c[~ok]] = True
    if unres.any():
        axs[0][2].imshow(np.where(unres, 1.0, np.nan), extent=geo.extent, origin="upper",
                         cmap=ListedColormap(["#8f8f8f"]), zorder=4)
        axs[0][2].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#8f8f8f", ec="none",
                                                label="refused: too few surveys to split")],
                         fontsize=6.5, loc="lower left", frameon=False)
    return S.finish(fig, os.path.join(out_dir, "m02_ceiling.png"),
                    f"1.0 would be a noiseless cell. Hatched cells have too few surveys in an era "
                    f"to split in half, so no independent observation exists and no ceiling can be "
                    f"formed — they are refused, not scored. {int((~ok).sum()):,} of "
                    f"{len(ok):,} cells.")


def map03_ladder_winner(run, maps, geo, out_dir):
    """Which rung wins here — the ladder asked geographically.

    Each rung is handed different information, so the winner names which claim survives at that
    place: the cell's own modern state, its own other years, its neighbours' change, joint
    space-time interpolation, or the covariates. Saturation is the MARGIN over the runner-up, so a
    1% win does not look like a 50% win.
    """
    rows, bars = L.ladder_bars(maps)
    if rows is None or len(bars) < 2:
        return None
    names = [n for n in S.ordered(list(bars)) if n != "no_change"]
    stack = np.vstack([bars[n] for n in names])                 # (rungs, n_rows)
    finite = np.isfinite(stack)
    # COMMON SUPPORT: only rank a row where every rung reached it. Scoring a bar's easy subset
    # against a reference's full set is the flattering that `win_rate_vs` intersects to avoid.
    keep = finite.all(axis=0)
    if keep.sum() < 10:
        return None
    sub = stack[:, keep]
    win = np.argmin(sub, axis=0)
    srt = np.sort(sub, axis=0)
    margin = (srt[1] - srt[0]) / np.maximum(srt[1], 1e-9)

    r, c = rows[keep, 0].astype(int), rows[keep, 1].astype(int)
    win_g = _geo.to_grid(r, c, win.astype("float64"), geo.shape)
    marg_g = _geo.to_grid(r, c, margin, geo.shape)

    fig, axs = _panel(geo, ncols=2, w=5.0)
    a = axs[0][0]
    geo.basemap(a)
    # ALPHA on the category itself, not a white veil over it. The fade reference is the
    # seed-to-seed spread of a fixed configuration: a margin smaller than that is a tie, and the
    # threshold is the one the rest of the suite already uses rather than a number picked here.
    rgba = np.zeros(geo.shape + (4,))
    alpha = np.clip(marg_g / S.SEED_NOISE, 0.0, 1.0)
    for i, nm in enumerate(names):
        m = np.isfinite(win_g) & (np.round(win_g) == i)
        if m.any():
            rgba[m] = matplotlib.colors.to_rgba(S.color(nm))
    rgba[..., 3] = np.where(np.isfinite(win_g), np.nan_to_num(alpha), 0.0)
    a.imshow(rgba, extent=geo.extent, origin="upper", zorder=2)
    geo.coastline(a)
    geo.great_plains(a)
    n_tie = int(np.nansum(alpha[np.isfinite(win_g)] < 1.0))
    a.set_title(f"which rung predicts this cell best\nfaded = margin under "
                f"{100 * S.SEED_NOISE:.1f}% (a tie): {n_tie:,} of "
                f"{int(np.isfinite(win_g).sum()):,} cells", fontsize=8.5)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=S.color(n), ec="none", label=S.label(n))
                        for n in names],
               fontsize=7, ncol=min(len(names), 4), loc="upper center", frameon=False,
               bbox_to_anchor=(0.5, 1.005))

    # Who wins WHERE, as a share -- the categorical map's own summary, cut by the one ecological
    # partition this project has. A map shows the texture; this says whether the texture is a
    # pattern.
    b = axs[0][1]
    zones = geo.gp_zones or {}
    cols_z = [z for z in ("west", "barrier", "east") if z in zones]
    if cols_z:
        bottom = np.zeros(len(cols_z))
        for i, nm in enumerate(names):
            share = []
            for z in cols_z:
                m = zones[z] & np.isfinite(win_g)
                share.append(float(np.mean(np.round(win_g[m]) == i)) if m.any() else 0.0)
            b.bar(cols_z, share, bottom=bottom, color=S.color(nm), label=S.label(nm))
            bottom += np.asarray(share)
        b.set_ylabel("share of cells won", fontsize=8)
        b.set_ylim(0, 1)
        b.set_title("who wins where, by Great Plains zone", fontsize=8.5)
        b.tick_params(labelsize=8)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)
    else:
        b.set_axis_off()
        b.set_title("Great Plains zone raster absent", fontsize=8.5)

    return S.finish(fig, os.path.join(out_dir, "m03_ladder_winner.png"),
                    "Scored only on cell-years every rung reached, so no rung is credited for "
                    "declining the hard rows. Faded regions are ties, not wins — a bare "
                    "winner-take-all map shows a confident colour for a 1% margin, which is how "
                    "a noise field comes to look like a spatial finding.",
                    tight_rect=(0, 0.06, 1, 0.93))


def map04_vs_bar(run, maps, geo, out_dir):
    """DESK against the honest bar, where the comparison can resolve anything.

    `spacetime_idw` is the competitor; `no_change` is a decomposition device that assumes sixty
    years of stasis and is nearly free to beat. Cells where the two differ by less than the
    seed-to-seed spread of a fixed configuration are greyed: that difference is noise.
    """
    rows, bars = L.ladder_bars(maps)
    if rows is None or "desk" not in bars or "spacetime_idw" not in bars:
        return None
    d, b = bars["desk"], bars["spacetime_idw"]
    keep = np.isfinite(d) & np.isfinite(b)
    if keep.sum() < 10:
        return None
    rel = (b[keep] - d[keep]) / np.maximum(b[keep], 1e-9)        # >0 => DESK better
    r, c = rows[keep, 0].astype(int), rows[keep, 1].astype(int)
    g = _geo.to_grid(r, c, rel, geo.shape)
    ho = maps.get("holdout")

    fig, axs = _panel(geo, ncols=2, w=5.0)
    _draw(geo, axs[0][0], g, "DESK vs spacetime IDW  (blue = DESK better)",
          cmap="RdBu", diverging=True)
    geo.great_plains(axs[0][0])
    if ho is not None:
        axs[0][0].contour(ho.astype(float), levels=[0.5], colors="#111111", linewidths=0.5,
                          extent=geo.extent, origin="upper", zorder=7)
    # Held out only, at BLOCK resolution -- the honest out-of-sample picture.
    gh = np.full(geo.shape, np.nan)
    if ho is not None:
        m = ho.astype(bool)
        gh = np.where(m, g, np.nan)
        gh = _blockify(gh)
        gh = np.where(_blockify(np.where(m, 1.0, np.nan)) > 0, gh, np.nan)
    quiet = np.abs(gh) < S.SEED_NOISE
    shown = np.where(quiet, np.nan, gh)
    n_block = int(np.isfinite(gh).sum()) // (BLOCK ** 2)
    n_kept = int(np.isfinite(shown).sum()) // (BLOCK ** 2)
    if np.isfinite(shown).any():
        _draw(geo, axs[0][1], shown, f"held-out blocks only\n{n_kept} of ~{n_block} blocks "
                                     f"exceed the seed spread", cmap="RdBu", diverging=True)
    else:
        # "(no cells)" would read as a broken panel. Every block being below the noise floor is a
        # RESULT -- it says the two predictors are indistinguishable out of sample -- and the
        # panel has to say that rather than look empty.
        geo.basemap(axs[0][1])
        geo.coastline(axs[0][1])
        axs[0][1].set_title(f"held-out blocks only\nnone of ~{n_block} blocks differ by more "
                            f"than the {100 * S.SEED_NOISE:.1f}% seed spread", fontsize=8.5)
        axs[0][1].text(0.5, 0.42, "indistinguishable out of sample", transform=axs[0][1].transAxes,
                       ha="center", fontsize=9, color="#8a3208")
    geo.great_plains(axs[0][1])
    return S.finish(fig, os.path.join(out_dir, "m04_vs_bar.png"),
                    f"Right panel is aggregated to the {BLOCK}x{BLOCK} holdout block because cells "
                    f"within a block are not independent, and blank where the two predictors "
                    f"differ by less than the {100 * S.SEED_NOISE:.1f}% seed-to-seed spread of a "
                    f"fixed configuration. A difference that small is not an effect.")


def map05_direction(run, maps, geo, out_dir):
    """Direction and magnitude, per epoch pair, each against its own per-cell ceiling.

    Never one without the other: they are the two halves of an exact split of the change error and
    they trade off, since shrinking is the MSE-optimal response to a poor angle. And never against
    1.0: the ceiling is what an INDEPENDENT observation of the same place scores, which is set by
    how often that cell was surveyed.
    """
    pairs = L.epoch_pairs_available(maps)
    if not pairs:
        return None
    pair = pairs[-1]
    d = L.epoch_pair_cells(maps, pair)
    if "cells" not in d or "dir_cos" not in d:
        return None
    cells = d["cells"].astype(int)
    r, c = cells[:, 0], cells[:, 1]
    ceil = np.full(len(cells), np.nan)
    if "ceiling_per_cell" in d and "ceiling_cell_idx" in d:
        ceil[d["ceiling_cell_idx"].astype(int)] = d["ceiling_per_cell"]

    fig, axs = _panel(geo, ncols=3, w=4.4)
    _draw(geo, axs[0][0], _geo.to_grid(r, c, d["dir_cos"], geo.shape),
          f"direction cosine  {pair.replace('_', '→')}", cmap="RdBu", diverging=True, vmax=1.0)
    _draw(geo, axs[0][1], _geo.to_grid(r, c, d.get("mag_ratio", np.full(len(r), np.nan)),
                                       geo.shape),
          "magnitude ratio  ‖Δpred‖/‖Δtruth‖\n(1.0 = right amount of change)",
          cmap="PuOr_r", diverging=True, vmax=1.0, center=1.0)
    share = np.where(np.isfinite(ceil) & (ceil > 1e-6), d["dir_cos"] / ceil, np.nan)
    _draw(geo, axs[0][2], _geo.to_grid(r, c, share, geo.shape),
          "share of the per-cell ceiling", cmap="RdBu", diverging=True, vmax=1.0)
    for ax in axs[0]:
        geo.great_plains(ax)
    return S.finish(fig, os.path.join(out_dir, "m05_direction.png"),
                    f"One epoch pair ({pair.replace('_', '→')}), never pooled — the pairs share "
                    f"cells and nest in time, so an average would overstate the evidence. A "
                    f"magnitude ratio above 1 is over-moving; the right panel is the only one of "
                    f"the three that is comparable between cells, because it divides by what each "
                    f"cell could achieve.")


def map06_ecology(run, maps, geo, out_dir):
    """Does the error follow the ecology, or the sampling?

    The Great Plains corridor is where `Z_disp` feeds `Q`, which is applied after convolution,
    never renormalised, and corrected by no data term anywhere — the highest-leverage error region
    in the project. The disease front is the other candidate structure, and the two are very
    nearly the same cut, which the caption says so it is not read as two pieces of evidence.
    """
    st = maps.get("spacetime")
    if st is None or "recon_err_desk" not in st:
        return None
    r, c = st["recon_rows"].astype(int), st["recon_cols"].astype(int)
    err = _geo.to_grid(r, c, st["recon_err_desk"], geo.shape)
    n = _geo.to_grid(r, c, np.ones_like(st["recon_err_desk"], dtype="float64"), geo.shape,
                     reduce="count")
    err, note = _geo.gate(err, n, MIN_SUPPORT, "cells")

    fig, axs = _panel(geo, ncols=3, w=4.4)
    _draw(geo, axs[0][0], err, f"DESK reconstruction error\n{note}", cmap="magma")
    geo.great_plains(axs[0][0])
    geo.scalebar(axs[0][0])
    if geo.front is not None:
        # The colonization front as a contour ON the error map, so the reader can see whether the
        # error tracks it without holding two panels in their head.
        axs[0][0].contour(geo.front, levels=[1970, 1980, 1990], colors=["#1f6fb4"],
                          linewidths=0.7, extent=geo.extent, origin="upper", zorder=6)

    b = axs[0][1]
    zones = geo.gp_zones or {}
    cols_z = [z for z in ("west", "barrier", "east") if z in zones]
    if cols_z:
        vals, labels = [], []
        for zname in cols_z:
            m = zones[zname] & np.isfinite(err)
            if m.any():
                vals.append(err[m])
                labels.append(f"{zname}\nn={int(m.sum()):,}")
        if vals:
            bp = b.boxplot(vals, tick_labels=labels, showfliers=False, patch_artist=True)
            for patch, col in zip(bp["boxes"], ("#8fb4d6", "#d9c48a", "#7fbcab")):
                patch.set_facecolor(col)
            b.set_ylabel("DESK reconstruction error", fontsize=8)
    b.set_title("error by Great Plains zone", fontsize=8.5)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)

    # AGAINST THE FRONT. This is the one panel here that asks a mechanistic question rather than a
    # geographic one: a range model that is worst exactly where the range was moving is failing at
    # the thing it exists to do, and that is not visible in any zone-stratified number.
    c_ax = axs[0][2]
    if geo.front is not None:
        edges = [1966, 1975, 1985, 1995, 2026]
        vals, labels = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = np.isfinite(err) & np.isfinite(geo.front) & (geo.front >= lo) & (geo.front < hi)
            if m.sum() >= 5:
                vals.append(err[m])
                labels.append(f"{lo}–{hi - 1}\nn={int(m.sum()):,}")
        if vals:
            c_ax.boxplot(vals, tick_labels=labels, showfliers=False)
            c_ax.set_xlabel("year the finch was first detected there", fontsize=8)
            c_ax.set_ylabel("DESK reconstruction error", fontsize=8)
        c_ax.set_title("error against the colonization front\n(eastern expansion only)",
                       fontsize=8.5)
        for sp in ("top", "right"):
            c_ax.spines[sp].set_visible(False)
    else:
        c_ax.text(0.5, 0.5, "colonization front absent\n"
                            "run scripts/build_colonization_front.py",
                  ha="center", va="center", transform=c_ax.transAxes, fontsize=8,
                  color="#999999")
        c_ax.set_axis_off()

    return S.finish(fig, os.path.join(out_dir, "m06_ecology.png"),
                    "The Great Plains zones and the disease-arrival front are NOT independent "
                    "cuts: median arrival is 1997 in the east, 2002 in the barrier and 2005 in "
                    "the west, so a difference across zones and a difference across disease eras "
                    "are largely the same difference. Cells below the coverage gate are blank "
                    "rather than dark. The blue contours on the map are the 1970/1980/1990 "
                    "colonization fronts; the native western range carries no front and is "
                    "excluded from that panel by construction.")


def map07_depth(run, maps, geo, out_dir):
    """How far each cell sits from anything the model was trained on.

    The by-distance table as a field. Extrapolation depth is the sweep's real independent
    variable and it is a distance on the ground, so it belongs on a map.
    """
    ho, bf = maps.get("holdout"), maps.get("buffer")
    st = maps.get("spacetime")
    if ho is None or st is None or "recon_err_desk" not in st:
        return None
    from scipy.ndimage import distance_transform_edt
    train = ~ho.astype(bool)
    if bf is not None:
        train &= ~bf.astype(bool)
    if geo.land is not None:
        train &= geo.land
    depth_km = distance_transform_edt(~train) * geo.res_m / 1000.0
    depth_km = np.where(ho.astype(bool), depth_km, np.nan)

    r, c = st["recon_rows"].astype(int), st["recon_cols"].astype(int)
    err = _geo.to_grid(r, c, st["recon_err_desk"], geo.shape)

    fig, axs = _panel(geo, ncols=2, w=5.0)
    _draw(geo, axs[0][0], depth_km, "km to the nearest training cell\n(held-out cells only)",
          cmap="cividis")
    geo.great_plains(axs[0][0])
    b = axs[0][1]
    m = np.isfinite(depth_km) & np.isfinite(err)
    if m.sum() >= 20:
        bins = np.array([0, 27, 54, 81, 108, 1e9])
        lab = ["0–27", "27–54", "54–81", "81–108", ">108"]
        vals = [err[m & (depth_km >= bins[i]) & (depth_km < bins[i + 1])]
                for i in range(len(lab))]
        keep = [i for i, v in enumerate(vals) if len(v) >= 5]
        b.boxplot([vals[i] for i in keep],
                  tick_labels=[f"{lab[i]}\nn={len(vals[i]):,}" for i in keep], showfliers=False)
        b.set_xlabel("km to the nearest training cell", fontsize=8)
        b.set_ylabel("DESK reconstruction error", fontsize=8)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)
    else:
        b.set_axis_off()
    b.set_title("error against extrapolation depth", fontsize=8.5)
    deepest = float(np.nanmax(depth_km)) if np.isfinite(depth_km).any() else float("nan")
    return S.finish(fig, os.path.join(out_dir, "m07_depth.png"),
                    f"Depth is capped by the split's own geometry — the deepest held-out cell in "
                    f"this run sits {deepest:.0f} km from any training data, and that is the most "
                    f"this experiment can test. A block-holdout says nothing about extrapolating "
                    f"further than its own blocks are wide.")


def map08_time(run, maps, geo, out_dir):
    """The error through time, and when each cell first came right.

    A single map cannot carry sixty years, and a sixty-year mean averages the extrapolated deep
    past together with the anchor year. Per-era panels separate them; the last panel compresses
    time into colour by asking when a cell's error first fell below its OWN ceiling — a threshold
    the data sets rather than one chosen.
    """
    st = maps.get("spacetime")
    if st is None or "recon_year" not in st:
        return None
    r = st["recon_rows"].astype(int)
    c = st["recon_cols"].astype(int)
    y = st["recon_year"].astype(int)
    e = st["recon_err_desk"]
    eras = [(1966, 1979), (1980, 1993), (1994, 2007), (2008, 2025)]
    fig, axs = _panel(geo, ncols=len(eras) + 1, w=3.9)
    vmax = float(np.nanpercentile(e[np.isfinite(e)], 98)) if np.isfinite(e).any() else None
    im = None
    for ax, (lo, hi) in zip(axs[0], eras):
        m = (y >= lo) & (y <= hi)
        g = _geo.to_grid(r[m], c[m], e[m], geo.shape)
        # ONE shared scale AND one shared bar across the era panels: these are the same quantity
        # at different times, so a per-panel scale would hide exactly the drift the panels exist
        # to show, and four identical bars would say otherwise.
        im = _draw(geo, ax, g, f"{lo}–{hi}", cmap="magma", vmax=vmax, cbar=False) or im
    if im is not None:
        # Horizontal, under the four it belongs to. A vertical shared bar steals width from the
        # axes it is attached to and lands on top of the last of them.
        cb = fig.colorbar(im, ax=list(axs[0][:len(eras)]), orientation="horizontal",
                          fraction=0.05, pad=0.03, shrink=0.55, location="bottom")
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label("‖z_desk − z_obs‖  (shared across the four era panels)", fontsize=7)
    nc = st.get("recon_err_nochange")
    last = axs[0][-1]
    if nc is not None:
        better = e < nc
        order = np.argsort(y, kind="stable")
        g = np.full(geo.shape, np.nan)
        for i in order[::-1]:
            if better[i]:
                g[r[i], c[i]] = y[i]
        _draw(geo, last, g, "first year DESK beat the null", cmap="viridis",
              vmin=float(np.nanmin(g)) if np.isfinite(g).any() else None,
              vmax=float(np.nanmax(g)) if np.isfinite(g).any() else None, unit="year",
              fmt="{:.0f}")
        geo.great_plains(last)
    else:
        last.set_axis_off()
    return S.finish(fig, os.path.join(out_dir, "m08_time.png"),
                    "The four era panels share one colour scale on purpose — they are the same "
                    "quantity at different times, and a per-panel scale would hide the drift they "
                    "exist to show. Coverage grows ~8x from the 1960s to the peak, so a cell "
                    "blank in an early panel was not surveyed, not error-free.")


MAPS = (map01_design, map02_ceiling, map03_ladder_winner, map04_vs_bar,
        map05_direction, map06_ecology, map07_depth, map08_time)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = L.load_run(args.run_dir)
    maps = L.load_maps(args.run_dir)
    out = args.out or os.path.join(args.run_dir, "validate_maps")
    os.makedirs(out, exist_ok=True)
    geo = _geo.GeoContext(shape=(133, 224))
    for n in geo.notes:
        print(f"[maps] {n}")
    if maps["missing"]:
        print(f"[maps] absent: {', '.join(maps['missing'])}")

    figs = []
    from validation_report import ACTS, build_html, summary          # noqa: F401
    for fn in MAPS:
        made = fn(run, maps, geo, out)
        if made:
            figs.append((made, (summary(fn), ("Maps", "Where, not how much"))))
            print(f"[maps] {os.path.basename(made)}")
        else:
            print(f"[maps] skip {fn.__name__} (inputs absent)")
    if figs:
        meta = f"{run['label']} — {len(figs)} of {len(MAPS)} maps"
        print(f"[maps] {build_html(out, figs, meta, title='DESK validation maps')}")


if __name__ == "__main__":
    main()
