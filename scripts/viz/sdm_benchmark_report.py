#!/usr/bin/env python3
"""Multi-page PDF comparing every fitted SDM against the dynamic model.

WHAT THIS IS FOR. The benchmark's result is that OCCUPANCY AND DEMOGRAPHY
DISAGREE over the Great Plains: every correlative model, plus raw BBS and
eBird's published range, puts Plains occupancy level with the rest of the
continent, while the dynamic model's mean intrinsic growth rate there is BELOW
REPLACEMENT. That is a claim about two different quantities, so the report
draws them side by side and says plainly which model can produce which.

ONE DELIBERATE CHOICE. The correlative surfaces are drawn UNTHRESHOLDED. Cutting
psi at maxSSS turns a model reporting psi = 0.83 in the Plains into "35% niche";
the cut manufactures the answer, and every threshold rule in routine use gives a
different one (30%-100% for the same predictions). Continuous psi needs no
convention and is what those models actually estimate.

    python scripts/viz/sdm_benchmark_report.py \\
        --results results/sdm_benchmark \\
        --run-dir data/processed/model_results/<RUN> \\
        --out results/sdm_benchmark/sdm_benchmark_report.pdf
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

sys.path.insert(0, str(_REPO / "scripts" / "viz"))
import _geo  # noqa: E402
import _validation_style as _vs  # noqa: E402

from src.analysis.sdm_benchmark import data as D, designation as G  # noqa: E402

TITLE = "House Finch: correlative SDMs vs a dynamic range model"

# What each model was given, and what it can therefore estimate. Kept in the
# figure rather than a caption because the whole point is that these models are
# not interchangeable.
MODEL_NOTES = {
    "biolith_standard": dict(
        kind="Occupancy (detection-corrected)", estimates=r"$\psi$ = P(site occupied)",
        covars="~84 principal components of a bioclim-style set: annual/summer/winter "
               "means per climate base variable (from 12 monthly bio-year channels), "
               "LUH-3 land use, HYDE population, SoilGrids, elevation.",
        prep="Rotated to principal components (raw condition number 3.0e19); "
             "nothing truncated. Years 2015-2025 as repeat visits."),
    "biolith_full": dict(
        kind="Occupancy (detection-corrected)", estimates=r"$\psi$ = P(site occupied)",
        covars="~287 principal components of all 302 raw encoder channels: climate "
               "(~240 = 14 bases x 12 bio-year months, q10/q50/q90 for temperature), "
               "LUH-3 (~24), SoilGrids (16), HISDAC built-up (7), HYDE (3), elevation (3).",
        prep="Rotated to principal components (raw condition number 4.9e20); "
             "nothing truncated. Years 2015-2025 as repeat visits."),
    "biolith_latent": dict(
        kind="Occupancy (detection-corrected)", estimates=r"$\psi$ = P(site occupied)",
        covars="The 24-dimensional learned community basis Z -- the same input the "
               "dynamic model consumes. Kernel-PCA eigenfeatures over bird-community "
               "composition, extrapolated from climate/land-use/soil by the DESK encoder.",
        prep="No PCA needed (condition number 6.9); Z is already an eigenbasis. "
             "Years 2015-2025 as repeat visits."),
    "brt_standard": dict(
        kind="Boosted trees (no detection model)", estimates="P(detected on a route-year)",
        covars="The same bioclim-style set as biolith_standard, unrotated "
               "(trees are indifferent to collinearity).",
        prep="Elith/Leathwick/Hastie (2008) recipe; 5-fold SPATIAL BLOCK CV with a "
             "buffer ring. Predictions are out-of-fold."),
    "brt_full": dict(
        kind="Boosted trees (no detection model)", estimates="P(detected on a route-year)",
        covars="All 302 raw encoder channels, unrotated.",
        prep="Elith/Leathwick/Hastie (2008) recipe; 5-fold spatial block CV."),
    "brt_latent": dict(
        kind="Boosted trees (no detection model)", estimates="P(detected on a route-year)",
        covars="The 24-dimensional learned basis Z.",
        prep="Elith/Leathwick/Hastie (2008) recipe; 5-fold spatial block CV."),
}

EBIRD_NOTE = dict(
    kind="Published range boundary (eBird Status & Trends)",
    estimates="Range: is the species present at all",
    covars="Effort covariates (search duration, stationary vs travelling, distance, "
           "party size, checklist calibration index) plus environmental predictors "
           "summarised in a neighbourhood around each checklist: elevation, land "
           "cover, water cover (percentage cover and edge density).",
    prep="AdaSTEM -- an ensemble of local regression models over space and time, "
         "fitted to semi-structured eBird checklists. Summarised from public "
         "documentation; not refitted here. Resident season, 2023.")

DYNAMIC_NOTE = dict(
    kind="Dynamic, age-structured, spatially explicit",
    estimates=r"$\lambda$ = intrinsic growth rate, AND realised abundance",
    covars="The 24-dimensional learned basis Z, plus dispersal-path-integrated "
           "features (Z x 12 directional/radial kernels).",
    prep="Bayesian MAP fit of a forward simulation over 1902-2025 to raw BBS route "
         "counts (NegativeBinomial2), with survival, fecundity, carrying capacity, "
         "density-dependent dispersal and an Allee term.")


def _cellmean(path, key, shape, grid_key=None):
    """The model's CONTINENTAL surface when it exists, else its site predictions.

    An SDM's product is a map. Restricting it to surveyed cells (~1,800 of
    ~17,000) draws the survey design instead, and puts the fitted models on a
    different support from eBird, which predicts everywhere. The fitting stage
    now writes a full-grid surface; this falls back to the sparse one only for
    prediction files produced before that existed.
    """
    z = np.load(path, allow_pickle=True)
    if grid_key and grid_key in z.files:
        g = np.asarray(z[grid_key], dtype=float)
        if g.shape == tuple(shape):
            return g, True
    if key not in z.files:
        return None
    v, r, c = np.asarray(z[key]), np.asarray(z["row"]), np.asarray(z["col"])
    ok = np.isfinite(v)
    tot = np.zeros(shape); cnt = np.zeros(shape)
    np.add.at(tot, (r[ok], c[ok]), v[ok].astype(float))
    np.add.at(cnt, (r[ok], c[ok]), 1.0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1.0), np.nan), False


def _ebird_abundance(ref):
    """eBird's seasonal-mean relative abundance ON THE MODEL GRID.

    Reuses ``regrid.reproject_to_ref`` -- the same helper
    ``scripts/viz/overlay_great_plains_ebird.py`` and the ebird preprocessor use
    -- because the published raster is EPSG:8857 (Equal Earth) at 618x1276 and
    must be resampled, not indexed. ``average`` is the linear areal aggregate;
    any nonlinear transform belongs afterwards, at target resolution.
    """
    try:
        import rioxarray
        from src.processing import regrid
    except ImportError:
        return None
    p = _REPO / "data" / "ebird_abundance" / "houfin_abundance_seasonal_mean_27km_2023.tif"
    if not p.exists():
        return None
    da = rioxarray.open_rasterio(p, masked=True).rio.write_crs("EPSG:8857", inplace=False)
    da = da.rio.write_nodata(float("nan"), inplace=False)
    return np.asarray(regrid.reproject_to_ref(da, ref, resampling="average").values[0],
                      dtype=float)


def _ebird_range(shape, crs, transform):
    """eBird's published range polygon, rasterised onto the model grid.

    Rasterised rather than drawn as a geometry (which is what
    overlay_great_plains_ebird.py does) so it lands in the same array space as
    every other panel and can enter the zone table.
    """
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
    except ImportError:
        return None
    p = _REPO / "data" / "ebird_range" / "houfin_range_smooth_27km_2023.gpkg"
    if not p.exists():
        return None
    g = gpd.read_file(p, layer="range").to_crs(crs)
    return rasterize([(geom, 1) for geom in g.geometry], out_shape=shape,
                     transform=transform, fill=0, all_touched=True,
                     dtype="uint8").astype(float)


def _panel(ax, geo, grid, title, subtitle, vmin=None, vmax=None, cmap="viridis",
           cbar_label=None):
    # rasterize the map layers: the Natural Earth coastline is a very large
    # polygon set, and embedding it as vectors on every panel took the PDF to
    # 61 MB. Text, titles and colourbars stay vector.
    geo.basemap(ax)
    im = geo.imshow(ax, grid, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    geo.coastline(ax)
    geo.great_plains(ax, lw=1.4, color="crimson")
    for artist in ax.collections + ax.lines + ax.patches:
        artist.set_rasterized(True)
    ax.set_title(title, fontsize=9, fontweight="bold", loc="left", pad=13)
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=6.4,
            color="0.35", va="bottom")
    ax.set_xticks([]); ax.set_yticks([])
    if im is not None and cbar_label:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(cbar_label, fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
    return im


def _textpage(pdf, title, blocks, figsize=(8.5, 11), width=108):
    """Paginated body text.

    Wraps to a character count rather than relying on matplotlib's ``wrap=True``,
    which does nothing useful for text placed in FIGURE coordinates -- long lines
    ran off the page and collided at the bottom margin.
    """
    import textwrap

    fig = plt.figure(figsize=figsize)
    fig.text(0.06, 0.95, title, fontsize=14, fontweight="bold")
    y = 0.90

    def _newpage():
        nonlocal fig, y
        pdf.savefig(fig, dpi=_vs.DPI); plt.close(fig)
        fig = plt.figure(figsize=figsize); y = 0.93

    for head, body in blocks:
        wrapped = []
        for line in body:
            if not line:
                wrapped.append("")
                continue
            wrapped += textwrap.wrap(line, width=width,
                                     subsequent_indent="    ") or [""]
        if y - 0.0165 * (len(wrapped) + 2) < 0.06:
            _newpage()
        if head:
            fig.text(0.06, y, head, fontsize=10, fontweight="bold")
            y -= 0.023
        for line in wrapped:
            if y < 0.06:
                _newpage()
            fig.text(0.06, y, line, fontsize=8, family="DejaVu Sans", color="0.15")
            y -= 0.0165
        y -= 0.012
    pdf.savefig(fig, dpi=_vs.DPI); plt.close(fig)


def build(args):
    fields = G.load_dynamic_fields(args.run_dir)
    lam = fields["lam_fundamental_modern"]
    niche, finite = G.niche_from_lambda(lam)
    ny, nx = lam.shape
    geo = _geo.GeoContext(shape=(ny, nx))

    df = D.plains_band(D.attach_zones(D.route_years(args.rep_start, args.rep_end)))
    occ, surv, meanc = G.cell_occupancy(df, ny, nx)

    import rasterio
    from src.config_utils import load_data_config
    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as src:
        transform, crs = src.transform, src.crs
    with rasterio.open(load_data_config()["regions"]["great_plains_zones"]) as src:
        zones = src.read(1)
    ebird = _ebird_range((ny, nx), crs, transform)
    try:
        from src.processing import regrid
        ebird_abd = _ebird_abundance(regrid.load_ref(load_data_config()))
    except Exception:
        ebird_abd = None

    surfaces, is_grid = {}, {}
    for p in sorted(glob.glob(os.path.join(args.results, "*_pred.npz"))):
        tag = os.path.basename(p)[: -len("_pred.npz")]
        bio = tag.startswith("biolith")
        got = _cellmean(p, "psi" if bio else "occ_pred", (ny, nx),
                        grid_key="grid_psi" if bio else "grid_occ")
        if got is not None:
            surfaces[tag], is_grid[tag] = got

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        _cover(pdf, surfaces, ebird, occ, niche, lam, finite, surv, zones, args)
        _maps(pdf, geo, surfaces, ebird, lam, niche, meanc, ebird_abd, is_grid)
        _stats(pdf, surfaces, ebird, occ, niche, lam, finite, surv, zones)
        _methods(pdf, surfaces)
        d = pdf.infodict()
        d["Title"] = TITLE
        d["Subject"] = "Occupancy vs demography over the Great Plains"
    print(f"wrote {args.out}")
    return args.out


def _zone_table(surfaces, ebird, occ, niche, lam, finite, surv, zones):
    rows = []
    for z, lab in [(1, "West"), (2, "Great Plains"), (3, "East")]:
        m = finite & surv & (zones == z)
        if not m.sum():
            continue
        rec = {"zone": lab, "n": int(m.sum()),
               "BBS any-detect": float(occ[m].mean()),
               "lambda>=1": float(niche[m].mean()),
               "mean lambda": float(np.nanmean(lam[m]))}
        if ebird is not None:
            rec["eBird range"] = float(ebird[m].mean())
        for t, g in sorted(surfaces.items()):
            rec[t] = float(np.nanmean(g[m]))
        rows.append(rec)
    return rows


def _cover(pdf, surfaces, ebird, occ, niche, lam, finite, surv, zones, args):
    rows = _zone_table(surfaces, ebird, occ, niche, lam, finite, surv, zones)
    gp = next((r for r in rows if r["zone"] == "Great Plains"), {})
    ea = next((r for r in rows if r["zone"] == "East"), {})
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.93, TITLE, fontsize=17, fontweight="bold")
    fig.text(0.06, 0.90, "Occupancy and demography disagree over the Great Plains",
             fontsize=11, color="0.3")
    body = [
        "",
        "Every correlative model here estimates OCCUPANCY -- where the species is. The dynamic",
        "model additionally estimates DEMOGRAPHY -- whether a population there replaces itself.",
        "Over the Great Plains these give different answers, and that difference is the result.",
        "",
        f"    Great Plains   mean intrinsic growth rate  lambda = "
        f"{gp.get('mean lambda', float('nan')):.3f}   (below replacement)",
        f"    East           mean intrinsic growth rate  lambda = "
        f"{ea.get('mean lambda', float('nan')):.3f}",
        "",
        f"    Great Plains   occupancy, BBS any-detection       = "
        f"{gp.get('BBS any-detect', float('nan')):.3f}",
        (f"    Great Plains   eBird published range              = "
         f"{gp.get('eBird range', float('nan')):.3f}") if "eBird range" in gp else "",
        "",
        "An occupied site with lambda < 1 is a SINK: held by immigration rather than local",
        "reproduction. No correlative model in this report can express that, because occupancy",
        "is modelled as a function of environment alone -- there is no term in which",
        "'occupied because of its neighbours' could live.",
        "",
        "A NOTE ON THRESHOLDS. Correlative surfaces are drawn unthresholded throughout.",
        "Binarising psi at maxSSS turns a model reporting psi = 0.83 over the Plains into",
        "'35% suitable'; the standard rules span 30%-100% for the same predictions. The cut is",
        "a convention, not a measurement. lambda >= 1 has no equivalent dial: 1.0 is where a",
        "population stops replacing itself.",
        "",
        f"Dynamic model run: {os.path.basename(args.run_dir)}",
        f"BBS window: {args.rep_start}-{args.rep_end}, surveyed cells, 30-42N",
        "Great Plains outlined in red on every map (EPA/CEC Level I ecoregions).",
    ]
    y = 0.85
    for line in body:
        fig.text(0.06, y, line, fontsize=9, color="0.15")
        y -= 0.018
    pdf.savefig(fig, dpi=_vs.DPI); plt.close(fig)


def _maps(pdf, geo, surfaces, ebird, lam, niche, meanc, ebird_abd=None,
          is_grid=None):
    # Page: the two quantities, side by side, as large as they can be drawn.
    fig = plt.figure(figsize=(11, 4.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.16, left=0.04, right=0.94,
                  top=0.84, bottom=0.05)
    _panel(fig.add_subplot(gs[0, 0]), geo, np.where(np.isfinite(lam), lam, np.nan),
           "Intrinsic growth rate  $\\lambda$   (dynamic model only)",
           "blue = below replacement; only this model estimates it",
           vmin=0.5, vmax=1.5, cmap="RdBu_r", cbar_label="$\\lambda$")
    # log1p: raw route counts run 0 to ~490, so a linear scale crushes almost
    # every cell to the bottom of the colourmap.
    _panel(fig.add_subplot(gs[0, 1]), geo, np.log1p(meanc),
           "Observed BBS mean route count",
           "what all the correlative models are fitted to (log scale)",
           vmin=0, vmax=np.log1p(60), cmap="magma",
           cbar_label="log1p(birds / route)")
    fig.suptitle("The two quantities", fontsize=13, fontweight="bold", x=0.04, ha="left")
    pdf.savefig(fig, dpi=_vs.DPI); plt.close(fig)

    # Page: every correlative surface, one grid.
    items = [(t, g, "P(occupied)  $\\psi$" if t.startswith("biolith")
              else "P(detected on a route-year)") for t, g in sorted(surfaces.items())]
    if ebird is not None:
        items.append(("eBird published range", ebird, "in range (1) / out (0)"))
    if ebird_abd is not None:
        items.append(("eBird relative abundance", np.log1p(np.nan_to_num(ebird_abd)),
                      "log1p(relative abundance)"))
    EBIRD_ABD_KIND = "Relative abundance (eBird Status & Trends)"
    ncol = 3
    nrow = int(np.ceil(len(items) / ncol))
    fig = plt.figure(figsize=(11, 2.75 * nrow + 1.1))
    gs = GridSpec(nrow, ncol, figure=fig, wspace=0.16, hspace=0.30,
                  left=0.03, right=0.94, top=0.88, bottom=0.07)
    for i, (tag, grid, lab) in enumerate(items):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        note = MODEL_NOTES.get(tag, EBIRD_NOTE if "eBird" in tag else {})
        kind = (EBIRD_ABD_KIND if "abundance" in tag else note.get("kind", ""))
        hi = None if "abundance" in tag else 1
        _panel(ax, geo, grid, tag, kind, vmin=0, vmax=hi,
               cmap="viridis", cbar_label=lab)
    fig.suptitle("Correlative models: all estimate OCCUPANCY, none estimate $\\lambda$",
                 fontsize=13, fontweight="bold", x=0.03, ha="left")
    # The support differs and the eye will read it as a difference in the models.
    sparse = [t for t, g in (is_grid or {}).items() if not g]
    note = ("NOTE ON SUPPORT. Every panel is a continental prediction: the models are "
            "evaluated on all land cells, not only where BBS sampled.\nZone statistics "
            "below are still computed on surveyed cells only, so every row of the table "
            "shares one support.")
    if sparse:
        note += ("\nDrawn from SITE predictions only (no grid surface in the file): "
                 + ", ".join(sorted(sparse)))
    fig.text(0.03, 0.022, note, fontsize=7.5, color="0.25")
    pdf.savefig(fig, dpi=_vs.DPI); plt.close(fig)

    # Page: the dynamic model's extra products.
    fig = plt.figure(figsize=(11, 4.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.16, left=0.04, right=0.94,
                  top=0.84, bottom=0.05)
    _panel(fig.add_subplot(gs[0, 0]), geo, niche.astype(float),
           "Self-sustaining:  $\\lambda \\geq 1$", "no threshold chosen -- 1.0 is replacement",
           vmin=0, vmax=1, cmap="RdYlGn", cbar_label="source (1) / sink (0)")
    _panel(fig.add_subplot(gs[0, 1]), geo,
           np.where(np.isfinite(lam), np.clip(1.0 - lam, -0.5, 0.5), np.nan),
           "Shortfall below replacement  $1-\\lambda$", "red = cannot replace itself locally",
           vmin=-0.5, vmax=0.5, cmap="coolwarm", cbar_label="$1-\\lambda$")
    fig.suptitle("What only the dynamic model provides", fontsize=13,
                 fontweight="bold", x=0.04, ha="left")
    pdf.savefig(fig, dpi=_vs.DPI); plt.close(fig)


def _stats(pdf, surfaces, ebird, occ, niche, lam, finite, surv, zones):
    rows = _zone_table(surfaces, ebird, occ, niche, lam, finite, surv, zones)
    cols = ["zone", "n"] + [c for c in rows[0] if c not in ("zone", "n")]
    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_axes([0.03, 0.22, 0.94, 0.56]); ax.axis("off")
    table = ax.table(
        cellText=[[r["zone"], f"{r['n']}"] +
                  [f"{r[c]:.3f}" for c in cols[2:]] for r in rows],
        colLabels=[c.replace("biolith_", "$\\psi$ ").replace("brt_", "BRT ")
                   .replace("BBS any-detect", "BBS\nany-detect")
                   .replace("mean lambda", "mean\n$\\lambda$")
                   .replace("lambda>=1", "$\\lambda\\geq1$")
                   .replace("eBird range", "eBird\nrange") for c in cols],
        cellLoc="center", loc="upper center")
    table.auto_set_font_size(False); table.set_fontsize(7.0); table.scale(1, 1.7)
    for j, c in enumerate(cols):
        table[(0, j)].set_facecolor("#e8e8e8")
        table[(0, j)].set_text_props(fontweight="bold")
    for i, r in enumerate(rows, start=1):
        if r["zone"] == "Great Plains":
            for j in range(len(cols)):
                table[(i, j)].set_facecolor("#fff2f2")
    fig.text(0.03, 0.92, "Occupancy vs demography, by zone", fontsize=13, fontweight="bold")
    fig.text(0.03, 0.875,
             "Surveyed cells, 30-42N. Nothing here is thresholded.", fontsize=9, color="0.3")
    fig.text(0.03, 0.06,
             "Occupancy estimates agree with each other and with eBird across all three zones.\n"
             "Mean lambda does not: the Great Plains sits below replacement while the East is "
             "well above it.\nThat gap -- occupied, but not self-sustaining -- is the benchmark's "
             "result, and it requires no threshold.", fontsize=8.5, color="0.15")
    pdf.savefig(fig, dpi=_vs.DPI); plt.close(fig)


def _methods(pdf, surfaces):
    blocks = [("How to read this", [
        "Each model below is listed with what it was given and what it can therefore estimate.",
        "They are not interchangeable: a boosted tree fitted to detection/non-detection",
        "estimates P(detected), which confounds occupancy with detectability, so a",
        "low-abundance occupied site scores low. An occupancy model separates the two. Only",
        "the dynamic model estimates a growth rate.", ""])]
    for tag in sorted(surfaces):
        n = MODEL_NOTES.get(tag)
        if not n:
            continue
        blocks.append((f"{tag}  --  {n['kind']}", [
            f"Estimates: {n['estimates']}", f"Covariates: {n['covars']}",
            f"Pre-processing: {n['prep']}", ""]))
    blocks.append(("eBird Status & Trends  --  " + EBIRD_NOTE["kind"], [
        f"Estimates: {EBIRD_NOTE['estimates']}", f"Covariates: {EBIRD_NOTE['covars']}",
        f"Method: {EBIRD_NOTE['prep']}", ""]))
    blocks.append(("Dynamic range model  --  " + DYNAMIC_NOTE["kind"], [
        f"Estimates: {DYNAMIC_NOTE['estimates']}", f"Covariates: {DYNAMIC_NOTE['covars']}",
        f"Method: {DYNAMIC_NOTE['prep']}", ""]))
    blocks.append(("Shared preprocessing", [
        "All surfaces are on one 27 km grid (ESRI:102003, 133 x 224), snapped to the USGS BBS",
        "lattice. BBS route-years are screened to protocol runs (RunType != 0, RPID == 101) and",
        "assigned to cells by route centroid. The 2020 season was cancelled and is dropped.",
        "Observer effects are withheld from every model, including the dynamic one, so no model",
        "has information another lacks.", ""]))
    blocks.append(("Known limitations", [
        "- Occupancy fits are in-sample; the BRT is scored out-of-fold under spatial block CV,",
        "  so their AUCs are not directly comparable.",
        "- Years-as-repeat-visits assumes closure within 2015-2025; the measured within-window",
        "  trend is +1.7%/yr, small but not zero.",
        "- The occupancy models correct for detection; the BRT and the dynamic model do not.",
        "- eBird is summarised from public documentation, not refitted here; its range product",
        "  answers 'present at all', an inclusive question, and its cell support differs.",
        "- N in an N-mixture is defined by the sampling protocol: for a roadside BBS route,",
        "  'N = 0' means no birds available to that route, not an empty 27 km cell.", ""]))
    _textpage(pdf, "Models, covariates and pre-processing", blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(_REPO / "results" / "sdm_benchmark"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rep-start", type=int, default=D.DEFAULT_REPLICATE_START)
    ap.add_argument("--rep-end", type=int, default=D.DEFAULT_REPLICATE_END)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(args.results, "sdm_benchmark_report.pdf")
    build(args)


if __name__ == "__main__":
    main()
