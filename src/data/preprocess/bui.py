"""HISDAC-US built-up intensity (BUI) onto the model grid, with an availability channel.

BUI is a far better urbanisation signal than HYDE where it exists -- 250 m indoor
building area back to 1810 -- but it is CONTERMINOUS-US ONLY, while the model grid spans
CONUS + southern Canada + a northern-Mexico strip. So this is an ADDITIVE stream, not a
replacement: the encoder gets BUI where it exists, HYDE everywhere, and an explicit
availability channel telling it which regime each cell is in.

Two properties of the source drive everything here.

**BUI cannot self-report absence.** ``nodata`` is unset and there are no NaNs or
negatives: ocean, Canada, Mexico and genuinely unbuilt CONUS land are all exactly 0.0. So
absence has to be established externally (the CONUS admin polygon) rather than read off
the raster, or every cell outside the US would arrive as a confident "no buildings here".

**It is in its own CRS on its own lattice.** EPSG:5070 (lat_0 = 23) at 250 m, while the
model grid is ESRI:102003 (lat_0 = 37.5) at 27 km with its origin snapped to the BBS
lattice. The deprecated module derived its output transform from the SOURCE profile and
so wrote in BUI's projection at BUI's origin -- rasters the streamer would either fail on
or, worse, align by index. This follows ``preprocess/elevation.py`` instead: reproject
onto a ``fine_factor`` x sub-grid **of the ref transform**, then take exact per-model-cell
quantiles with ``regrid.block_quantiles``. Nesting into the ref lattice is what makes the
aggregation exact and the output alignable.

Outputs (one single-band raster per variable per year, the layout
``PerVariableYearStreamer`` reads -- it reads band 1 only, so the deprecated module's
one-file-7-bands product was unusable):

    bui_q05_{year}_grid.tif ... bui_q99_{year}_grid.tif    within-cell BUI quantiles
    bui_avail_{year}_grid.tif                              fraction of the cell with data
    manifest.json                                          authoritative channel order

Three decisions worth understanding before changing anything:

1. **Absent cells are filled with the in-coverage MEAN, not 0.** This is the only choice
   that puts them at exactly 0 after standardisation -- the value this pipeline already
   treats as "no information" everywhere (``covariate_io.norm_grid`` zero-fills invalid
   cells; the masker masks to 0). Filling with 0 instead would put them at ``-mu/sd``, a
   systematic negative offset that is structured signal, not neutrality. Leaving them NaN
   is worse still: ``norm_grid`` validity is per-CELL and all-or-nothing across every
   concatenated channel, so one NaN would invalidate the cell across all ~300 channels,
   the cube would gap-fill it, and stage 2's year-invariant backfill would make predicted
   turnover exactly 0 over ~40% of the domain -- silently degenerate rather than an error.
   Documented caveat: filling at the mean deflates each channel's ``sd`` by roughly
   sqrt(coverage fraction). Per-channel normalisation fit masks are the follow-on if that
   matters.

2. **The fill is ONE value per channel for ALL years.** BUI grows monotonically, so a
   per-year fill would trend upward and inject spurious *temporal* signal into exactly
   the region that has no data -- predicted turnover in Canada driven by the fill
   trajectory. A constant fill makes the absent region temporally flat, which is what
   "no information" has to look like.

3. **The availability channel rides inside this stream**, declared as
   ``indicator_variable`` so ``augment.ChannelGroupMasker`` masks it atomically with the
   values. A standalone 1-wide stream would cost ~500k parameters (+23%), because the
   stream count enters both the mixer and the decoder, versus ~1k folded in. Note the
   stream ``transform`` (log1p, mandatory here -- raw BUI is square feet running to
   ~1e7-1e8 with a heavy right tail) also hits the availability channel; log1p is
   monotone on [0, 1], so it is an order-preserving rescale of a bounded channel that
   standardisation then absorbs. It changes nothing the encoder can use.

    python -m src.data.preprocess.bui
"""
import glob
import json
import os
import re

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from src.processing import regrid

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.90, 0.99)
# 27000/250 = 108 exactly, so 108 would be a lossless nesting -- but that is a ~2.8 GB
# float64 intermediate per year. 27 (~1 km sub-cells, 729 samples per model cell) is
# ample for a quantile and 16x cheaper.
DEFAULT_FINE_FACTOR = 27
WARMUP_YEARS = 20            # write snapshots this far before first_year, for EMA warm-up
AVAIL_VAR = "bui_avail"
MANIFEST = "manifest.json"
_YEAR_RE = re.compile(r"(\d{4})_BUI\.tif$")


def discover_bui_rasters(bui_dir):
    """``[(year, path)]`` for raw ``NNNN_BUI.tif`` rasters, recursively, sorted by year.

    Handles the depositor's deeply-nested archive layout and skips the macOS ``._``
    AppleDouble junk files that ship inside the tarball.
    """
    hits = []
    for path in glob.glob(os.path.join(bui_dir, "**", "*_BUI.tif"), recursive=True):
        base = os.path.basename(path)
        if base.startswith("._"):
            continue
        m = _YEAR_RE.search(base)
        if m:
            hits.append((int(m.group(1)), path))
    return sorted(hits)


def quantile_names(quantiles=QUANTILES):
    """Channel names for the quantile bands, e.g. 0.05 -> ``bui_q05``, 0.5 -> ``bui_q50``."""
    return [f"bui_q{int(round(q * 100)):02d}" for q in quantiles]


def bui_to_fine_grid(path, ref_transform, ref_crs, H, W, fine_factor):
    """Reproject one BUI snapshot onto a ref-aligned grid ``fine_factor`` x finer.

    Returns ``(fine, block)`` where ``fine`` is ``(H*ff, W*ff)`` in the ref CRS, whose
    blocks nest exactly into model cells. NaN marks *outside the source raster*, which is
    the only absence the source can express -- 0 inside it means "no buildings", not "no
    data", which is why the CONUS polygon is needed as well.

    ``average`` resampling: BUI is an extensive per-pixel area sum, but the sub-cells are
    ~1 km against a 250 m source, so each sub-cell aggregates ~16 source pixels and the
    quantiles that follow are over the sub-cell DISTRIBUTION. An average keeps those
    sub-cell values on the source's own per-pixel scale, so a quantile stays comparable
    across years and cells; a sum would make each sub-cell's magnitude depend on how many
    source pixels happened to land in it.
    """
    ff = int(fine_factor)
    fine_transform = ref_transform * rasterio.Affine.scale(1.0 / ff, 1.0 / ff)
    fine = np.full((H * ff, W * ff), np.nan, dtype="float64")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=fine,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=fine_transform, dst_crs=ref_crs,
            src_nodata=src.nodata, dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return fine, ff


def conus_fine_mask(exclusion_source, ref_crs, shape, fine_transform):
    """Binary fine-grid mask of the conterminous US, from the Natural Earth admin-0 layer.

    Reuses ``land_mask.rasterize_country_exclusions``, which already rasterises that
    shapefile by ISO-A3 -- the same call, read as an inclusion. The shapefile is an
    existing live pipeline input (``coastline.study_exclusion_source``), so nothing new
    has to be acquired.

    Note this is the US *polygon*, which includes Alaska and Hawaii; both fall outside the
    study box, and the intersection with the BUI raster footprint below removes any
    residue in any case.
    """
    from src.data.preprocess.land_mask import rasterize_country_exclusions
    return rasterize_country_exclusions(exclusion_source, ["USA"], ref_crs,
                                        shape, fine_transform) > 0


def cell_quantiles(fine, valid, block, quantiles=QUANTILES):
    """Per model cell, the quantiles of its valid fine sub-cells: ``(Q, H, W)``.

    Sub-cells outside ``valid`` are NaN'd first, so they are excluded from the
    distribution rather than contributing a false 0. A cell with no valid sub-cell at all
    comes back NaN and is filled later.
    """
    masked = np.where(valid, fine, np.nan)
    return regrid.block_quantiles(masked, block, quantiles)


def availability(valid, block):
    """Per model cell, the fraction (0..1) of sub-cells that carry real BUI data.

    Continuous rather than binary because border cells genuinely are partial, and a
    fraction tells the encoder how much of the cell it can trust.
    """
    return regrid.block_reduce(valid.astype("float64"), block, how="mean")


def neutral_fill(stacks, transform=None):
    """Per-channel fill value (in RAW units) that lands at 0 after standardisation.

    ``stacks`` is ``{year: (Q, H, W)}``. Returns ``(Q,)`` fill values: the mean of the
    in-coverage values **in the transformed space**, mapped back through the transform's
    inverse, so that the eventual ``covariate_io._transform`` puts the filled cells at
    exactly the pooled in-coverage mean. Doing the average in transformed space is the
    point -- the mean of log1p is not log1p of the mean, and it is the transformed values
    that get standardised.

    Pooled over ALL years, so the fill is one constant per channel and the absent region
    carries no temporal signal (see the module docstring).
    """
    kind = (transform or {}).get("type")
    if kind == "log1p":
        fwd, inv = np.log1p, np.expm1
    elif kind == "pow":
        p = float(transform["p"])
        fwd, inv = (lambda a: np.power(np.clip(a, 0.0, None), p),
                    lambda a: np.power(np.clip(a, 0.0, None), 1.0 / p))
    elif kind is None:
        fwd = inv = lambda a: a
    else:
        raise ValueError(f"neutral_fill cannot invert transform {transform!r}")

    n_q = next(iter(stacks.values())).shape[0]
    fill = np.empty(n_q, dtype="float64")
    for k in range(n_q):
        vals = np.concatenate([s[k][np.isfinite(s[k])].ravel() for s in stacks.values()])
        if vals.size == 0:
            raise SystemExit(
                "BUI has no in-coverage cells at all on the model grid. Check that the "
                "CONUS polygon and the BUI rasters overlap the study box before trusting "
                "any of this output.")
        fill[k] = inv(float(np.mean(fwd(vals))))
    return fill


def write_grid(path, arr, ref_profile):
    """One single-band float32 model-grid raster, on the ref transform."""
    prof = dict(ref_profile)
    prof.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan,
                height=arr.shape[0], width=arr.shape[1])
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(np.asarray(arr, dtype="float32"), 1)


def write_manifest(out_dir, variables, years, fine_factor, quantiles, fill):
    """Record the authoritative channel order, which is otherwise only ``sorted(glob)``.

    ``build_states.discover_variables`` validates this against what is on disk and
    refuses a mismatch, because channel identity is positional and feeds the saved
    mu/sd -- a stray or missing raster would silently renumber every later channel.
    """
    manifest = {
        "variables": sorted(variables),
        "n_variables": len(variables),
        "indicator_variable": AVAIL_VAR,
        "years": [int(min(years)), int(max(years))],
        "fine_factor": int(fine_factor),
        "quantiles": [float(q) for q in quantiles],
        "raw_fill_value": {n: float(v) for n, v in zip(quantile_names(quantiles), fill)},
        "_fill_comment": (
            "raw_fill_value is the per-channel constant written outside the CONUS "
            "footprint, chosen so that after the stream's log1p transform the filled "
            "cells sit at the pooled in-coverage mean and therefore at 0 after "
            "standardisation. Constant across years on purpose: BUI grows monotonically, "
            "so a per-year fill would inject a spurious trend into the no-data region."),
    }
    with open(os.path.join(out_dir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def main():
    import argparse

    from src.config_utils import load_data_config
    from src.temporal import load_timeline

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bui-dir", help="Raw BUI dir (default: {datasets_root}/{dataverse.out_subdirs.bui}).")
    ap.add_argument("--out-dir", help="Default: {datasets_root}/bui_grid.")
    ap.add_argument("--fine-factor", type=int)
    ap.add_argument("--exclusion-source", help="Natural Earth admin-0 polygons (default: coastline.study_exclusion_source).")
    args = ap.parse_args()

    cfg = load_data_config()
    dr = cfg["datasets_root"]
    bcfg = cfg.get("bui", {})
    tl = load_timeline(cfg)
    year_lo, year_hi = tl["first_year"] - WARMUP_YEARS, tl["end_year"]

    bui_dir = args.bui_dir or os.path.join(
        dr, cfg.get("dataverse", {}).get("out_subdirs", {}).get("bui", "HBUI"))
    out_dir = args.out_dir or os.path.join(dr, bcfg.get("out_subdir", "bui_grid"))
    fine_factor = args.fine_factor or int(bcfg.get("fine_factor", DEFAULT_FINE_FACTOR))
    excl = args.exclusion_source or cfg.get("coastline", {}).get("study_exclusion_source")
    if not excl:
        raise SystemExit("no admin-0 polygon source (set coastline.study_exclusion_source)")
    if not os.path.isabs(excl):
        excl = os.path.join(dr, excl)
    if not os.path.exists(excl):
        raise SystemExit(
            f"admin-0 polygons not found: {excl}. BUI absence cannot be read off the "
            f"rasters (0 means 'no buildings', not 'no data'), so the CONUS polygon is "
            f"required, not optional. Run scripts/tacc/download_all.sh.")

    rasters = [(y, p) for y, p in discover_bui_rasters(bui_dir) if year_lo <= y <= year_hi]
    if not rasters:
        raise SystemExit(f"No *_BUI.tif rasters in {year_lo}..{year_hi} under {bui_dir}")

    with rasterio.open(cfg["grid"]["ref_raster"]) as ref:
        ref_transform, ref_crs, H, W = ref.transform, ref.crs, ref.height, ref.width
        ref_profile = ref.profile

    ff = int(fine_factor)
    fine_transform = ref_transform * rasterio.Affine.scale(1.0 / ff, 1.0 / ff)
    conus = conus_fine_mask(excl, ref_crs, (H * ff, W * ff), fine_transform)
    print(f"[bui] {len(rasters)} snapshots {rasters[0][0]}..{rasters[-1][0]} "
          f"(timeline {year_lo}..{year_hi}); fine_factor={ff} "
          f"({H * ff}x{W * ff} sub-cells); CONUS covers {conus.mean():.1%} of the box",
          flush=True)

    # Pass 1: quantiles + availability per year. The fine grid is transient (~174 MB at
    # ff=27); only the (Q,H,W) summaries are retained, which is a few MB in total.
    stacks, avail = {}, None
    for year, path in rasters:
        fine, block = bui_to_fine_grid(path, ref_transform, ref_crs, H, W, ff)
        valid = np.isfinite(fine) & conus
        stacks[year] = cell_quantiles(fine, valid, block)
        if avail is None:
            avail = availability(valid, block)
        n_cov = int(np.isfinite(stacks[year][0]).sum())
        print(f"[bui]   {year}: {n_cov}/{H * W} cells with data "
              f"({n_cov / (H * W):.1%})", flush=True)
        del fine, valid

    # Availability is written once and reused for every year: the CONUS footprint is
    # static, and the BUI raster extent does not move between snapshots. If that ever
    # changes, the per-year covered-cell counts printed above will disagree.
    covered = [int(np.isfinite(s[0]).sum()) for s in stacks.values()]
    if min(covered) != max(covered):
        print(f"[bui] WARNING coverage varies across snapshots ({min(covered)}..{max(covered)} "
              f"cells); the single availability raster describes the first year only.",
              flush=True)

    transform = bcfg.get("transform", {"type": "log1p"})
    fill = neutral_fill(stacks, transform)
    names = quantile_names()
    print(f"[bui] raw fill outside coverage (constant across years, lands at the pooled "
          f"in-coverage mean after {transform.get('type')}):", flush=True)
    for n, v in zip(names, fill):
        print(f"[bui]   {n:<10} {v:,.1f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    for year, stack in stacks.items():
        for k, name in enumerate(names):
            band = np.where(np.isfinite(stack[k]), stack[k], fill[k])
            write_grid(os.path.join(out_dir, f"{name}_{year}_grid.tif"), band, ref_profile)
        write_grid(os.path.join(out_dir, f"{AVAIL_VAR}_{year}_grid.tif"), avail, ref_profile)

    variables = names + [AVAIL_VAR]
    write_manifest(out_dir, variables, list(stacks), ff, QUANTILES, fill)
    print(f"[bui] wrote {len(stacks)} years x {len(variables)} variables -> {out_dir}",
          flush=True)
    print(f"[bui] availability: mean {avail.mean():.3f}, "
          f"{int((avail > 0).sum())}/{H * W} cells with any coverage", flush=True)


if __name__ == "__main__":
    main()
