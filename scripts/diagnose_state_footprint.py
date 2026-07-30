#!/usr/bin/env python3
"""Why is the covariate footprint smaller than the land mask?

``covariate_io.norm_grid`` marks a cell valid only if **every** channel is finite
(``~np.isnan(cov).any(axis=-1)``). With ~295 channels across 5 streams that is
all-or-nothing: a single product whose coverage stops at the Mexican border
invalidates the whole cell, and the DESK cube then *gap-fills* it. Stage-2 of that
fill assigns a year-invariant static field, which is what makes predicted turnover
collapse to exactly 0 over Mexico and the Canadian fringe.

So before masking those cells out of the figures, find out whether they are
recoverable. This reports, per stream:

  valid          cells where that stream alone is fully finite
  uniquely lost  cells that ALL OTHER streams cover but this one does not
                 <- the actionable number: fix this stream, gain these cells

plus the all-stream intersection (what the pipeline actually gets), a per-channel
breakdown inside the worst offender (whole product missing vs one bad band), and
the stage-2 footprint ``land & ~cov_valid & esk_valid`` -- the cells that will show
zero turnover -- rendered to PNG so the geography is visible.

    python scripts/diagnose_state_footprint.py                     # latest year on disk
    python scripts/diagnose_state_footprint.py --years 1980,2025
    python scripts/diagnose_state_footprint.py --no-plot
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK import covariate_io as cio
from src.config_utils import load_config, load_data_config
from src.data.masks import read_land_mask


def _states_dir(cfg):
    hist = cfg["paths"]["hist_dir"]
    cand = os.path.join(hist, "yearly_states")
    return cand if os.path.isdir(cand) else hist


def _available_years(states_dir):
    out = []
    for fp in sorted(glob.glob(os.path.join(states_dir, "state_*.npz"))):
        try:
            out.append(int(os.path.basename(fp).split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    return out


def stream_masks(year, states_dir, schema):
    """``{stream_name: (H,W) bool}`` -- that stream's own all-channels-finite mask."""
    z = np.load(os.path.join(states_dir, f"state_{year}.npz"))
    return {s["name"]: ~np.isnan(z[s["name"]]).any(axis=-1) for s in schema["streams"]}


def report_year(year, states_dir, schema, land, esk_valid=None, plot_path=None):
    masks = stream_masks(year, states_dir, schema)
    names = list(masks)
    n_land = int(land.sum())
    inter = np.ones_like(land)
    for m in masks.values():
        inter &= m
    inter &= land

    print(f"\n=== state_{year}.npz — land cells: {n_land} of {land.size} grid cells ===")
    print(f"{'stream':<11}{'chans':>6}{'valid':>9}{'valid%':>8}{'missing':>9}{'uniq lost':>11}")
    uniq = {}
    for s in schema["streams"]:
        n = s["name"]
        m = masks[n] & land
        # Cells every OTHER stream covers but this one does not: exactly what fixing
        # this stream's coverage would buy back.
        others = np.ones_like(land)
        for o in names:
            if o != n:
                others &= masks[o]
        uniq[n] = int((land & others & ~masks[n]).sum())
        print(f"{n:<11}{int(s['dim']):>6}{int(m.sum()):>9}{100 * m.sum() / max(n_land, 1):>7.1f}%"
              f"{n_land - int(m.sum()):>9}{uniq[n]:>11}")
    print(f"{'ALL (AND)':<11}{'':>6}{int(inter.sum()):>9}"
          f"{100 * inter.sum() / max(n_land, 1):>7.1f}%{n_land - int(inter.sum()):>9}")

    worst = max(uniq, key=uniq.get) if uniq else None
    if worst and uniq[worst] > 0:
        print(f"\n[{worst}] is the binding constraint ({uniq[worst]} cells lost to it alone). "
              f"Per-channel valid counts within it:")
        z = np.load(os.path.join(states_dir, f"state_{year}.npz"))
        arr = z[worst]
        spec = next(s for s in schema["streams"] if s["name"] == worst)
        varnames = list(spec.get("variables") or [])
        counts = [(int((~np.isnan(arr[:, :, k]) & land).sum()), k) for k in range(arr.shape[-1])]
        lo = sorted(counts)[:8]
        for cnt, k in lo:
            label = varnames[k] if k < len(varnames) else f"ch{k}"
            print(f"    {label:<28}{cnt:>8} / {n_land}")
        spread = max(c for c, _ in counts) - min(c for c, _ in counts)
        print(f"    channel spread within stream: {spread} cells -> "
              + ("ONE/few bands are the problem" if spread > 0
                 else "the WHOLE product shares one footprint"))

    if esk_valid is not None:
        # Exactly the cube's stage-2 target: ESK/eBird has a value, covariates do not.
        stage2 = land & (~inter) & esk_valid
        stage3 = land & (~inter) & (~esk_valid)
        print(f"\ncube gap-fill footprint for this year:")
        print(f"    predicted (cov valid)          {int(inter.sum()):>8}")
        print(f"    stage-2 static, YEAR-INVARIANT {int(stage2.sum()):>8}  <- these show 0 turnover")
        print(f"    stage-3 nearest                {int(stage3.sum()):>8}")
        if plot_path:
            _plot(inter, stage2, stage3, land, year, plot_path)
    return inter, uniq


def _plot(predicted, stage2, stage3, land, year, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cat = np.full(land.shape, np.nan, dtype="float32")
    cat[land] = 0.0
    cat[predicted] = 1.0
    cat[stage3] = 2.0
    cat[stage2] = 3.0
    cmap = ListedColormap(["#dddddd", "#08519c", "#fdae61", "#d7301f"])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    im = ax.imshow(cat, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_title(f"Covariate footprint vs cube gap fill — {year}")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["land, no class", "DESK predicted", "stage-3 nearest",
                           "stage-2 static (0 turnover)"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default=None, help="comma-separated (default: latest on disk)")
    ap.add_argument("--out-dir", default=None, help="where to write the footprint PNGs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    dcfg = load_data_config()
    states_dir = _states_dir(cfg)
    schema = cio.load_schema(states_dir)

    res_km = dcfg["grid"]["target_res_m"] // 1000
    mask_path = cfg.get("latent_cube", {}).get("water_mask_path") \
        or os.path.join(dcfg["datasets_root"], "land_mask", f"ocean_mask_{res_km}km.tif")
    land = read_land_mask(mask_path)

    # The ESK reference mask defines where stage-2 can act at all (it needs a static Z).
    z_dir = cfg.get("latent_cube", {}).get("z_dir") or cfg["desk"].get("z_dir") \
        or cfg["paths"]["desk_output_dir"]
    esk_valid = None
    mref = cfg.get("latent_cube", {}).get("mask_ref_path") or os.path.join(z_dir, "valid_mask.npy")
    if os.path.exists(mref):
        esk_valid = np.load(mref)
        if esk_valid.shape != land.shape:
            print(f"[warn] ESK valid_mask {esk_valid.shape} != land {land.shape}; skipping "
                  f"stage-2 breakdown")
            esk_valid = None
    else:
        print(f"[warn] no ESK valid_mask at {mref}; skipping the stage-2 breakdown")

    avail = _available_years(states_dir)
    if not avail:
        raise SystemExit(f"no state_*.npz in {states_dir}")
    years = [int(y) for y in args.years.split(",")] if args.years else [avail[-1]]
    missing = [y for y in years if y not in avail]
    if missing:
        raise SystemExit(f"years not on disk: {missing} (have {avail[0]}..{avail[-1]})")

    out_dir = args.out_dir or os.path.join(cfg["paths"]["desk_output_dir"], "diagnostics")
    for y in years:
        plot = None if args.no_plot else os.path.join(out_dir, f"state_footprint_{y}.png")
        report_year(y, states_dir, schema, land, esk_valid, plot)

    print("\nRead the 'uniq lost' column first: a large value for one stream means that "
          "product's coverage -- not the model -- is what shrinks the cube to gap fill.")


if __name__ == "__main__":
    main()
