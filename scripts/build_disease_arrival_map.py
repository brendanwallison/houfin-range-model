"""Build a smoothed per-pixel *year of arrival* map for House Finch mycoplasmal
conjunctivitis, on the 27 km model grid.

Why this exists
---------------
The age-structured model has no covariate for the 1994- epizootic. Its effect on
carrying capacity is modeled structurally -- severity(x) x onset_gate(x,t) x
(1 - recovery) -- and the onset gate needs an exogenous answer to "when, if ever,
did the disease reach this pixel?", which is what this script produces. Supplying
that timing from outside the fit is what keeps the term a disease term: without it
the model would have to discover the wavefront from abundance data alone, and a
free spatiotemporal field asked to do that instead annihilated eastern carrying
capacity (see src/model/age_fields.py).

Where the numbers come from
---------------------------
There is no published continental arrival-year raster, so the surface is built by
kernel-smoothing a set of hand-placed anchor points whose years encode the
documented spread history (House Finch Disease Survey / Project FeederWatch
reporting, Dhondt et al., Hochachka & Dhondt, Ley et al.):

  1994  mid-Atlantic ground zero (DC / MD / VA; winter 1993-94)
  1995  PA, DE, NJ, NY, CT
  1996  WV, OH, KY, TN, NC, SC
  1997  MI, IN, IL, WI, New England, GA, AL, MS; southern Ontario
  1998  eastern MN/IA/MO/AR/LA (the Plains border)
  1999-2002  slow traverse of the Great Plains (ND, SD, NE, KS, OK, TX)
  2003  first western detection, western Montana (Missoula)
  2004  ID, WA, OR
  2005  WY, UT, CO, NV, NM, AZ
  2006  CA
  2007  northern Mexico (poorly documented; nominal)

Anchor years are the first *breeding season* (≈ June, the BBS census point)
expected to be affected, i.e. the narrative detection year advanced by the
following spring where the report is a fall/winter one. Split states (MN, IA, MO,
AR, LA, MT, TX) get separate eastern and western anchors because the front
crossed them over several years.

Method
------
Nadaraya-Watson (Gaussian kernel) regression of anchor year on Albers x/y with a
``--bandwidth-km`` (default 300 km) kernel, evaluated on every grid cell. This is
deliberately simple: it yields a smooth, monotone-in-distance-from-source surface
with no ridges or interpolation artifacts, and it degrades gracefully far from any
anchor (toward the anchor-weighted mean). It does NOT model the Rocky Mountain
barrier as a discontinuity -- the ~1998 plains and ~2003 Montana anchors let the
smoother produce a steep but continuous gradient there instead. That is the
intended behavior: the model's onset gate has a learned lag/steepness on top of
this surface, so a soft front is more honest than a fake hard one.

Outputs (under ``<processed>/disease/``)
  disease_arrival_year_27km.tif   float32, model grid, arrival year per pixel
  disease_arrival_year_27km.png   preview
  disease_arrival_anchors.csv     the anchor table actually used

Run: uv run python scripts/build_disease_arrival_map.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import rasterio
from rasterio.warp import transform as project_coords

from src.config_utils import load_age_model_config, load_data_config

# (label, lat, lon, first affected breeding season)
ANCHORS = [
    # --- Phase 1: ground zero, winter 1993-94 ---
    ("Washington DC",        38.90, -77.04, 1994),
    ("MD Baltimore",         39.29, -76.61, 1994),
    ("MD Eastern Shore",     38.77, -75.99, 1994),
    ("VA Richmond",          37.54, -77.44, 1994),
    ("VA Shenandoah",        38.45, -78.87, 1994),
    # --- Phase 2a: mid-Atlantic / Northeast, late 1994 ---
    ("DE Dover",             39.16, -75.52, 1995),
    ("NJ Trenton",           40.22, -74.76, 1995),
    ("PA Philadelphia",      39.95, -75.17, 1995),
    ("PA Pittsburgh",        40.44, -79.996, 1995),
    ("NY New York City",     40.71, -74.01, 1995),
    ("NY Albany",            42.65, -73.76, 1995),
    ("CT Hartford",          41.76, -72.68, 1995),
    # --- Phase 2b: Southeast / Ohio Valley, 1995 ---
    ("WV Charleston",        38.35, -81.63, 1996),
    ("OH Columbus",          39.96, -83.00, 1996),
    ("KY Lexington",         38.04, -84.50, 1996),
    ("TN Nashville",         36.16, -86.78, 1996),
    ("NC Raleigh",           35.78, -78.64, 1996),
    ("SC Columbia",          34.00, -81.03, 1996),
    ("NY Buffalo",           42.89, -78.88, 1996),
    ("MA Boston",            42.36, -71.06, 1996),
    # --- Phase 3: Midwest / Deep South saturation, 1996-97 ---
    ("MI Detroit",           42.33, -83.05, 1997),
    ("MI Traverse City",     44.76, -85.62, 1997),
    ("IN Indianapolis",      39.77, -86.16, 1997),
    ("IL Chicago",           41.88, -87.63, 1997),
    ("IL Springfield",       39.80, -89.64, 1997),
    ("WI Madison",           43.07, -89.40, 1997),
    ("GA Atlanta",           33.75, -84.39, 1997),
    ("AL Birmingham",        33.52, -86.81, 1997),
    ("MS Jackson",           32.30, -90.18, 1997),
    ("VT/NH White Mtns",     44.10, -71.80, 1997),
    ("ME Portland",          43.66, -70.26, 1997),
    ("ON Toronto",           43.65, -79.38, 1997),
    # --- Phase 3 border: eastern edge of the Plains, 1997 ---
    ("MN Minneapolis",       44.98, -93.27, 1998),
    ("IA Des Moines",        41.59, -93.62, 1998),
    ("MO St Louis",          38.63, -90.20, 1998),
    ("AR Little Rock",       34.75, -92.29, 1998),
    ("LA New Orleans",       29.95, -90.07, 1998),
    ("QC Montreal",          45.50, -73.57, 1998),
    ("NS Halifax",           44.65, -63.58, 1999),
    # --- Phase 4: slow traverse of the Great Plains, 1998-2001 ---
    ("MO Kansas City",       39.10, -94.58, 1999),
    ("MN Fargo-Moorhead",    46.87, -96.79, 2000),
    ("NE Omaha",             41.26, -95.93, 2000),
    ("OK Tulsa",             36.15, -95.99, 2000),
    ("TX Dallas",            32.78, -96.80, 2000),
    ("TX Houston",           29.76, -95.37, 2000),
    ("MB Winnipeg",          49.90, -97.14, 2000),
    ("SD Sioux Falls",       43.55, -96.73, 2001),
    ("KS Wichita",           37.69, -97.34, 2001),
    ("ND Bismarck",          46.81, -100.78, 2002),
    ("SD Rapid City",        44.08, -103.23, 2002),
    ("NE North Platte",      41.12, -100.77, 2002),
    ("TX Amarillo",          35.22, -101.83, 2002),
    ("TX San Antonio",       29.42, -98.49, 2002),
    # --- Phase 5: the western leap, 2002-2006 ---
    ("MT Missoula",          46.87, -113.99, 2003),
    ("MT Billings",          45.78, -108.50, 2003),
    ("AB Calgary",           51.05, -114.07, 2004),
    ("ID Boise",             43.62, -116.20, 2004),
    ("WA Seattle",           47.61, -122.33, 2004),
    ("WA Spokane",           47.66, -117.43, 2004),
    ("OR Portland",          45.52, -122.68, 2004),
    ("BC Vancouver",         49.28, -123.12, 2005),
    ("WY Casper",            42.87, -106.31, 2005),
    ("UT Salt Lake City",    40.76, -111.89, 2005),
    ("CO Denver",            39.74, -104.99, 2005),
    ("NV Las Vegas",         36.17, -115.14, 2005),
    ("NM Albuquerque",       35.08, -106.65, 2005),
    ("AZ Phoenix",           33.45, -112.07, 2005),
    ("CA Sacramento",        38.58, -121.49, 2006),
    ("CA Los Angeles",       34.05, -118.24, 2006),
    ("CA San Diego",         32.72, -117.16, 2006),
    ("CA Fresno",            36.74, -119.79, 2006),
    # --- Northern Mexico: nominal, essentially undocumented ---
    ("MX Chihuahua",         28.63, -106.08, 2007),
    ("MX Monterrey",         25.69, -100.32, 2007),
    ("MX Hermosillo",        29.07, -110.96, 2007),
    ("MX Mexico City",       19.43, -99.13, 2008),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth-km", type=float, default=300.0,
                    help="Gaussian kernel bandwidth for the year surface.")
    ap.add_argument("--grid", default=None,
                    help="Reference raster defining the model grid "
                         "(default: age-model ocean mask, else data/ref_grid_27km.tif).")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_age_model_config()
    grid_path = args.grid or cfg["ocean_mask"]
    if not os.path.exists(grid_path):
        fallback = os.path.join(load_data_config().get("data_dir", str(_REPO / "data")),
                                "ref_grid_27km.tif")
        if not os.path.exists(fallback):
            fallback = str(_REPO / "data" / "ref_grid_27km.tif")  # repo-relative, not cwd-relative
        print(f"  {grid_path} unavailable; falling back to {fallback}")
        grid_path = fallback

    with rasterio.open(grid_path) as src:
        Ny, Nx = src.height, src.width
        transform, crs = src.transform, src.crs

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(cfg["input_dir"].rstrip("/")), "..", "disease")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Cell-center coordinates in the grid CRS (projected metres).
    rows, cols = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
    gx, gy = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)

    lons = np.array([a[2] for a in ANCHORS], dtype=np.float64)
    lats = np.array([a[1] for a in ANCHORS], dtype=np.float64)
    years = np.array([a[3] for a in ANCHORS], dtype=np.float64)
    ax, ay = project_coords("EPSG:4326", crs, lons.tolist(), lats.tolist())
    ax = np.asarray(ax, dtype=np.float64)
    ay = np.asarray(ay, dtype=np.float64)

    # Nadaraya-Watson smoother. Distances in km; log-sum-exp for stability so
    # far-field cells fall back to the anchor-weighted mean rather than 0/0.
    h = float(args.bandwidth_km)
    d2 = ((gx[:, None] - ax[None, :]) ** 2 + (gy[:, None] - ay[None, :]) ** 2) / 1e6
    logw = -0.5 * d2 / (h ** 2)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    arrival = (w @ years) / w.sum(axis=1)
    arrival = arrival.reshape(Ny, Nx).astype(np.float32)

    print(f"  Grid {Ny}x{Nx} @ {transform[0]/1000:.0f} km, bandwidth {h:.0f} km, "
          f"{len(ANCHORS)} anchors")
    print(f"  Arrival year range: {arrival.min():.1f} - {arrival.max():.1f} "
          f"(median {np.median(arrival):.1f})")

    tif_path = os.path.join(out_dir, "disease_arrival_year_27km.tif")
    with rasterio.open(tif_path, "w", driver="GTiff", height=Ny, width=Nx, count=1,
                       dtype="float32", crs=crs, transform=transform,
                       nodata=np.float32(-9999.0), compress="deflate") as dst:
        dst.write(arrival, 1)
        dst.update_tags(source="scripts/build_disease_arrival_map.py",
                        method="Nadaraya-Watson on documented spread anchors",
                        bandwidth_km=str(h), n_anchors=str(len(ANCHORS)),
                        units="first breeding season with disease present")
    print(f"  Wrote {tif_path}")

    csv_path = os.path.join(out_dir, "disease_arrival_anchors.csv")
    with open(csv_path, "w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(["label", "lat", "lon", "arrival_year"])
        wri.writerows(ANCHORS)
    print(f"  Wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axp = plt.subplots(figsize=(11, 6.5))
        im = axp.imshow(arrival, cmap="magma_r", vmin=1994, vmax=2007)
        cs = axp.contour(arrival, levels=np.arange(1995, 2008), colors="k",
                         linewidths=0.4, alpha=0.5)
        axp.clabel(cs, fmt="%d", fontsize=6)
        r, c = rasterio.transform.rowcol(transform, ax.tolist(), ay.tolist())
        axp.scatter(c, r, s=6, c="cyan", edgecolors="none")
        axp.set(title=f"House Finch conjunctivitis: modeled arrival year "
                      f"(bandwidth {h:.0f} km)", xticks=[], yticks=[])
        fig.colorbar(im, ax=axp, label="first affected breeding season")
        png = os.path.join(out_dir, "disease_arrival_year_27km.png")
        fig.savefig(png, dpi=170, bbox_inches="tight")
        print(f"  Wrote {png}")
    except ImportError:
        print("  matplotlib unavailable; skipped preview")


if __name__ == "__main__":
    main()
