"""Climate acquire: drive climr (R) to downscale monthly climate on the model grid.

This is the Linux/HPC-native climate route: it shells out to ``climate_climr.R``
via ``Rscript`` (climr is pure-R ClimateNA downscaling — no Windows .exe, no
throttled web API). Prerequisites (external to Python, like the ClimateNA .exe
would have been): R + the ``climr`` package installed, and the elevation step
(preprocess/elevation.py) already run to produce ``cell_centroids.csv`` (the
model-grid centroids + p10/p50/p90 elevation per cell).

For each of the three elevation levels, climr downscales monthly observed
climate (CRU TS temp + GPCC precip) for FIRST_YEAR..END_YEAR at the cell
centroids, giving climate at low/median/high sub-cell elevation directly on the
model grid — no 1 km climate is ever materialized. See docs/TEMPORAL.md and
docs/DATA_SOURCES.md.

Fallbacks (documented, not wired): the Windows ClimateNA .exe via Wine, and the
throttled point web API (api.climatena.ca, 100 pts/req, 100 req/day).

Usage:
    python scripts/climate_climr.py --centroids <dir>/cell_centroids.csv --out <dir>
"""
import argparse
import os
import subprocess
import sys

from src.config_utils import load_data_config
from src.temporal import load_timeline

_R_SCRIPT = os.path.join(os.path.dirname(__file__), "climate_climr.R")


def build_command(centroids_csv, out_dir, start_year, end_year, rscript="Rscript"):
    """Construct the Rscript command (kept pure/testable, separate from execution)."""
    return [rscript, _R_SCRIPT, centroids_csv, out_dir, str(start_year), str(end_year)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centroids", required=True, help="cell_centroids.csv from preprocess/elevation.py")
    ap.add_argument("--out", required=True, help="output dir for climate_{q10,q50,q90}.csv")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    tl = load_timeline(load_data_config())
    cmd = build_command(args.centroids, args.out, tl["first_year"], tl["end_year"], args.rscript)
    print("climr command:", " ".join(cmd))
    if args.dry_run:
        return
    if not os.path.exists(args.centroids):
        raise SystemExit(f"centroids file not found: {args.centroids} (run preprocess/elevation.py first)")
    os.makedirs(args.out, exist_ok=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
