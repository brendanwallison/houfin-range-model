import os
import sys
import rasterio
from pathlib import Path

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from src.config_utils import load_data_config
_CFG = load_data_config()
_DR = _CFG["datasets_root"]

# Metadata sanity-check over the raw downloaded eBird rasters (same dir the
# downloader writes to and project_ebird reads from).
folder = Path(f"{_DR}/{_CFG.get('ebird_raw_subdir', 'ebird_weekly_2023')}")

meta = []

for tif in folder.glob("*.tif"):
    with rasterio.open(tif) as src:
        meta.append({
            "file": tif.name,
            "crs": src.crs,
            "res": src.res,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "transform": src.transform
        })

# Quick check: do all resolutions match?
resolutions = {m["res"] for m in meta}
print("Unique resolutions:", resolutions)

# Check extents
bounds = {m["bounds"] for m in meta}
print("Unique bounds:", len(bounds))

crs_set = {m["crs"] for m in meta}
print("Unique CRS:", crs_set)

transform_set = {m["transform"] for m in meta}
print("Unique transforms:", len(transform_set))
