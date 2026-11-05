"""Render existing hypothesis schematics into figures/external.

Run with --ecoregions PATH pointing to a CEC shapefile with NA_L1NAME.
The Level III release retains Level I labels and is dissolved by the original
schematic code. Uses local Natural Earth 50m country polygons for land outlines.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hypothesis_scenarios as scenarios
import hypothesis_r_field as field
import r_field_stack as stack
from src.config_utils import load_data_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecoregions", required=True)
    args = parser.parse_args()
    cfg = deepcopy(load_data_config(ROOT / "config/data_config.json"))
    cfg["datasets_root"] = str(ROOT / "data")
    cfg["grid"]["ref_raster"] = str(ROOT / "data/audubon/houfin/generalists_HOFI_houfin_breeding_2010_NA_NA_range_27km.tif")
    cfg["coastline"]["land_source"] = "naturalearth/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp"
    scenarios.base.load_data_config = lambda: deepcopy(cfg)
    scenarios.base.ECOREGION_SHP = args.ecoregions
    out = ROOT / "figures/external"
    out.mkdir(parents=True, exist_ok=True)
    scenarios.OUT_DIR = str(out)
    field.OUT_DIR = str(out)
    stack.OUT_DIR = str(out)
    stack.OUT_PNG = str(out / "hypothesis_r_field_stack.png")
    scenarios.main()
    field.main()
    stack.main()


if __name__ == "__main__":
    main()
