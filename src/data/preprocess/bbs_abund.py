"""Align USGS BBS abundance rasters (ra{AOU}.tif, birds/route) onto the model grid.

Companion to ``bbs_trend`` (shares its builder). The BBS abundance product is the
2018-2022 mean relative abundance (birds per survey route) on the same 27 km
ESRI:102003 lattice as the trend rasters, so alignment is a nearest clip/pad with
zero resampling. Used ONLY as a reliability/scale signal for the method-B deep
reconstruction (``trend_community``: deep past = k·B/f, tying the historical spatial
pattern to BBS) -- never as community abundance values (those come from eBird).

Output ``trends.bbs_abund_grid`` (.npz): ``abund`` (n_species, H, W) float32,
``species_code``, ``aou``, ``valid``.

    python -m src.data.preprocess.bbs_abund
"""
import argparse
import os

from src.config_utils import load_data_config
from src.data.preprocess.bbs_trend import build


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default=None, help="community_trend.csv (default: config).")
    ap.add_argument("--abund-dir", default=None, help="Dir of BBS ra{AOU}.tif (default: config).")
    ap.add_argument("--out", default=None, help="Output .npz (default: trends.bbs_abund_grid).")
    args = ap.parse_args()

    dcfg = load_data_config()
    dr = dcfg["datasets_root"]
    community = args.community or dcfg["community_trend_list"]
    abund_dir = args.abund_dir or os.path.join(dr, dcfg["sciencebase"]["out_subdirs"]["bbs_abundance"])
    out = args.out or dcfg["trends"]["bbs_abund_grid"]
    build(community, abund_dir, out, prefix="ra", field="abund")


if __name__ == "__main__":
    main()
