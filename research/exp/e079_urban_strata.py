"""E079 data step: urbanization strata of the cells where BBS and CBC co-occur (season-link test by urbanization).

    python research/exp/e079_urban_strata.py --cbc cbc_circles_comm.npz --cache <research cache> --out urban_strata.npz

From the yearly states (DESK's transforms), averaged over 1990-2010: HISDAC built-up intensity within-cell median
(bui_q50) and 99th percentile (bui_q99), and the LUH2 urban fraction. Among cells holding both a CBC circle (1966-2024)
and BBS routes (the cache keys), strata (each an (H, W) bool mask):
  urb_low / urb_mid / urb_high       terciles of bui_q50 (how built-up the typical part of the cell is)
  town_in_rural                       bui_q99 in the top half AND bui_q50 in the bottom half (a town in farmland)
  not_town_in_rural                   the other co-occurrence cells
  luh_low / luh_mid / luh_high        terciles of the LUH2 urban fraction (reported)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from anchored_delta import STATES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cbc", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.community_encoder.train_DESK import covariate_io as cio
    schema = cio.load_schema(STATES)
    ydir = os.path.join(STATES, "yearly_states")
    names, stream = [], []
    for st in schema["streams"]:
        v = st.get("variables") or [f"{st['name']}_{i}" for i in range(int(st["dim"]))]
        v = list(v) if len(v) == int(st["dim"]) else [f"{st['name']}_{i}" for i in range(int(st["dim"]))]
        names += v; stream += [st["name"]] * int(st["dim"])
    names, stream = np.array(names), np.array(stream)
    ix = lambda sname, n: int(np.where((stream == sname) & (names == n))[0][0])
    acc = None
    for y in range(1990, 2011):
        s = cio.load_state_stack(y, ydir, schema)
        v = np.stack([s[..., ix("bui", "bui_q50")], s[..., ix("bui", "bui_q99")], s[..., ix("landuse", "urban")]], -1)
        acc = v if acc is None else acc + v
    q50, q99, luh = np.moveaxis(acc / 21.0, -1, 0)
    D = np.load(a.cbc, allow_pickle=False)
    keys = np.load(os.path.join(a.cache, "keys.npy"))
    H, W = q50.shape
    cbc = np.zeros((H, W), bool)
    m = (D["row"] >= 0) & (D["year_end"] >= 1966)
    cbc[D["row"][m], D["col"][m]] = True
    bbs = np.zeros((H, W), bool); bbs[keys[:, 0], keys[:, 1]] = True
    both = cbc & bbs & np.isfinite(q50) & np.isfinite(q99) & np.isfinite(luh)
    t = lambda x: np.quantile(x[both], [1 / 3, 2 / 3])
    a50, a99, al = t(q50), np.median(q99[both]), t(luh)
    out = {"both": both,
           "urb_low": both & (q50 <= a50[0]), "urb_mid": both & (q50 > a50[0]) & (q50 <= a50[1]), "urb_high": both & (q50 > a50[1]),
           "town_in_rural": both & (q99 > a99) & (q50 <= np.median(q50[both])),
           "luh_low": both & (luh <= al[0]), "luh_mid": both & (luh > al[0]) & (luh <= al[1]), "luh_high": both & (luh > al[1])}
    out["not_town_in_rural"] = both & ~out["town_in_rural"]
    np.savez_compressed(a.out, **out)
    print("[E079] cells:", {k: int(v.sum()) for k, v in out.items()})


if __name__ == "__main__":
    main()
