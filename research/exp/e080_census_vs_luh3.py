"""E080: does LUH3 land use (DESK's covariate) agree with the county agricultural census, 1929-1982?

    python research/exp/e080_census_vs_luh3.py --icpsr <ICPSR_35206 dir> --counties cb_2018_us_county_500k.shp
        --out DIR

Census (Haines, Fishback & Rhode, ICPSR 35206; county level): harvested cropland and irrigated land as a share of the
county's land area, for 1929 (DS0013 var13 / var6; irrigated DS0019 var2524), 1939 (DS0016 var10 / var5; DS0019
var2520), 1949 (DS0021 var15 / var2, var49), 1959 (DS0027 var11 / var3, var54), 1969 (DS0033 item01038 / item01004,
item01057) and 1982 (DS0040 item01016 / item05006, item01018).
LUH3 (raw yearly states, the year itself): cropland = sum of the five crop fractions; irrigated = sum over crop types
of crop fraction x irrigated fraction. Averaged over each county on a sub-grid (each 27-km cell split 9 x 9). Counties:
2018 boundaries, kept when the census land area is within 10% of the polygon area (boundary changes) in that year.
Zones (great_plains_zones_27km.tif) by the county's majority sub-pixel.
Per zone (GREAT PLAINS primary; west, east, all): Spearman across counties of the census and LUH3 shares per year
(LEVEL); Spearman of each county's change between years (CHANGE: 1929->1949, 1949->1969, 1929->1982); and the ratio of
zone totals (census and LUH3) later/earlier (TREND).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from anchored_delta import STATES, ZONES_TIF  # noqa: E402

SPEC = {1929: ("DS0013", "var13", "var6", ("DS0019", "var2524")),
        1939: ("DS0016", "var10", "var5", ("DS0019", "var2520")),
        1949: ("DS0021", "var15", "var2", ("DS0021", "var49")),
        1959: ("DS0027", "var11", "var3", ("DS0027", "var54")),
        1969: ("DS0033", "item01038", "item01004", ("DS0033", "item01057")),
        1982: ("DS0040", "item01016", "item05006", ("DS0040", "item01018"))}
CHANGES = ((1929, 1949), (1949, 1969), (1929, 1982))


def read_part(root, ds, cols):
    import glob
    import pandas as pd
    f = glob.glob(os.path.join(root, ds, "*-Data.tsv"))[0]
    hdr = pd.read_csv(f, sep="\t", nrows=0).columns
    low = {c.lower(): c for c in hdr}
    use = [low["fips"], low["level"]] + [low[c.lower()] for c in cols]
    d = pd.read_csv(f, sep="\t", usecols=use, low_memory=False)
    d.columns = [c.lower() for c in d.columns]
    d = d[pd.to_numeric(d["level"], errors="coerce") == 1]
    for c in cols:
        d[c.lower()] = pd.to_numeric(d[c.lower()], errors="coerce")
    d["fips"] = pd.to_numeric(d["fips"], errors="coerce")
    return d.dropna(subset=["fips"]).set_index("fips")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--icpsr", required=True); ap.add_argument("--counties", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import geopandas as gpd
    import rasterio
    from rasterio import features
    from scipy.stats import spearmanr
    from src.community_encoder.train_DESK import covariate_io as cio
    with rasterio.open(ZONES_TIF) as src:
        zones, tr, crs = src.read(1), src.transform, src.crs
    H, W = zones.shape
    k = 9
    tr_f = tr * tr.scale(1 / k, 1 / k)
    cty = gpd.read_file(a.counties).to_crs(crs)
    cty["fips"] = cty["GEOID"].astype(int)
    cty = cty[~cty["STATEFP"].isin(["02", "15", "60", "66", "69", "72", "78"])]
    lab = features.rasterize(zip(cty.geometry, cty["fips"]), out_shape=(H * k, W * k), transform=tr_f, fill=0,
                             dtype="int32")
    poly_acres = cty.set_index("fips").geometry.area / 4046.86
    zf = np.repeat(np.repeat(zones, k, 0), k, 1)
    ids, inv = np.unique(lab.ravel(), return_inverse=True)
    cnt = np.bincount(inv)
    zmaj = {}
    for z in (1, 2, 3):
        zc = np.bincount(inv, weights=(zf.ravel() == z).astype(float))
        for i, f in enumerate(ids):
            if f and zc[i] > zmaj.get(f, (0, 0))[0]:
                zmaj[f] = (zc[i], z)
    schema = cio.load_schema(STATES); ydir = os.path.join(STATES, "yearly_states")
    names, stream = [], []
    for st in schema["streams"]:
        v = st.get("variables") or [f"{st['name']}_{i}" for i in range(int(st["dim"]))]
        v = list(v) if len(v) == int(st["dim"]) else [f"{st['name']}_{i}" for i in range(int(st["dim"]))]
        names += v; stream += [st["name"]] * int(st["dim"])
    names, stream = np.array(names), np.array(stream)
    ix = lambda n: int(np.where((stream == "landuse") & (names == n))[0][0])
    crops = ["c3ann", "c3nfx", "c3per", "c4ann", "c4per"]
    luh, cen = {}, {}
    for y, (ds, hv, lv, (ids_, iv)) in SPEC.items():
        s = cio.load_state_stack(y, ydir, schema)
        crop = np.nansum(np.stack([s[..., ix(c)] for c in crops], -1), -1)
        irr = np.nansum(np.stack([s[..., ix(c)] * s[..., ix("irrig_" + c)] for c in crops], -1), -1)
        out = {}
        for nm, g in (("crop", crop), ("irr", irr)):
            gf = np.repeat(np.repeat(np.nan_to_num(g), k, 0), k, 1).ravel()
            out[nm] = dict(zip(ids, np.bincount(inv, weights=gf) / np.maximum(cnt, 1)))
        luh[y] = out
        d = read_part(a.icpsr, ds, [hv, lv]); di = read_part(a.icpsr, ids_, [iv])
        area = d[lv.lower()]
        ok = area.index.isin(poly_acres.index)
        rel = (area[ok] / poly_acres.reindex(area[ok].index)).between(0.9, 1.1)
        good = area[ok][rel].index
        cen[y] = {"crop": (d.loc[good, hv.lower()] / area[good]).to_dict(),
                  "irr": (di[iv.lower()].reindex(good) / area[good]).to_dict()}
    res = {}
    for zone_name, zsel in (("great_plains", {2}), ("west", {1}), ("east", {3}), ("all", {1, 2, 3})):
        r = {}
        for var in ("crop", "irr"):
            lv_ = {}
            for y in SPEC:
                f = [c for c in cen[y][var] if zmaj.get(c, (0, 0))[1] in zsel and c in luh[y][var]
                     and np.isfinite(cen[y][var][c])]
                if len(f) >= 20:
                    x = np.array([cen[y][var][c] for c in f]); l_ = np.array([luh[y][var][c] for c in f])
                    lv_[str(y)] = {"counties": len(f), "spearman": float(spearmanr(x, l_).correlation),
                                   "census_total_share": float(x.mean()), "luh3_total_share": float(l_.mean())}
            ch = {}
            for y0, y1 in CHANGES:
                f = [c for c in cen[y0][var] if c in cen[y1][var] and zmaj.get(c, (0, 0))[1] in zsel
                     and np.isfinite(cen[y0][var][c]) and np.isfinite(cen[y1][var][c])]
                if len(f) >= 20:
                    dc = np.array([cen[y1][var][c] - cen[y0][var][c] for c in f])
                    dl = np.array([luh[y1][var][c] - luh[y0][var][c] for c in f])
                    c0 = np.array([cen[y0][var][c] for c in f]); c1 = np.array([cen[y1][var][c] for c in f])
                    l0 = np.array([luh[y0][var][c] for c in f]); l1 = np.array([luh[y1][var][c] for c in f])
                    ch[f"{y0}-{y1}"] = {"counties": len(f), "spearman_change": float(spearmanr(dc, dl).correlation),
                                        "census_trend_ratio": float(c1.mean() / max(c0.mean(), 1e-9)),
                                        "luh3_trend_ratio": float(l1.mean() / max(l0.mean(), 1e-9))}
            r[var] = {"level": lv_, "change": ch}
        res[zone_name] = r
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e080.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
