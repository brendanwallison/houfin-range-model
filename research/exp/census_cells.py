"""County agricultural census (ICPSR 35206) by TYPE and PRACTICE on the 27 km grid: early (1978, 1982) vs modern
(2002, 2007, 2012).

    python research/exp/census_cells.py --icpsr <.../ICPSR_35206> --counties <cb_2018_us_county_500k.shp>
        --grid <ref_grid_27km.tif> --out <census_27km.npz>

Channels (acres as a share of county LAND area; cattle as head per km2 / 100), county rows only (level 1), averaged
over each epoch's census years, then area-weighted onto cells (27 x 27 sub-pixels per cell):
  crop_nopast  total cropland minus cropland pasture (the inventory's robust cropland measure)
  harvested    harvested cropland             pasture   all pastureland (incl. cropland & woodland pasture)
  woodland     woodland on farms              hay       hay (2002+ adds haylage / grass silage / greenchop)
  rowcrop      corn grain + corn silage + soybeans + sorghum grain + cotton
  smallgrain   wheat + oats + barley
  insect       acres treated for insects      herbicide acres treated for weeds
  fert         acres fertilized (2002+ wording adds lime and soil conditioners)
  irrigated    irrigated land                 cattle    cattle and calves, head / km2 / 100
  crp          CRP / WRP enrollment (0 before the program, 1986: the early epoch truly had none)
Blank crop / chemical cells are read as 0 (mostly "none"; withheld small-farm counties are a stated caveat); hidden
match codes (>= 3e8) are dropped. Miami-Dade 12025 -> 12086. Output for guild_dose.py --aux: years 1940-2025, early
value before 1996 and modern after; keys census (all 13), census_land (crop_nopast..smallgrain, irrigated, cattle,
crp), census_chem (insect, herbicide, fert); census_mask = cells with county data.
"""
import argparse
import glob
import json
import os

import numpy as np

NAMES = ("crop_nopast", "harvested", "pasture", "woodland", "hay", "rowcrop", "smallgrain", "insect", "herbicide",
         "fert", "irrigated", "cattle", "crp")
LAND = ("crop_nopast", "harvested", "pasture", "woodland", "hay", "rowcrop", "smallgrain", "irrigated", "cattle", "crp")
CHEM = ("insect", "herbicide", "fert")
E78 = {"cropland": ["item01014"], "cpast": ["item05037"], "harvested": ["item01016"], "pasture": ["item05061"],
       "woodland": ["item05049"], "hay": ["item01107"],
       "rowcrop": ["item01056", "item01059", "item01086", "item01062", "item01080"],
       "smallgrain": ["item01065", "item01071", "item01068"], "insect": ["item09011"], "herbicide": ["item09017"],
       "fert": ["item09002"], "irrigated": ["item01018"], "cattle": ["item01038"], "crp": []}
SPEC = {1978: ("DS0039", E78), 1982: ("DS0040", E78),
        2002: ("DS0044", {"cropland": ["item01015"], "cpast": ["item08084"], "harvested": ["item01017"],
                          "pasture": ["item08124"], "woodland": ["item08104"], "hay": ["item01117"],
                          "rowcrop": ["item01063", "item01066", "item01093", "item01087", "item01099"],
                          "smallgrain": ["item01069", "item01081", "item01084"], "insect": ["item39019"],
                          "herbicide": ["item39023"], "fert": ["item39003"], "irrigated": ["item01019"],
                          "cattle": ["item01045"], "crp": ["item08128"]}),
        2007: ("DS0045", {"cropland": ["data1_15"], "cpast": ["data8_84"], "harvested": ["data1_17"],
                          "pasture": ["data8_124"], "woodland": ["data8_104"], "hay": ["data1_111"],
                          "rowcrop": ["data1_63", "data1_66", "data1_93", "data1_87", "data1_99"],
                          "smallgrain": ["data1_69", "data1_81", "data1_84"], "insect": ["data42_19"],
                          "herbicide": ["data42_23"], "fert": ["data42_3"], "irrigated": ["data1_19"],
                          "cattle": ["data1_45"], "crp": ["data8_128"]}),
        2012: ("DS0047", {"cropland": ["DATA1_16"], "cpast": ["DATA8_84"], "harvested": ["DATA1_18"],
                          "pasture": ["DATA8_124"], "woodland": ["DATA8_104"], "hay": ["DATA1_112"],
                          "rowcrop": ["DATA1_64", "DATA1_67", "DATA1_94", "DATA1_88", "DATA1_100"],
                          "smallgrain": ["DATA1_70", "DATA1_82", "DATA1_85"], "insect": ["DATA41_19"],
                          "herbicide": ["DATA41_23"], "fert": ["DATA41_3"], "irrigated": ["DATA1_20"],
                          "cattle": ["DATA1_46"], "crp": ["DATA8_128"]})}
EARLY, MODERN = (1978, 1982), (2002, 2007, 2012)


def read_year(root, ds, spec):
    import pandas as pd
    f = glob.glob(os.path.join(root, ds, "*-Data.tsv"))[0]
    hdr = pd.read_csv(f, sep="\t", nrows=0).columns
    low = {c.lower(): c for c in hdr}
    need = {c for v in spec.values() for c in v}
    fips_c, lev_c = low["fips"], low["level"]
    df = pd.read_csv(f, sep="\t", usecols=[fips_c, lev_c] + [low[c.lower()] for c in need], dtype=str)
    df = df[pd.to_numeric(df[lev_c], errors="coerce") == 1]
    fips = pd.to_numeric(df[fips_c], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    fips = fips.replace({"12025": "12086"})
    out = {}
    for k, cols in spec.items():
        if not cols:
            out[k] = pd.Series(0.0, index=fips.values)
            continue
        acc = None
        for c in cols:
            v = pd.to_numeric(df[low[c.lower()]], errors="coerce")
            v = v.where(v < 3e8)                                   # hidden match codes
            v = v.fillna(0.0) if k not in ("cropland", "harvested", "pasture", "woodland") else v
            acc = v if acc is None else acc + v
        out[k] = pd.Series(acc.to_numpy(float), index=fips.values)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--icpsr", required=True); ap.add_argument("--counties", required=True)
    ap.add_argument("--grid", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fine", type=int, default=27)
    a = ap.parse_args()
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio.features import rasterize
    yrs = {y: read_year(a.icpsr, ds, sp) for y, (ds, sp) in SPEC.items()}
    sums = {y: {k: round(float(d[k].sum()) / 1e6, 2) for k in d.columns} for y, d in yrs.items()}
    g = gpd.read_file(a.counties)
    g["FIPS"] = g["STATEFP"].astype(str).str.zfill(2) + g["COUNTYFP"].astype(str).str.zfill(3)
    acres_land = g["ALAND"].astype(float).to_numpy() / 4046.8564224
    km2_land = g["ALAND"].astype(float).to_numpy() / 1e6

    def channels(d):
        d = d.reindex(g["FIPS"].to_numpy())
        ch = {"crop_nopast": d["cropland"] - d["cpast"].fillna(0.0), "harvested": d["harvested"],
              "pasture": d["pasture"], "woodland": d["woodland"], "hay": d["hay"], "rowcrop": d["rowcrop"],
              "smallgrain": d["smallgrain"], "insect": d["insect"], "herbicide": d["herbicide"], "fert": d["fert"],
              "irrigated": d["irrigated"], "crp": d["crp"]}
        M = np.stack([np.clip(ch[k].to_numpy(float) / np.maximum(acres_land, 1.0), 0, 2) if k in ch else
                      d["cattle"].to_numpy(float) / np.maximum(km2_land, 1e-6) / 100.0 for k in NAMES], 0)
        return M
    ep = {}
    for name, ys in (("early", EARLY), ("modern", MODERN)):
        with np.errstate(all="ignore"):
            ep[name] = np.nanmean(np.stack([channels(yrs[y]) for y in ys], 0), 0)    # (K, counties)
    matched = {y: int(np.isin(g["FIPS"].to_numpy(), yrs[y].index.to_numpy()).sum()) for y in yrs}
    grid = rasterio.open(a.grid)
    H, W, f = grid.height, grid.width, a.fine
    gg = g.to_crs(grid.crs)
    ids = rasterize(((geom, i + 1) for i, geom in enumerate(gg.geometry)), out_shape=(H * f, W * f),
                    transform=grid.transform * rasterio.Affine.scale(1.0 / f), fill=0, dtype="int32")
    cid = ids.reshape(H, f, W, f).transpose(0, 2, 1, 3).reshape(H, W, f * f)
    cells = {}
    for name in ("early", "modern"):
        V = np.concatenate([np.full((len(NAMES), 1), np.nan), ep[name]], 1)          # id 0 = no county
        Vg = V[:, cid]                                                                # (K, H, W, f*f)
        with np.errstate(all="ignore"):
            cells[name] = np.nanmean(Vg, -1)
    mask = np.isfinite(cells["early"]).all(0) & np.isfinite(cells["modern"]).all(0)
    years = np.arange(1940, 2026)
    arr = np.where((years < 1996)[:, None, None, None], cells["early"][None], cells["modern"][None])
    arr = np.where(mask[None, None], arr, np.nan).astype("float32")
    li = [NAMES.index(k) for k in LAND]; ci = [NAMES.index(k) for k in CHEM]
    info = {"national_sums_Macres": sums, "counties_matched": matched, "cells": int(mask.sum()),
            "early_years": EARLY, "modern_years": MODERN}
    np.savez(a.out, years=years, census=arr, census_land=arr[:, li], census_chem=arr[:, ci], census_mask=mask,
             census_names=np.array(NAMES), info=np.array(json.dumps(info)))
    print(f"[census] -> {a.out}: cells {int(mask.sum())}, counties matched {matched}")
    print(pd.DataFrame(sums).to_string())
    for k_, nm in enumerate(NAMES):
        e, m = cells["early"][k_][mask], cells["modern"][k_][mask]
        print(f"  {nm:12s} cell mean early {np.mean(e):.4f} modern {np.mean(m):.4f}  corr(early, modern) "
              f"{np.corrcoef(e, m)[0, 1]:+.2f}  sd(change) {np.std(m - e):.4f}")


if __name__ == "__main__":
    main()
