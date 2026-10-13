"""E064 data step: raw CBC circle-years (study box, US + Canada) with stratum, effort and zero-filled counts.

    python research/exp/cbc_prepare.py --raw-dir <CBC data request dir> --products-csv <Audubon indices csv>
        --strata-shp A --strata-shp2 B --cache <research cache> --out cbc_circles.npz

Circle-years are every (circle, count) in the species report (any species recorded), inside the study box (lon -125..-66,
lat 24..53), US and Canada; winter END year = 1900 + count_yr (count 1 = Dec 1900 / Jan 1901). Each circle's centre
takes its state/province x BCR stratum from the BBS USGS polygons (CWS fills codes only it has). Effort per circle-year:
party_hours = summed field hours over every travel mode (foot, car, boats, skis, ...; NaN when none recorded --
nearly all counts before 1940), field_counters and max_parties (max over duplicate summary rows), feeder_hrs,
feeder_counters, nocturnal_hrs. Counts: per species the count-day total (0 = recorded in count week only), and 0 for a
species not reported by a circle-year that reported anything (zero-filling, standard for CBC).
Species: the research cache's 96 reference + 273 dev species (never sealed, never House Finch) that Audubon's 2023
product covers with medium/high survey suitability at the stratum level (dev species are used here only to VALIDATE the
model against Audubon; any training target built later uses reference species only).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
PRE = "20250205_FullSet_AllCountries_AllYears_CBC_"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--products-csv", required=True)
    ap.add_argument("--strata-shp", required=True)
    ap.add_argument("--strata-shp2", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    import rasterio
    from src.config_utils import load_data_config
    from cbc_bridge_check import stratum_map

    dcfg = load_data_config()
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    allowed = set(meta["community"]) | set(meta["dev_species"])
    assert "houfin" not in allowed and not allowed & set(meta["sealed_species"])
    tax = pd.read_csv(os.path.join(dcfg["datasets_root"], "avonet", "eBird_taxonomy.csv"),
                      usecols=["SCIENTIFIC_NAME", "SPECIES_CODE", "CATEGORY"])
    tax = tax[tax["CATEGORY"] == "species"]
    sci2code = dict(zip(tax["SCIENTIFIC_NAME"], tax["SPECIES_CODE"]))
    prod = pd.read_csv(a.products_csv, encoding="latin-1",
                       usecols=["scientific_name", "region_type", "survey_suitability"])
    prod = prod[(prod["region_type"] == "stratum") & prod["survey_suitability"].isin(["medium", "high"])]
    prod_codes = {sci2code.get(s) for s in prod["scientific_name"].unique()} - {None}
    species = sorted(prod_codes & allowed)
    ref = sorted(set(species) & set(meta["community"]))
    print(f"[E064] species: {len(species)} ({len(ref)} reference) in the product (medium/high) and the cache lists",
          flush=True)

    raw = lambda n: os.path.join(a.raw_dir, PRE + n + ".csv")
    sp = pd.read_csv(raw("Circle_Species_Report"), encoding="utf-8-sig",
                     usecols=["abbrev", "latitude", "longitude", "country_code", "count_yr", "sci_name", "how_many"])
    cy = sp.drop_duplicates(["abbrev", "count_yr"])[["abbrev", "count_yr", "latitude", "longitude", "country_code"]]
    cy = cy[cy["country_code"].isin(["US", "CA"]) & cy["latitude"].between(24, 53) & cy["longitude"].between(-125, -66)]
    cy = cy.reset_index(drop=True)
    cy["year_end"] = 1900 + cy["count_yr"]
    with rasterio.open(dcfg["grid"]["ref_raster"]) as r:
        crs, tr, H, W = r.crs, r.transform, r.height, r.width
    smap = stratum_map([s for s in (a.strata_shp, a.strata_shp2) if s], H, W, tr, crs)
    from rasterio.warp import transform as warp_transform
    xs, ys = warp_transform("EPSG:4326", crs, cy["longitude"].tolist(), cy["latitude"].tolist())
    col = np.floor((np.asarray(xs) - tr.c) / tr.a).astype(int)
    row = np.floor((np.asarray(ys) - tr.f) / tr.e).astype(int)
    inside = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    cy["row"], cy["col"] = np.where(inside, row, -1), np.where(inside, col, -1)
    cy["stratum"] = [smap[r_, c_] if r_ >= 0 else None for r_, c_ in zip(cy["row"], cy["col"])]
    cy = cy[cy["stratum"].notna()].reset_index(drop=True)

    eff = pd.read_csv(raw("Effort_Many_Types_Report"), encoding="utf-8-sig", usecols=["abbrev", "count_yr", "hours"])
    ph = eff.groupby(["abbrev", "count_yr"])["hours"].sum().rename("party_hours")
    summ = pd.read_csv(raw("Effort_Summary_Report"), encoding="utf-8-sig")
    sm = summ.groupby(["abbrev", "count_yr"])[["field_counters", "feeder_counters", "max_parties", "feeder_hrs",
                                               "nocturnal_hrs"]].max()
    cy = cy.merge(ph, on=["abbrev", "count_yr"], how="left").merge(sm, on=["abbrev", "count_yr"], how="left")
    cy.loc[~(cy["party_hours"] > 0), "party_hours"] = np.nan
    cy.loc[~(cy["field_counters"] > 0), "field_counters"] = np.nan

    sp["code"] = sp["sci_name"].map(sci2code)
    sp = sp[sp["code"].isin(set(species))]
    assert "houfin" not in set(sp["code"])
    key = cy[["abbrev", "count_yr"]].reset_index().rename(columns={"index": "k"})
    sp = sp.merge(key, on=["abbrev", "count_yr"])
    si = {s: i for i, s in enumerate(species)}
    Y = np.zeros((len(cy), len(species)), dtype=np.float32)
    agg = sp.groupby(["k", "code"])["how_many"].sum().reset_index()
    Y[agg["k"].to_numpy(), agg["code"].map(si).to_numpy()] = agg["how_many"].to_numpy(dtype="float32")
    circles = np.array(sorted(cy["abbrev"].unique()))
    ci = {c: i for i, c in enumerate(circles)}
    strata = np.array(sorted(cy["stratum"].unique()))
    sti = {s: i for i, s in enumerate(strata)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out, Y=Y, species=np.array(species), reference=np.array(ref), circles=circles, strata=strata,
        circle_idx=cy["abbrev"].map(ci).to_numpy(np.int32), stratum_idx=cy["stratum"].map(sti).to_numpy(np.int32),
        year_end=cy["year_end"].to_numpy(np.int32), row=cy["row"].to_numpy(np.int32), col=cy["col"].to_numpy(np.int32),
        lat=cy["latitude"].to_numpy(), lon=cy["longitude"].to_numpy(),
        party_hours=cy["party_hours"].to_numpy(np.float32), field_counters=cy["field_counters"].to_numpy(np.float32),
        max_parties=cy["max_parties"].to_numpy(np.float32), feeder_hrs=cy["feeder_hrs"].to_numpy(np.float32),
        feeder_counters=cy["feeder_counters"].to_numpy(np.float32),
        nocturnal_hrs=cy["nocturnal_hrs"].to_numpy(np.float32))
    dec = (cy["year_end"] // 10 * 10)
    info = {"circle_years": int(len(cy)), "circles": int(len(circles)), "strata": int(len(strata)),
            "species": species, "reference": ref, "years": [int(cy.year_end.min()), int(cy.year_end.max())],
            "by_decade": {int(d): {"n": int(n), "ph_share": float(cy.loc[dec == d, "party_hours"].notna().mean()),
                                   "counters_share": float(cy.loc[dec == d, "field_counters"].notna().mean())}
                          for d, n in dec.value_counts().sort_index().items()}}
    json.dump(info, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(f"[E064] {len(cy):,} circle-years, {len(circles):,} circles, {len(strata)} strata, {len(species)} species, "
          f"years {info['years']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
