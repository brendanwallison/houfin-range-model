"""E061: build DESK's auxiliary CBC target from Audubon's CBC relative-abundance indices (version 2023.0).

    python research/exp/build_cbc_aux.py --cache <research cache> --cbc-csv <indices csv> --routes <BBS Routes.csv>
        --out <npz> --min-year 2001 [--max-dist-km 150]

The CBC product gives, per species, an effort-corrected annual relative-abundance index for each MODELING STRATUM
(a state/province x Bird Conservation Region intersection, e.g. US-AZ-33), winters 1966-67 .. 2023-24. A winter count
ending in January of year y is labelled y (the winter before breeding season y).

Stratum map on the 27 km grid: with --strata-shp (the BBS "USGS" stratification polygons, state/province x BCR, field
ST_12 in CBC's own code format, e.g. US-KY-28; shipped in the bbsBayes R package, inst/maps/BBS_USGS_strata.shp) each
cell takes the polygon containing its centre; --strata-shp2 (BBS_CWS_strata.shp) fills strata only it carries.
Without polygons the fallback is the NEAREST BBS route start (Routes.csv: state/province and BCR) within
--max-dist-km. Only strata the CBC file models are kept.

Species: scientific name -> eBird code through the eBird taxonomy (pooled multi-species rows dropped). Kept: the 96
REFERENCE species and species that are graded NOWHERE (neither dev nor sealed BBS species -- mostly winter visitors
absent from BBS), with survey_suitability medium or high and modelled in >= --min-strata strata. Dev and sealed
species never enter (they would contaminate the species-level grading); House Finch never.

Target, (n_years, H, W, 2S): S LEVEL columns = log1p(stratum index) averaged over the kept years (static), and S
TEMPORAL-DEVIATION columns = log1p(index in year y) minus that stratum mean -- the same split as E060's eBird ST
target, so the trainer's per-column standardization keeps the temporal part from vanishing. A species not modelled in
a stratum is written as 0 (rare or absent there in winter).

Years: --min-year .. 2024. LEAKAGE GUARD: the research models withhold 1966..T0-1, and the CBC model smooths across
years, so indices just after T0 borrow strength from withheld winters. --min-year must be >= T0 + 5 (default 2001 for
T0 1996); the trainer additionally drops every withheld year. The output name should carry T0.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

# BBS CountryNum/StateNum -> CBC region prefix (US-XX / CA-XX)
BBS_REGION = {
    (840, 2): "US-AL", (840, 3): "US-AK", (840, 6): "US-AZ", (840, 7): "US-AR", (840, 14): "US-CA", (840, 17): "US-CO",
    (840, 18): "US-CT", (840, 21): "US-DE", (840, 25): "US-FL", (840, 27): "US-GA", (840, 33): "US-ID", (840, 34): "US-IL",
    (840, 35): "US-IN", (840, 36): "US-IA", (840, 38): "US-KS", (840, 39): "US-KY", (840, 42): "US-LA", (840, 44): "US-ME",
    (840, 46): "US-MD", (840, 47): "US-MA", (840, 49): "US-MI", (840, 50): "US-MN", (840, 51): "US-MS", (840, 52): "US-MO",
    (840, 53): "US-MT", (840, 54): "US-NE", (840, 55): "US-NV", (840, 58): "US-NH", (840, 59): "US-NJ", (840, 60): "US-NM",
    (840, 61): "US-NY", (840, 63): "US-NC", (840, 64): "US-ND", (840, 66): "US-OH", (840, 67): "US-OK", (840, 69): "US-OR",
    (840, 72): "US-PA", (840, 77): "US-RI", (840, 80): "US-SC", (840, 81): "US-SD", (840, 82): "US-TN", (840, 83): "US-TX",
    (840, 85): "US-UT", (840, 87): "US-VT", (840, 88): "US-VA", (840, 89): "US-WA", (840, 90): "US-WV", (840, 91): "US-WI",
    (840, 92): "US-WY",
    (124, 4): "CA-AB", (124, 11): "CA-BC", (124, 43): "CA-NT", (124, 45): "CA-MB", (124, 56): "CA-NB", (124, 57): "CA-NL",
    (124, 62): "CA-NU", (124, 65): "CA-NS", (124, 68): "CA-ON", (124, 75): "CA-PE", (124, 76): "CA-QC", (124, 79): "CA-SK",
    (124, 93): "CA-YT",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--cbc-csv", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--taxonomy", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-year", type=int, default=2001)
    ap.add_argument("--max-year", type=int, default=2024)
    ap.add_argument("--max-dist-km", type=float, default=150.0)
    ap.add_argument("--min-strata", type=int, default=5)
    ap.add_argument("--strata-shp", default=None, help="state/province x BCR polygons with an ST_12 field")
    ap.add_argument("--strata-shp2", default=None, help="second polygon set; fills codes the first lacks")
    a = ap.parse_args()
    import pandas as pd
    import rasterio
    from rasterio.warp import transform as warp_transform
    from scipy.spatial import cKDTree
    from src.config_utils import load_data_config

    dcfg = load_data_config()
    tax_path = a.taxonomy or os.path.join(dcfg["datasets_root"], "avonet", "eBird_taxonomy.csv")
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    ref, dev, sealed = set(meta["community"]), set(meta["dev_species"]), set(meta["sealed_species"])
    graded = dev | sealed | {"houfin"}

    with rasterio.open(dcfg["grid"]["ref_raster"]) as r:
        crs, tr, H, W = r.crs, r.transform, r.height, r.width
    rt = pd.read_csv(a.routes, dtype=str, encoding="latin-1")
    rt["key"] = [BBS_REGION.get((int(c), int(s))) for c, s in zip(rt["CountryNum"], rt["StateNum"])]
    rt = rt[rt["key"].notna() & rt["BCR"].str.strip().ne("")]
    rt["stratum"] = rt["key"] + "-" + rt["BCR"].str.strip().astype(int).astype(str)
    xs, ys = warp_transform("EPSG:4326", crs, rt["Longitude"].astype(float).tolist(),
                            rt["Latitude"].astype(float).tolist())
    gx = tr.c + tr.a * (np.arange(W) + 0.5)
    gy = tr.f + tr.e * (np.arange(H) + 0.5)
    GX, GY = np.meshgrid(gx, gy)
    if a.strata_shp:
        import geopandas as gpd
        from rasterio.features import rasterize
        cell_stratum = np.full((H, W), None, dtype=object)
        source = {}
        for shp in [a.strata_shp] + ([a.strata_shp2] if a.strata_shp2 else []):
            g = gpd.read_file(shp).to_crs(crs)
            codes = [c for c in g["ST_12"].astype(str).unique() if c not in source]
            for c in codes:
                source[c] = os.path.basename(shp)
            lut = {c: i + 1 for i, c in enumerate(codes)}
            gs = g[g["ST_12"].astype(str).isin(lut)]
            ras = rasterize(((geom, lut[str(c)]) for geom, c in zip(gs.geometry, gs["ST_12"])),
                            out_shape=(H, W), transform=tr, fill=0, dtype="int32", all_touched=False)
            inv = {v: k for k, v in lut.items()}
            take = (ras > 0) & np.equal(cell_stratum, None)
            cell_stratum[take] = [inv[v] for v in ras[take]]
        print(f"[E061] strata from polygons: {len(source)} codes "
              f"({sum(v == os.path.basename(a.strata_shp) for v in source.values())} from the first set)", flush=True)
    else:
        dist, idx = cKDTree(np.column_stack([xs, ys])).query(np.column_stack([GX.ravel(), GY.ravel()]))
        cell_stratum = np.where(dist <= a.max_dist_km * 1000.0, rt["stratum"].to_numpy()[idx], None).reshape(H, W)

    d = pd.read_csv(a.cbc_csv, encoding="latin-1")
    d = d[(d["region_type"] == "stratum") & d["survey_suitability"].isin(["medium", "high"])]
    d = d[(d["year_end"] >= a.min_year) & (d["year_end"] <= a.max_year)]
    tax = pd.read_csv(tax_path, usecols=["SCIENTIFIC_NAME", "SPECIES_CODE", "CATEGORY"])
    tax = tax[tax["CATEGORY"] == "species"]
    sci2code = dict(zip(tax["SCIENTIFIC_NAME"], tax["SPECIES_CODE"]))
    d = d[~d["scientific_name"].str.contains("/|,| and ", regex=True, na=True)]
    d["code"] = d["scientific_name"].map(sci2code)
    n_unmapped = int(d["code"].isna().sum())
    d = d[d["code"].notna() & ~d["code"].isin(graded)]
    strata_ok = set(d["region"]) & set(x for x in np.unique(cell_stratum[cell_stratum != None]))  # noqa: E711
    d = d[d["region"].isin(strata_ok)]
    nstr = d.groupby("code")["region"].nunique()
    keep = sorted(nstr[nstr >= a.min_strata].index)
    d = d[d["code"].isin(keep)]
    assert not set(keep) & graded, "a graded or focal species reached the CBC target"

    years = list(range(a.min_year, a.max_year + 1))
    strata = sorted(strata_ok)
    si = {s: i for i, s in enumerate(strata)}
    ci = {c: i for i, c in enumerate(keep)}
    yi = {y: i for i, y in enumerate(years)}
    V = np.zeros((len(years), len(strata), len(keep)), dtype="float64")   # 0 = not modelled there
    V[d["year_end"].map(yi).to_numpy(), d["region"].map(si).to_numpy(), d["code"].map(ci).to_numpy()] = \
        np.log1p(np.clip(d["annual_index"].to_numpy(dtype="float64"), 0, None))
    lev = V.mean(0, keepdims=True)
    dev_ = V - lev
    Vs = np.concatenate([np.repeat(lev, len(years), 0), dev_], axis=-1).astype("float32")   # (n, n_strata, 2S)
    foot = np.array([[c in si for c in row] for row in cell_stratum])
    sidx = np.array([[si.get(c, -1) for c in row] for row in cell_stratum])
    Y = np.full((len(years), H, W, 2 * len(keep)), np.nan, dtype="float32")
    Y[:, foot] = Vs[:, sidx[foot]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, years=np.array(years), Y=Y, footprint=foot, species=np.array(keep))
    info = {"years": [years[0], years[-1]], "strata": len(strata), "footprint_cells": int(foot.sum()),
            "species": len(keep), "reference_species": sorted(set(keep) & ref),
            "n_reference": len(set(keep) & ref), "n_ungraded_other": len(set(keep) - ref),
            "cbc_rows_unmapped_names": n_unmapped,
            "temporal_sd": float(dev_[:, :, :].std()), "level_sd_between_strata": float(lev[0].std())}
    json.dump(info, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(f"[E061] CBC target: {len(keep)} species ({info['n_reference']} reference, {info['n_ungraded_other']} graded "
          f"nowhere), {len(strata)} strata over {int(foot.sum())} cells, winters ending {years[0]}-{years[-1]}; "
          f"temporal sd {info['temporal_sd']:.3f} vs between-strata level sd {info['level_sd_between_strata']:.3f}")


if __name__ == "__main__":
    main()
