"""Stream the raw eBird Basic Dataset (EBD) down to the checklists and species this project can use.

The EBD is one tab-separated row per (checklist, taxon), all species, worldwide: far too large to load. This makes one
streaming pass and writes two compact parquet tables:

  checklists.parquet  one row per KEPT checklist: checklist_id, observer_id, locality_id, locality_type, lat, lon,
                      date, year, doy, time_started, protocol, duration_min, distance_km, n_observers
  counts.parquet      one row per (kept checklist, kept species) DETECTION: checklist_id, species_code, count
                      (int32; -1 where the observer wrote "X", present without a count)

A kept checklist with no row in counts.parquet for a species is a genuine zero for that species: only complete
checklists ("ALL SPECIES REPORTED" = 1) are kept, and every complete checklist that reported anything appears in the
EBD, so no sampling-event file is needed for zero-filling.

Filters (defaults follow the usual eBird analysis guidance -- complete checklists, stationary or traveling, <= 5 h,
<= 5 km, <= 10 observers -- plus this project's box and a breeding-season window to match BBS):
  approved records only; complete checklists; protocol Stationary or Traveling; duration <= --max-duration; distance
  <= --max-distance (stationary counts as 0); observers <= --max-observers; years --years; day of year --doy; inside
  --bbox (lon/lat); shared (group) checklists collapsed to one, keeping the lowest checklist id.

Species are mapped by scientific name through the eBird taxonomy (categories species / issf / form / domestic report
at species level in the EBD's SCIENTIFIC NAME column). Only codes in --species-list are kept. HOUSE FINCH (houfin) is
refused outright, and so is any code in --exclude-list (pass the sealed species): the downstream contract forbids
House Finch data anywhere in the encoder, and the sealed half must not shape a representation.

    python -m src.data.preprocess.ebd_reduce --ebd ebd_relXXX.txt.gz --species-list codes.csv --out DIR
        [--exclude-list sealed.txt] [--limit-rows N]

Column names are resolved from the file's own header (the EBD renamed some columns over releases, e.g. PROTOCOL TYPE
-> PROTOCOL NAME), so a release change fails loudly at the header instead of silently reading the wrong column.
--limit-rows reads only the first N data rows: a smoke test that also works on a partially downloaded .gz.
"""
import argparse
import csv
import gzip
import json
import os
import time

import numpy as np
import pandas as pd

FOCAL = "houfin"
# logical field -> accepted EBD header spellings (first match wins)
FIELDS = {
    "checklist_id": ["SAMPLING EVENT IDENTIFIER"],
    "category": ["CATEGORY"],
    "sci": ["SCIENTIFIC NAME"],
    "count": ["OBSERVATION COUNT"],
    "lat": ["LATITUDE"],
    "lon": ["LONGITUDE"],
    "date": ["OBSERVATION DATE"],
    "time_started": ["TIME OBSERVATIONS STARTED"],
    "observer_id": ["OBSERVER ID"],
    "locality_id": ["LOCALITY ID"],
    "locality_type": ["LOCALITY TYPE"],
    "protocol": ["PROTOCOL NAME", "PROTOCOL TYPE", "OBSERVATION TYPE"],
    "duration_min": ["DURATION MINUTES"],
    "distance_km": ["EFFORT DISTANCE KM"],
    "n_observers": ["NUMBER OBSERVERS"],
    "complete": ["ALL SPECIES REPORTED"],
    "group_id": ["GROUP IDENTIFIER"],
    "approved": ["APPROVED"],
}
PROTOCOLS = {"Stationary", "Traveling", "eBird - Stationary Count", "eBird - Traveling Count"}
SPECIES_LEVEL = {"species", "issf", "form", "domestic"}


def resolve_columns(header):
    """Map each logical field to its column name in this file's header. Raises on any missing field."""
    have = set(header)
    out, missing = {}, []
    for f, names in FIELDS.items():
        hit = next((n for n in names if n in have), None)
        if hit is None:
            missing.append(f"{f} (tried {names})")
        out[f] = hit
    if missing:
        raise SystemExit("EBD header lacks required fields: " + "; ".join(missing))
    return out


def sci_to_code(taxonomy_csv, keep, exclude):
    """Scientific name -> species code, restricted to ``keep`` and refusing House Finch and ``exclude``."""
    tax = pd.read_csv(taxonomy_csv, usecols=["SCIENTIFIC_NAME", "SPECIES_CODE", "CATEGORY"])
    tax = tax[tax["CATEGORY"] == "species"]
    keep = set(keep) - {FOCAL} - set(exclude)
    m = {s: c for s, c in zip(tax["SCIENTIFIC_NAME"], tax["SPECIES_CODE"]) if c in keep}
    assert FOCAL not in m.values() and not set(m.values()) & set(exclude)
    return m


def filter_chunk(df, cols, sci2code, years, doy, bbox, max_duration, max_distance, max_observers):
    """Pure: one chunk of raw EBD rows (str dtype) -> (checklist rows, detection rows) passing every filter."""
    g = lambda f: df[cols[f]]
    ok = (g("approved") == "1") & (g("complete") == "1") & g("protocol").isin(PROTOCOLS)
    date = pd.to_datetime(g("date"), errors="coerce", format="%Y-%m-%d")
    ok &= date.notna()
    yr, dy = date.dt.year, date.dt.dayofyear
    ok &= (yr >= years[0]) & (yr <= years[1]) & (dy >= doy[0]) & (dy <= doy[1])
    lat = pd.to_numeric(g("lat"), errors="coerce")
    lon = pd.to_numeric(g("lon"), errors="coerce")
    ok &= (lon >= bbox[0]) & (lat >= bbox[1]) & (lon <= bbox[2]) & (lat <= bbox[3])
    dur = pd.to_numeric(g("duration_min"), errors="coerce")
    dist = pd.to_numeric(g("distance_km"), errors="coerce").fillna(0.0)
    nobs = pd.to_numeric(g("n_observers"), errors="coerce")
    ok &= dur.notna() & (dur <= max_duration) & (dist <= max_distance) & nobs.notna() & (nobs <= max_observers)
    df = df[ok]
    if df.empty:
        return None, None
    cid = df[cols["checklist_id"]]
    grp = df[cols["group_id"]].fillna("")
    chk = pd.DataFrame({
        "checklist_id": cid.values, "group_id": grp.values, "observer_id": df[cols["observer_id"]].values,
        "locality_id": df[cols["locality_id"]].values, "locality_type": df[cols["locality_type"]].values,
        "lat": lat[ok].values.astype("float64"), "lon": lon[ok].values.astype("float64"),
        "date": df[cols["date"]].values, "year": yr[ok].values.astype("int16"), "doy": dy[ok].values.astype("int16"),
        "time_started": df[cols["time_started"]].values, "protocol": df[cols["protocol"]].values,
        "duration_min": dur[ok].values.astype("float32"), "distance_km": dist[ok].values.astype("float32"),
        "n_observers": nobs[ok].values.astype("int16"),
    }).drop_duplicates("checklist_id")
    sp = df[df[cols["category"]].isin(SPECIES_LEVEL)]
    code = sp[cols["sci"]].map(sci2code)
    sp = sp[code.notna()]
    code = code[code.notna()]
    cnt = sp[cols["count"]]
    det = pd.DataFrame({"checklist_id": sp[cols["checklist_id"]].values, "species_code": code.values,
                        "count": np.where(cnt.values == "X", -1,
                                          pd.to_numeric(cnt, errors="coerce").fillna(-1).values).astype("int32")})
    # an issf and its species can both appear on one checklist: sum counts, X (-1) wins only if nothing is counted
    det = det.groupby(["checklist_id", "species_code"], as_index=False)["count"].agg(
        lambda s: int(s[s >= 0].sum()) if (s >= 0).any() else -1)
    assert FOCAL not in set(det["species_code"])
    return chk, det


def _open(path):
    raw = gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") else \
        open(path, "rt", encoding="utf-8", newline="")
    return raw


def reduce(ebd, out, sci2code, years, doy, bbox, max_duration, max_distance, max_observers,
           chunk_rows=1_000_000, limit_rows=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    # explicit schemas: a chunk where a column is all-missing would otherwise infer a different type and the
    # writer would refuse it mid-run
    s = pa.string()
    chk_schema = pa.schema([("checklist_id", s), ("group_id", s), ("observer_id", s), ("locality_id", s),
                            ("locality_type", s), ("lat", pa.float64()), ("lon", pa.float64()), ("date", s),
                            ("year", pa.int16()), ("doy", pa.int16()), ("time_started", s), ("protocol", s),
                            ("duration_min", pa.float32()), ("distance_km", pa.float32()), ("n_observers", pa.int16())])
    det_schema = pa.schema([("checklist_id", s), ("species_code", s), ("count", pa.int32())])
    os.makedirs(out, exist_ok=True)
    with _open(ebd) as fh:
        header = fh.readline().rstrip("\n").rstrip("\r").split("\t")
    cols = resolve_columns(header)
    use = sorted(set(cols.values()))
    reader = pd.read_csv(ebd, sep="\t", usecols=use, dtype=str, quoting=csv.QUOTE_NONE, chunksize=chunk_rows,
                         nrows=limit_rows, compression="infer", on_bad_lines="warn", na_filter=True)
    wc = wd = None
    n_rows = n_chk = n_det = 0
    t0 = time.time()
    seen = set()
    try:
        for df in reader:
            n_rows += len(df)
            chk, det = filter_chunk(df, cols, sci2code, years, doy, bbox, max_duration, max_distance, max_observers)
            if chk is not None:
                chk = chk[~chk["checklist_id"].isin(seen)]   # a checklist's rows can straddle a chunk boundary
                seen.update(chk["checklist_id"].tolist())
                tc = pa.Table.from_pandas(chk, schema=chk_schema, preserve_index=False)
                td = pa.Table.from_pandas(det.astype({"count": "int32"}), schema=det_schema, preserve_index=False)
                wc = wc or pq.ParquetWriter(os.path.join(out, "checklists_raw.parquet"), chk_schema)
                wd = wd or pq.ParquetWriter(os.path.join(out, "counts_raw.parquet"), det_schema)
                wc.write_table(tc)
                wd.write_table(td)
                n_chk += len(chk)
                n_det += len(det)
            print(f"[ebd] {n_rows:,} rows read, {n_chk:,} checklists, {n_det:,} detections kept "
                  f"({n_rows / max(time.time() - t0, 1e-9):,.0f} rows/s)", flush=True)
    except (EOFError, OSError, gzip.BadGzipFile) as e:   # a partially downloaded .gz ends mid-stream
        if limit_rows is None:
            raise
        print(f"[ebd] stream ended early ({e}); smoke mode keeps what was read", flush=True)
    finally:
        for w in (wc, wd):
            if w is not None:
                w.close()
    return finalize(out, dict(rows_read=n_rows, years=years, doy=doy, bbox=bbox, max_duration=max_duration,
                              max_distance=max_distance, max_observers=max_observers, source=os.path.abspath(ebd),
                              limit_rows=limit_rows, species_kept=sorted(set(sci2code.values()))))


def finalize(out, info):
    """Collapse shared (group) checklists to the lowest checklist id; drop their detections; write the final tables."""
    chk = pd.read_parquet(os.path.join(out, "checklists_raw.parquet"))
    det = pd.read_parquet(os.path.join(out, "counts_raw.parquet"))
    det = det.groupby(["checklist_id", "species_code"], as_index=False)["count"].max()   # straddled checklists
    grouped = chk["group_id"].astype(bool)
    group_keep = chk[grouped].sort_values("checklist_id").drop_duplicates("group_id")["checklist_id"]
    keep_ids = set(chk.loc[~grouped, "checklist_id"]) | set(group_keep)
    shared_dropped = int(grouped.sum()) - len(group_keep)
    chk = chk[chk["checklist_id"].isin(keep_ids)].drop(columns="group_id").reset_index(drop=True)
    det = det[det["checklist_id"].isin(keep_ids)].reset_index(drop=True)
    chk.to_parquet(os.path.join(out, "checklists.parquet"), index=False)
    det.to_parquet(os.path.join(out, "counts.parquet"), index=False)
    for f in ("checklists_raw.parquet", "counts_raw.parquet"):
        os.remove(os.path.join(out, f))
    info.update(checklists=len(chk), detections=len(det), shared_dropped=shared_dropped,
                x_counts=int((det["count"] < 0).sum()),
                years_present=sorted(int(y) for y in chk["year"].unique()))
    json.dump(info, open(os.path.join(out, "reduce_meta.json"), "w"), indent=1)
    print(f"[ebd] final: {len(chk):,} checklists, {len(det):,} detections "
          f"({info['x_counts']:,} 'X'), years {info['years_present'][:1]}..{info['years_present'][-1:]}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ebd", required=True, help="EBD .txt or .txt.gz (the relMMM-YYYY observations file)")
    ap.add_argument("--species-list", required=True, help="CSV with a species_code column (the codes to keep)")
    ap.add_argument("--exclude-list", default=None, help="text/CSV of codes to refuse (e.g. the sealed species)")
    ap.add_argument("--taxonomy", default=None, help="eBird taxonomy CSV (default {datasets_root}/avonet/...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=int, nargs=2, default=[2002, 2025])
    ap.add_argument("--doy", type=int, nargs=2, default=[145, 196], help="day-of-year window (default ~May 25-Jul 15)")
    ap.add_argument("--bbox", type=float, nargs=4, default=[-125.0, 24.0, -66.0, 53.0], help="lon0 lat0 lon1 lat1")
    ap.add_argument("--max-duration", type=float, default=300.0)
    ap.add_argument("--max-distance", type=float, default=5.0)
    ap.add_argument("--max-observers", type=int, default=10)
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    ap.add_argument("--limit-rows", type=int, default=None)
    a = ap.parse_args()
    if a.taxonomy is None:
        from src.config_utils import load_data_config
        a.taxonomy = os.path.join(load_data_config()["datasets_root"], "avonet", "eBird_taxonomy.csv")
    keep = pd.read_csv(a.species_list)["species_code"].astype(str).tolist()
    excl = []
    if a.exclude_list:
        excl = [l.strip().split(",")[0] for l in open(a.exclude_list) if l.strip() and not l.startswith("#")]
    m = sci_to_code(a.taxonomy, keep, excl)
    print(f"[ebd] {len(m)} of {len(keep)} requested codes mapped by scientific name "
          f"({len(set(keep) - set(m.values()) - {FOCAL})} unmapped); {len(excl)} excluded codes refused", flush=True)
    reduce(a.ebd, a.out, m, a.years, a.doy, a.bbox, a.max_duration, a.max_distance, a.max_observers,
           a.chunk_rows, a.limit_rows)


if __name__ == "__main__":
    main()
