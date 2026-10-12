"""ebd_reduce: filters, species guards, X counts, shared-checklist collapse, chunk straddling."""
import json
import os

import pandas as pd
import pytest

from src.data.preprocess import ebd_reduce as er

HEADER = ["GLOBAL UNIQUE IDENTIFIER", "CATEGORY", "COMMON NAME", "SCIENTIFIC NAME", "OBSERVATION COUNT", "LATITUDE",
          "LONGITUDE", "OBSERVATION DATE", "TIME OBSERVATIONS STARTED", "OBSERVER ID", "SAMPLING EVENT IDENTIFIER",
          "LOCALITY ID", "LOCALITY TYPE", "PROTOCOL NAME", "DURATION MINUTES", "EFFORT DISTANCE KM",
          "NUMBER OBSERVERS", "ALL SPECIES REPORTED", "GROUP IDENTIFIER", "APPROVED"]
TAX = pd.DataFrame({"SCIENTIFIC_NAME": ["Turdus migratorius", "Haemorhous mexicanus", "Spizella passerina",
                                        "Cardinalis cardinalis"],
                    "SPECIES_CODE": ["amerob", "houfin", "chispa", "norcar"],
                    "CATEGORY": ["species"] * 4})


def _row(cid, sci, count="2", cat="species", date="2015-06-10", lat="40.0", lon="-90.0", proto="Stationary",
         dur="30", dist="", nobs="1", complete="1", group="", approved="1"):
    return ["URN:x", cat, "n", sci, count, lat, lon, date, "07:00:00", "obsr1", cid, "L1", "H", proto, dur, dist,
            nobs, complete, group, approved]


def _write(tmp_path, rows):
    p = tmp_path / "ebd.txt"
    with open(p, "w", newline="") as fh:
        fh.write("\t".join(HEADER) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    return str(p)


def _run(tmp_path, rows, keep=("amerob", "houfin", "chispa", "norcar"), exclude=(), chunk_rows=1000):
    tax = tmp_path / "tax.csv"
    TAX.to_csv(tax, index=False)
    m = er.sci_to_code(str(tax), keep, exclude)
    out = tmp_path / "out"
    er.reduce(_write(tmp_path, rows), str(out), m, (2002, 2025), (145, 196), (-125, 24, -66, 53), 300, 5, 10,
              chunk_rows=chunk_rows)
    return (pd.read_parquet(out / "checklists.parquet"), pd.read_parquet(out / "counts.parquet"),
            json.load(open(out / "reduce_meta.json")))


def test_filters_and_counts(tmp_path):
    rows = [
        _row("S1", "Turdus migratorius", "3"), _row("S1", "Spizella passerina", "X"),
        _row("S2", "Turdus migratorius", complete="0"),                   # incomplete
        _row("S3", "Turdus migratorius", proto="Incidental"),              # protocol
        _row("S4", "Turdus migratorius", dur="400"),                       # too long
        _row("S5", "Turdus migratorius", proto="Traveling", dist="9"),     # too far
        _row("S6", "Turdus migratorius", date="2015-01-10"),               # out of season
        _row("S7", "Turdus migratorius", lon="10.0"),                      # out of box
        _row("S8", "Turdus migratorius", approved="0"),                    # unapproved
        _row("S9", "Cardinalis cardinalis", proto="Traveling", dist="2"),  # kept
        _row("S10", "Strix varia", "1"),                                   # kept checklist, unrequested species
    ]
    chk, det, meta = _run(tmp_path, rows)
    assert sorted(chk["checklist_id"]) == ["S1", "S10", "S9"]
    d = {(r.checklist_id, r.species_code): r.count for r in det.itertuples()}
    assert d == {("S1", "amerob"): 3, ("S1", "chispa"): -1, ("S9", "norcar"): 2}
    assert meta["x_counts"] == 1


def test_house_finch_and_excluded_are_refused(tmp_path):
    rows = [_row("S1", "Haemorhous mexicanus", "5"), _row("S1", "Spizella passerina", "1"),
            _row("S1", "Turdus migratorius", "1")]
    chk, det, _ = _run(tmp_path, rows, exclude=("chispa",))
    assert set(det["species_code"]) == {"amerob"}
    assert "houfin" not in set(det["species_code"])
    assert list(chk["checklist_id"]) == ["S1"]   # the checklist stays: its zeros for kept species are real


def test_shared_checklists_collapse_to_lowest_id(tmp_path):
    rows = [_row("S20", "Turdus migratorius", "2", group="G1"), _row("S21", "Turdus migratorius", "4", group="G1"),
            _row("S30", "Turdus migratorius", "1")]
    chk, det, meta = _run(tmp_path, rows)
    assert sorted(chk["checklist_id"]) == ["S20", "S30"]
    assert set(det["checklist_id"]) == {"S20", "S30"}
    assert meta["shared_dropped"] == 1


def test_issf_sums_with_species_and_chunk_straddle(tmp_path):
    rows = [_row("S1", "Turdus migratorius", "2"), _row("S1", "Turdus migratorius", "1", cat="issf"),
            _row("S1", "Spizella passerina", "1"), _row("S2", "Turdus migratorius", "1")]
    chk, det, _ = _run(tmp_path, rows, chunk_rows=1)   # every row its own chunk: S1 straddles three
    assert sorted(chk["checklist_id"]) == ["S1", "S2"]
    d = {(r.checklist_id, r.species_code): r.count for r in det.itertuples()}
    assert d[("S1", "amerob")] >= 2 and d[("S1", "chispa")] == 1 and d[("S2", "amerob")] == 1


def test_header_must_resolve():
    with pytest.raises(SystemExit):
        er.resolve_columns(["SAMPLING EVENT IDENTIFIER"])
    cols = er.resolve_columns(HEADER)
    assert cols["protocol"] == "PROTOCOL NAME"
