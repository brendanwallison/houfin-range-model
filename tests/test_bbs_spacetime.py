"""Tests for the BBS spatiotemporal-community pipeline pure cores.

Covers the numerically-sensitive, cluster-free pieces: climate bio-year
aggregation + gridding (A1), the AOU↔eBird crosswalk join (B1), the BBS
community aggregation (B2), and the spatiotemporal smoother / robust anomaly /
amplitude modulation (B3). Runs standalone or under pytest.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.combine import climate_io as CIO
from src.data.identify import bbs_crosswalk as XW
from src.data.preprocess import bbs_community as BC
from src.community_encoder.train_DESK import spacetime_community as SC


def test_climate_bioyear_and_grid():
    rows = []
    for pid in (10, 20):
        for yr in (2000, 2001):
            r = {"id": pid, "PERIOD": yr}
            for m in range(1, 13):
                r[f"Tmax{m:02d}"] = yr * 100 + m
                r[f"PPT_{m:02d}"] = 1.0
            rows.append(r)
    df = pd.DataFrame(rows)
    assert set(CIO.parse_month_columns(df.columns)) == {"Tmax", "PPT"}
    assert CIO._is_sum_base("PPT") and not CIO._is_sum_base("Tmax")
    agg = CIO.bioyear_aggregate(df, 2001, start_month=8)
    vals = [2000 * 100 + m for m in range(8, 13)] + [2001 * 100 + m for m in range(1, 8)]
    assert np.allclose(agg["PPT"], 12.0)               # 12 months summed
    assert np.allclose(agg["Tmax"], np.mean(vals))     # intensive -> mean
    assert len(CIO.bioyear_aggregate(df, 2000, start_month=8)) == 0  # straddles missing year
    cen = pd.DataFrame({"id": [10, 20], "row": [0, 1], "col": [1, 0]})
    grid = CIO.grid_from_centroids(agg["Tmax"], cen, 2, 2)
    assert np.isnan(grid[0, 0]) and not np.isnan(grid[1, 0])
    print("climate bio-year + grid OK")


def test_crosswalk_core():
    tax = pd.DataFrame({"SPECIES_CODE": ["houfin", "amegfi", "xxxxxx"],
                        "SCIENTIFIC_NAME": ["Haemorhous mexicanus", "Spinus tristis", "Foo bar"]})
    ebird = tax.rename(columns={"SCIENTIFIC_NAME": "SCI_NAME"})
    ebird["species_code"] = ebird["SPECIES_CODE"].str.lower()
    ebird["sci_norm"] = ebird["SCI_NAME"].apply(XW.normalize_name)
    ebird = ebird[["species_code", "sci_norm"]]
    bbs_df = pd.DataFrame({"AOU": [5190, 9999, 4200, 1],
                           "Genus": ["Haemorhous", "Haemorhous", "Spinus", "No"],
                           "Species": ["mexicanus", "mexicanus", "tristis", "match"]})
    bnorm = XW.normalize_bbs_species(bbs_df)
    matched, diag = XW.crosswalk(bnorm, ebird, ["houfin", "amegfi", "casfin"])
    assert diag["n_matched"] == 2 and diag["n_community"] == 3
    assert (matched.species_code == "houfin").sum() == 2   # lump preserved
    assert 1 not in set(matched.aou)                        # non-community dropped
    assert diag["split_aous"] == []
    print("crosswalk core OK")


def test_bbs_community_aggregation():
    obs = pd.DataFrame({
        "CountryNum": [840] * 4, "StateNum": [1] * 4, "Route": [1, 1, 2, 2],
        "Year": [2000] * 4, "AOU": [10, 11, 10, 10], "SpeciesTotal": [3, 5, 7, 0]})
    cx = pd.DataFrame({"aou": [10, 11], "species_code": ["spA", "spA"]})
    cov = pd.DataFrame({"CountryNum": [840, 840], "StateNum": [1, 1],
                        "Route": [1, 2], "Year": [2000, 2000]})
    rc = pd.DataFrame({"CountryNum": [840, 840], "StateNum": [1, 1],
                       "Route": [1, 2], "row": [0, 0], "col": [0, 0]})
    mean_df, cov_df = BC.build_community_matrix(obs, cov, cx, rc)
    assert int(cov_df.iloc[0]["n_routes"]) == 2
    # cell (0,0) 2000: (3+5)+(7+0) = 15 summed / 2 covered route-years = 7.5
    assert abs(float(mean_df.iloc[0]["mean_count"]) - 7.5) < 1e-6
    print("BBS community aggregation OK")


def test_spacetime_numerics():
    T, H, W = 10, 5, 5
    years = list(range(1990, 2000))
    mean = np.ones((T, H, W))
    for t, y in enumerate(years):
        if y >= 1997:
            mean[t, 2, 2] = 3.0
    eff = np.ones((T, H, W))
    field, support = SC.gaussian_nw_field(mean * eff, eff, 0.4, 0.4, 1e-3)
    ref = SC.reference_field(field, years, [1990, 1991, 1992])
    anom = SC.robust_anomaly(field, support, ref, pseudocount=0.5,
                             cap=np.log(10), support_floor=1e-3, shrink_k=1.0)
    i98 = years.index(1998)
    assert anom[i98, 2, 2] > 1.3 and anom[i98, 2, 2] > anom[i98, 0, 0] + 0.3
    assert abs(anom[i98, 0, 0] - 1.0) < 0.25
    # zero support -> exactly no-change
    eff2 = eff.copy(); eff2[:, 0, 0] = 0.0
    f2, s2 = SC.gaussian_nw_field(mean * eff2, eff2, 0.01, 0.01, 1e-3)
    a2 = SC.robust_anomaly(f2, s2, SC.reference_field(f2, years, [1990, 1991, 1992]),
                           pseudocount=0.5, support_floor=1e-3)
    assert abs(a2[i98, 0, 0] - 1.0) < 1e-9
    # cap
    m3 = np.ones((T, H, W))
    for t, y in enumerate(years):
        if y >= 1997:
            m3[t, 2, 2] = 1000.0
    f3, s3 = SC.gaussian_nw_field(m3, eff, 0.01, 0.01, 1e-3)
    a3 = SC.robust_anomaly(f3, s3, SC.reference_field(f3, years, [1990, 1991, 1992]),
                           pseudocount=0.5, cap=np.log(10), support_floor=1e-3, shrink_k=0.0)
    assert a3[i98, 2, 2] <= 10 + 1e-6 and a3[i98, 2, 2] > 9
    # amplitude modulation, species-blocked
    x = SC.apply_amplitude(np.arange(12.0), [2.0, 1.0, 0.0], 4)
    assert np.allclose(x[0:4], np.arange(4) * 2) and np.allclose(x[8:12], 0)
    print("spacetime numerics OK")


if __name__ == "__main__":
    test_climate_bioyear_and_grid()
    test_crosswalk_core()
    test_bbs_community_aggregation()
    test_spacetime_numerics()
    print("\nALL BBS-SPACETIME CHECKS PASSED")
