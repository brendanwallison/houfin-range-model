"""Pure-core tests for the data layer's numerically-sensitive, cluster-free pieces.

Covers climate bio-year aggregation + gridding, the AOU-to-eBird crosswalk join, the
BBS community aggregation, and the spacetime validation metrics. Runs standalone or
under pytest, with no data tree.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.combine import climate_io as CIO
from src.data.identify import bbs_crosswalk as XW
from src.data.preprocess import bbs_community as BC


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


def test_climate_month_parsing():
    """Bases containing digits must parse — climr's degree-day names do.

    The original ``[A-Za-z]``-only base class silently dropped every DD variable
    before aggregation, despite ``_SUM_PREFIXES`` listing "dd" for them.
    """
    cols = ["id", "PERIOD", "DATASET"]
    bases = ["Tmax", "PPT", "DD_0", "DD5", "DD_18", "DD18", "NFFD", "Eref", "CMD"]
    for b in bases:
        cols += [f"{b}_{m:02d}" for m in range(1, 13)]
    groups = CIO.parse_month_columns(cols, warn=False)
    assert set(groups) == set(bases), set(bases) - set(groups)
    assert groups["DD_18"][12] == "DD_18_12"        # split at the LAST digit group
    assert groups["DD18"][1] == "DD18_01"
    # every DD spelling is treated as a flux (summed) by the legacy annual path
    assert all(CIO._is_sum_base(b) for b in ("DD_0", "DD5", "DD_18", "DD18"))

    # incomplete bases are dropped, but only complete ones are returned
    partial = CIO.parse_month_columns(["id", "Tmax01", "Tmax02"], warn=False)
    assert partial == {}
    # a bare numeric column is not a variable
    assert CIO.parse_month_columns(["id", "01"], warn=False) == {}
    # two spellings of the same (base, month) must not silently overwrite
    try:
        CIO.parse_month_columns(["id", "Tmax01", "Tmax_01"], warn=False)
        raise AssertionError("expected ValueError on duplicate (base, month)")
    except ValueError:
        pass
    # annual/seasonal columns are nameable rather than invisible
    assert CIO.annual_columns(cols + ["MAT"], month_groups=groups) == ["MAT"]
    print("climate month-column parsing OK")


def test_climate_bioyear_monthly():
    """The monthly path keeps values verbatim and in bio-year window order."""
    rows = []
    for pid in (10, 20):
        for yr in (2000, 2001):
            r = {"id": pid, "PERIOD": yr}
            for m in range(1, 13):
                r[f"Tmax{m:02d}"] = yr * 100 + m
                r[f"PPT_{m:02d}"] = float(m)
            rows.append(r)
    df = pd.DataFrame(rows)

    names = CIO.bioyear_month_columns("Tmax", 8)
    assert names[0] == "Tmax_b01m08" and names[-1] == "Tmax_b12m07"
    assert len(names) == 12 and len(set(names)) == 12
    # b01..b12 sorts lexicographically into window order -- discover_variables
    # relies on this, so channel index == months since the window start.
    assert names == sorted(names)

    mon = CIO.bioyear_monthly(df, 2001, start_month=8)
    assert mon.shape == (2, 24)
    # FIDELITY: b01m08 is August of T-1; b12m07 is July of T. No aggregation.
    assert mon.loc[10, "Tmax_b01m08"] == 2000 * 100 + 8
    assert mon.loc[10, "Tmax_b12m07"] == 2001 * 100 + 7
    assert mon.loc[10, "PPT_b06m01"] == 1.0        # Jan of T sits at position 6

    # REGRESSION: the legacy annual value is exactly the mean/sum of these
    # columns, proving the shared ``_bioyear_frame`` refactor changed nothing.
    agg = CIO.bioyear_aggregate(df, 2001, start_month=8)
    tmax_cols = CIO.bioyear_month_columns("Tmax", 8)
    ppt_cols = CIO.bioyear_month_columns("PPT", 8)
    assert np.allclose(agg["Tmax"], mon[tmax_cols].mean(axis=1))
    assert np.allclose(agg["PPT"], mon[ppt_cols].sum(axis=1))

    # gap-straddling year -> empty on BOTH paths, with columns still declared
    empty = CIO.bioyear_monthly(df, 2000, start_month=8)
    assert len(empty) == 0 and len(empty.columns) == 24
    print("climate bio-year monthly OK")


def test_climate_grid_levels():
    """q10/q90 are temperature-only; q50 carries every base."""
    from src.data.preprocess import climate_grid as CG

    all_lv = ["q10", "q50", "q90"]
    for temp in ("Tmax", "Tmin", "Tave", "tave"):
        assert CG.levels_for_base(temp, all_lv) == all_lv
    for flux in ("PPT", "DD_18", "NFFD", "CMD"):
        assert CG.levels_for_base(flux, all_lv) == ["q50"]
    # an explicit --levels subset still constrains temperatures
    assert CG.levels_for_base("Tmax", ["q50"]) == ["q50"]
    assert CG.levels_for_base("PPT", ["q10", "q90"]) == []
    # the v2 token guard must accept new tokens and reject v1 annual ones
    assert CG._V2_TOKEN.search("Tmax_b01m08_q50")
    assert not CG._V2_TOKEN.search("Tmax_q50")
    print("climate grid level assignment OK")


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


def test_validate_metrics():
    from src.community_encoder.train_DESK import validate_spacetime as V
    X = np.array([[1., 2, 3], [2., 4, 6], [0., 1, 0]])
    S = V.ruzicka_similarity_matrix(X)
    assert np.allclose(np.diag(S), 1.0) and abs(S[0, 1] - 0.5) < 1e-9  # x vs 2x
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((30, 5)); K = Z @ Z.T
    assert abs(V.linear_cka(K, K) - 1.0) < 1e-9
    assert abs(V.linear_cka(K, (2 * Z) @ (2 * Z).T) - 1.0) < 1e-9      # scale-invariant
    Kr = rng.standard_normal((30, 30)); Kr = Kr @ Kr.T
    assert V.linear_cka(K, Kr) < 0.9 and abs(V.mantel_r(K, K) - 1.0) < 1e-9
    print("validate metrics OK")


def test_temporal_metrics():
    """Turnover-magnitude agreement + spatiotemporal analog-direction (both basis-invariant)."""
    from src.community_encoder.train_DESK import validate_spacetime as V
    rng = np.random.default_rng(0)
    rec = 2020

    # ruzicka_rect matches the elementwise definition
    A = np.array([[1., 2, 0], [0, 1, 1]]); B = np.array([[1., 1, 1], [2, 0, 0]])
    man = np.array([[np.minimum(A[i], B[j]).sum() / np.maximum(A[i], B[j]).sum()
                     for j in range(2)] for i in range(2)])
    assert np.allclose(V.ruzicka_rect(A, B), man, atol=1e-6)

    # turnover: hist blends further from recent per site -> monotone turnover, both agree
    base, other = np.eye(8)[0], np.eye(8)[4]
    Xs, pidx = [], []
    for k in range(5):
        f = k / 4.0
        Xs += [base.copy(), (1 - f) * base + f * other]
        pidx += [[k, 0, rec], [k, 0, 1990]]
    Xs = np.array(Xs); t = V.temporal_turnover_agreement(Xs.copy(), Xs, np.array(pidx), rec)
    order = [(int(r), int(c)) for r, c in zip(t["rows"], t["cols"])]
    tp = t["turnover_pred"]
    assert t["spearman_turnover"] > 0.9
    assert tp[order.index((0, 0))] == min(tp) and tp[order.index((4, 0))] == max(tp)

    # analog: hist community matches a present cell due NORTH -> displacement +y, models agree
    south, north = np.eye(6)[0], np.eye(6)[3]
    X2 = np.array([south, north] + [north] * 4)
    pidx2 = np.array([[0, 0, rec], [0, 1, rec], [0, 0, 1980], [1, 0, 1980], [2, 0, 1980], [3, 0, 1980]])
    xy = np.array([[0., 0.], [0., 100.], [0., 0.], [10., 0.], [20., 0.], [30., 0.]])
    a = V.analog_displacement(X2.copy(), X2, pidx2, xy, rec, rng, n_hist=10, n_present=10, topk=1)
    assert (a["d_obs"][:, 1] > 50).all() and a["mean_cos_displacement"] > 0.99
    print("temporal metrics (turnover + analog direction) OK")


if __name__ == "__main__":
    test_climate_bioyear_and_grid()
    test_climate_month_parsing()
    test_climate_bioyear_monthly()
    test_climate_grid_levels()
    test_crosswalk_core()
    test_bbs_community_aggregation()
    test_spacetime_numerics()
    test_validate_metrics()
    test_temporal_metrics()
    print("\nALL BBS-SPACETIME CHECKS PASSED")


def test_mexico_stop_columns_absorb_the_usgs_header_typo():
    """The real USGS Mexico release ships stop 46 as 'Sto 46' -- the 'p' replaced by a
    space. A startswith('stop') test drops it and silently undercounts that route by one
    stop, which is invisible: the sum is still a plausible number."""
    import pandas as pd
    from src.data.preprocess.bbs import _mexico_count, _mexico_stop_columns

    cols = ([f"Stop{i}" for i in range(1, 46)] + ["Sto 46"]
            + [f"Stop{i}" for i in range(47, 51)])            # the real header, verbatim
    df = pd.DataFrame([[1] * 50], columns=cols)
    stops = _mexico_stop_columns(df)
    assert set(stops) == set(range(1, 51)), "did not recover all 50 stops"
    assert stops[46] == "Sto 46"
    assert int(_mexico_count(df).iloc[0]) == 50               # not 49


def test_mexico_stop_columns_refuse_a_gap_rather_than_undercount():
    """An unrecognised header mangling must stop the ingest, not contribute a zero."""
    import pandas as pd
    from src.data.preprocess.bbs import _mexico_stop_columns

    cols = [f"Stop{i}" for i in range(1, 46)] + ["S46"] + [f"Stop{i}" for i in range(47, 51)]
    try:
        _mexico_stop_columns(pd.DataFrame([[0] * 50], columns=cols))
    except ValueError as exc:
        assert "46" in str(exc) and "undercount" in str(exc)
    else:
        raise AssertionError("a gap in the stop numbering was silently accepted")


def test_mexico_count_prefers_speciestotal_when_present():
    import pandas as pd
    from src.data.preprocess.bbs import _mexico_count
    df = pd.DataFrame({"SpeciesTotal": [7], "Stop1": [1], "Stop2": [2]})
    assert int(_mexico_count(df).iloc[0]) == 7


# ----------------------------- observer first-year flag -----------------------------

def _runs(rows):
    """rows = [(route, obsn, year), ...] -> the frame first_year_flags expects."""
    import pandas as pd
    return pd.DataFrame([{"CountryNum": 840, "StateNum": 2, "Route": r, "Year": y, "ObsN": o}
                         for r, o, y in rows])


def test_first_year_is_grouped_by_observer_AND_route():
    """bbsBayes2 groups on obs_route, not observer. An experienced observer moving to a new
    route is flagged AGAIN, because the effect is unfamiliarity with that route -- not
    inexperience. Grouping by observer alone would miss every route change."""
    from src.data.preprocess.bbs import first_year_flags
    # observer 7 runs route 1 from 1990, then also starts route 2 in 1995
    out = first_year_flags(_runs([(1, 7, 1990), (1, 7, 1991), (1, 7, 1992),
                                  (2, 7, 1995), (2, 7, 1996)]))
    got = {(int(r.Route), int(r.Year)): int(r.first_year) for r in out.itertuples()}
    assert got[(1, 1990)] == 1                       # first year on route 1
    assert got[(1, 1991)] == 0 and got[(1, 1992)] == 0
    assert got[(2, 1995)] == 1, "route change must re-flag an experienced observer"
    assert got[(2, 1996)] == 0


def test_first_year_flags_each_observer_independently_on_a_shared_route():
    """Two observers on one route: each gets their own first year, and a later observer's
    debut is flagged even though the route itself has been surveyed for years."""
    from src.data.preprocess.bbs import first_year_flags
    out = first_year_flags(_runs([(1, 7, 1990), (1, 7, 1991), (1, 9, 2005), (1, 9, 2006)]))
    got = {(int(r.ObsN), int(r.Year)): int(r.first_year) for r in out.itertuples()}
    assert got[(7, 1990)] == 1 and got[(7, 1991)] == 0
    assert got[(9, 2005)] == 1 and got[(9, 2006)] == 0


def test_first_year_flags_is_pure_and_total():
    """Does not mutate its input, and flags exactly one year per (route, observer)."""
    from src.data.preprocess.bbs import first_year_flags
    runs = _runs([(1, 7, 1990), (1, 7, 1991), (2, 8, 1975)])
    before = runs.copy()
    out = first_year_flags(runs)
    assert "first_year" not in runs.columns and runs.equals(before)
    n_groups = len({(int(r.Route), int(r.ObsN)) for r in out.itertuples()})
    assert int(out["first_year"].sum()) == n_groups
    assert set(out["first_year"].unique()) <= {0, 1}
