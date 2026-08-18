"""Per-axis community selection: the rank penalty, the admission bars, and the
target solve. Pure numpy -- no pandas, no data files, no network.

The property most worth pinning is the one that is easiest to get wrong by
intuition: the cut is on key VALUE, so the threshold key exceeds N and RISES with
the penalty, and a migrant with a good enough raw rank is admitted anyway.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.identify.community_axes import (
    AXES, MIGRATORY, PARTIAL, SEDENTARY, admission_bars, composition,
    interleaved_order, migration_penalty, rank_average, select_axis,
    select_union, solve_penalty,
)


def _linear_pool(n=12, migratory=(0, 1, 2)):
    """``n`` species; ``phylo_distance`` = 1..n so raw rank == position.

    The indices in ``migratory`` are class 3, the rest class 1 -- so the penalty's
    effect is exactly predictable by hand.
    """
    codes = [f"sp{i:02d}" for i in range(n)]
    mig = np.ones(n, dtype=int)
    for i in migratory:
        mig[i] = MIGRATORY
    return {
        "species_code": codes,
        "migration": mig,
        "phylo_distance": np.arange(1.0, n + 1.0),
        "trait_distance": np.arange(1.0, n + 1.0),
        "urban_tolerance": np.arange(1.0, n + 1.0),
    }


# ---- the ranking primitive

def test_rank_average_ties_and_direction():
    assert np.allclose(rank_average([10, 20, 20, 30]), [1, 2.5, 2.5, 4])
    assert np.allclose(rank_average([10, 20, 20, 30], ascending=False), [4, 2.5, 2.5, 1])
    assert np.allclose(rank_average([5.0]), [1.0])


def test_rank_average_refuses_nonfinite():
    """A NaN ranked silently last would look like a legitimate worst-place finish."""
    for bad in ([1.0, np.nan, 3.0], [1.0, np.inf]):
        try:
            rank_average(bad)
        except ValueError:
            continue
        raise AssertionError(f"non-finite accepted: {bad}")


def test_migration_penalty_is_linear_and_zero_for_sedentary():
    p = migration_penalty([SEDENTARY, PARTIAL, MIGRATORY], 30)
    assert np.allclose(p, [0.0, 30.0, 60.0])
    assert np.allclose(migration_penalty([1, 2, 3], 0), 0.0)
    try:
        migration_penalty([0, 1], 10)
    except ValueError:
        pass
    else:
        raise AssertionError("class 0 accepted")


# ---- the mechanism: the threshold key is NOT N

def test_threshold_key_exceeds_n_and_admits_a_strong_migrant():
    """Hand-checkable: 12 species, raw ranks 1..12, top 3, p=2.

    Migrants sp00/sp01/sp02 (raw 1,2,3) pay +4, so keys are 5,6,7. The sedentary
    keys are just their raw ranks. Sorted: sp03=4, then key 5 twice (sp04 with no
    penalty, then sp00), so the top 3 is sp03, sp04, sp00 -- threshold key 5, well
    above N=3, and the migrant at raw rank 1 IS admitted.
    """
    pool = _linear_pool()
    sel = select_axis(pool["species_code"], pool["phylo_distance"],
                      pool["migration"], True, 3, 2)
    assert sel["codes"] == ["sp03", "sp04", "sp00"], sel["codes"]
    assert sel["threshold_key"] == 5.0
    assert sel["threshold_key"] > sel["n_requested"]          # the whole point
    assert sel["bars"] == {SEDENTARY: 5.0, PARTIAL: 3.0, MIGRATORY: 1.0}
    # the admitted migrant's raw rank is exactly at its bar
    assert sel["raw_rank"][0] == 1.0 and sel["penalty"][0] == 4.0


def test_threshold_rises_and_sedentary_bar_only_loosens():
    """Raising p can never exclude a sedentary species: class 1 is unpenalised, so
    its bar is the threshold, and the threshold is non-decreasing in p."""
    pool = _linear_pool()
    thresholds, sed_bars, n_migr = [], [], []
    for p in (0, 1, 2, 3, 4, 8):
        sel = select_axis(pool["species_code"], pool["phylo_distance"],
                          pool["migration"], True, 3, p)
        thresholds.append(sel["threshold_key"])
        sed_bars.append(sel["bars"][SEDENTARY])
        picked = {c: i for i, c in enumerate(pool["species_code"])}
        n_migr.append(sum(pool["migration"][picked[c]] == MIGRATORY for c in sel["codes"]))
    assert thresholds == sorted(thresholds), thresholds
    assert sed_bars == thresholds                              # bar == threshold for class 1
    assert n_migr[0] >= n_migr[-1] and n_migr[-1] == 0
    # p=0 is the plain top-N: threshold is exactly N
    assert thresholds[0] == 3.0


def test_admission_bars_are_the_selection_rule():
    """The bars are not decoration: raw_rank <= bar must reproduce the selection
    exactly (ties at the threshold aside, which the tie-break decides)."""
    pool = _linear_pool(n=20, migratory=(0, 1, 5, 9))
    pool["migration"][3] = PARTIAL
    sel = select_axis(pool["species_code"], pool["phylo_distance"],
                      pool["migration"], True, 6, 3)
    chosen = set(sel["codes"])
    for i, code in enumerate(pool["species_code"]):
        bar = sel["bars"][int(pool["migration"][i])]
        if sel["raw_rank"][i] < bar:                           # strictly inside: must be in
            assert code in chosen, (code, sel["raw_rank"][i], bar)
        elif sel["raw_rank"][i] > bar:                         # strictly outside: must be out
            assert code not in chosen, (code, sel["raw_rank"][i], bar)


def test_bars_go_nonpositive_when_a_class_cannot_place():
    assert admission_bars(30, 60)[MIGRATORY] == -90.0
    assert admission_bars(30, 0) == {SEDENTARY: 30.0, PARTIAL: 30.0, MIGRATORY: 30.0}


# ---- commensurability across axes

def test_penalty_acts_on_the_rank_not_the_value():
    """The urban axes arrive as signed values, the distance axes as distances, so
    the penalty has to be added to the RANK or one p means four different things.
    Equivalent statement: selection is invariant to any increasing rescale of the
    axis column. If the penalty hit the value, a x1000 rescale would swamp it."""
    pool = _linear_pool()
    base = select_axis(pool["species_code"], pool["phylo_distance"],
                       pool["migration"], True, 4, 2)
    for scaled in (pool["phylo_distance"] * 1000.0 + 7.0,
                   np.log(pool["phylo_distance"]),
                   pool["phylo_distance"] * 1e-6):
        s = select_axis(pool["species_code"], scaled, pool["migration"], True, 4, 2)
        assert s["codes"] == base["codes"], (s["codes"], base["codes"])
        assert s["threshold_key"] == base["threshold_key"]


def test_descending_axis_reverses_the_ranking():
    pool = _linear_pool(migratory=())
    up = select_axis(pool["species_code"], pool["urban_tolerance"],
                     pool["migration"], True, 3, 0)
    down = select_axis(pool["species_code"], pool["urban_tolerance"],
                       pool["migration"], False, 3, 0)
    assert up["codes"] == ["sp00", "sp01", "sp02"]
    assert down["codes"] == ["sp11", "sp10", "sp09"]


# ---- the union

def test_union_is_exactly_the_union_of_per_axis_top_n():
    pool = _linear_pool(n=40, migratory=())
    rng = np.random.default_rng(0)
    pool["trait_distance"] = rng.permutation(40).astype(float)
    pool["urban_tolerance"] = rng.permutation(40).astype(float)
    sel = select_union(pool, 5, 0)
    for name, a in sel["axes"].items():
        assert len(a["codes"]) == 5, (name, a["codes"])
    expected = set().union(*(set(a["codes"]) for a in sel["axes"].values()))
    assert set(sel["codes"]) == expected
    assert len(sel["codes"]) == len(set(sel["codes"]))          # no duplicates
    assert sel["composition"]["n"] == len(sel["codes"])
    # every selected species records which axes chose it
    for code, cats in sel["category"].items():
        assert cats and all(c in {a["name"] for a in AXES} for c in cats)
        for c in cats:
            assert code in sel["axes"][c]["codes"]


def test_urban_axes_split_the_two_tails_disjointly():
    pool = _linear_pool(n=20, migratory=())
    sel = select_union(pool, 4, 0)
    loving = set(sel["axes"]["urban_loving"]["codes"])
    avoiding = set(sel["axes"]["urban_avoiding"]["codes"])
    assert not (loving & avoiding)
    med = float(np.median(pool["urban_tolerance"]))
    idx = {c: i for i, c in enumerate(pool["species_code"])}
    assert all(pool["urban_tolerance"][idx[c]] > med for c in loving)
    assert all(pool["urban_tolerance"][idx[c]] < med for c in avoiding)


def test_axis_shortfall_is_reported_not_hidden():
    """A pool smaller than N yields fewer codes, with n_requested/n_available kept
    so a gate that starved an axis shows up instead of silently rebalancing."""
    pool = _linear_pool(n=3, migratory=())
    sel = select_axis(pool["species_code"], pool["phylo_distance"],
                      pool["migration"], True, 10, 0)
    assert len(sel["codes"]) == 3
    assert sel["n_requested"] == 10 and sel["n_available"] == 3


def test_interleaved_order_is_round_robin_and_deterministic():
    axis_codes = {"phylo": ["a", "b", "c"], "morph": ["b", "d"],
                  "urban_loving": ["e"], "urban_avoiding": ["f", "a"]}
    order = interleaved_order(axis_codes, AXES)
    assert order == ["a", "b", "e", "f", "c", "d"], order
    assert order == interleaved_order(axis_codes, AXES)         # stable
    assert sorted(order) == sorted({c for v in axis_codes.values() for c in v})


# ---- solving for a target composition

def _mixed_pool(n=200, seed=0):
    rng = np.random.default_rng(seed)
    codes = [f"s{i:03d}" for i in range(n)]
    mig = rng.choice([SEDENTARY, PARTIAL, MIGRATORY], size=n, p=[0.28, 0.19, 0.53])
    # migrants nearer the focal on morphology, as measured in the real table
    trait = rng.normal(0, 1, n) - 0.4 * (mig == MIGRATORY)
    return {"species_code": codes, "migration": mig,
            "phylo_distance": rng.normal(0, 1, n),
            "trait_distance": trait,
            "urban_tolerance": rng.normal(0, 1, n)}


def test_solve_penalty_returns_the_smallest_qualifying_p():
    pool = _mixed_pool()
    sel, trace = solve_penalty(pool, 15, 0.25)
    assert sel["target_met"] is True
    assert sel["composition"]["fracs"][MIGRATORY] <= 0.25
    p = int(sel["penalty"])
    assert trace[-1]["p"] == p
    # every smaller p missed the target -- so p really is the smallest that works
    assert all(t["migratory_frac"] > 0.25 for t in trace[:-1])
    for q in range(p):
        assert select_union(pool, 15, q)["composition"]["fracs"][MIGRATORY] > 0.25


def test_solve_penalty_reports_failure_instead_of_pretending():
    pool = _mixed_pool()
    sel, trace = solve_penalty(pool, 15, -1.0, p_max=5)         # unreachable target
    assert sel["target_met"] is False
    assert sel["penalty"] == 5.0 and len(trace) == 6


def test_p_zero_is_the_plain_top_n_per_axis():
    pool = _mixed_pool()
    sel = select_union(pool, 10, 0)
    for a in sel["axes"].values():
        assert a["threshold_key"] == 10.0                       # no penalty -> cut at N
        assert a["bars"][MIGRATORY] == 10.0


def test_composition_handles_the_empty_set():
    c = composition([])
    assert c["n"] == 0 and c["fracs"][MIGRATORY] == 0.0


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except Exception:
            fails += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    print("\n" + (f"{fails} FAILED" if fails else "ALL COMMUNITY-AXIS CHECKS PASSED"))
    sys.exit(1 if fails else 0)
