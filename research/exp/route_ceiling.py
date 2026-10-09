"""O3 null check: does route turnover between epochs masquerade as real change in every ceiling?

    python research/exp/route_ceiling.py --cache <dir> --out <dir>
    python research/exp/route_ceiling.py --summarize <out>

The ABBA split-half splits YEARS within a cell-epoch, not ROUTES. A cell surveyed by routes {1, 2} early and
{3} late differs between epochs partly because different stretches of land were counted, and both halves of
each epoch inherit that -- so it counts as signal, inflating ``ms_obs - noise`` (the "available change") that
every captured share and ceiling divides by. Compare the split-half change budget on cells surveyed by the
SAME route set in both epochs against all gated cells. If real change shrinks sharply on the same-route
subset, part of every ceiling is route turnover, and DESK is closer to the reachable ceiling than reported.

The same-route subset is long-running routes, a biased sample of places; the report gives its region mix
and a size-matched random subset of all cells so the comparison is not just a different population.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def route_sets(route_years, cells_keys, early, modern):
    """``{(row, col): (frozenset early routes, frozenset modern routes)}`` from the route-year table."""
    out = {}
    rid = (route_years["country"].astype(str) + "-" + route_years["state"].astype(str) + "-"
           + route_years["route"].astype(str))
    for (r, c, y), g in zip(route_years[["row", "col", "year"]].itertuples(index=False), rid):
        e = 0 if early[0] <= y <= early[1] else (1 if modern[0] <= y <= modern[1] else None)
        if e is None:
            continue
        sets = out.setdefault((int(r), int(c)), (set(), set()))
        sets[e].add(g)
    return {k: (frozenset(a), frozenset(b)) for k, (a, b) in out.items()}


def budget(d_full, d_a, d_b):
    from src.community_encoder.train_DESK.validation_core import change_noise
    nz = change_noise(d_full, d_a, d_b)
    ok = nz["ms_obs"] > 0
    return {"n_cells": int(d_full.shape[0]), "n_resolvable": int(nz["resolvable"].sum()),
            "pooled_signal_share": (float(1 - nz["noise"][ok].sum() / nz["ms_obs"][ok].sum())
                                    if ok.any() else None),
            "median_signal_share_resolvable": (float(np.median(nz["signal_share"][nz["resolvable"]]))
                                               if nz["resolvable"].any() else None),
            "mean_real_change_per_cell": float(np.mean(nz["ms_obs"] - nz["noise"]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    from src.community_encoder.train_DESK.validate_bbs_routes import (EPOCH_EARLY, EPOCH_MODERN,
                                                                       MIN_EPOCH_YEARS, epoch_gate)
    from src.community_encoder.train_DESK.validation_core import split_half_change
    os.makedirs(a.out, exist_ok=True)
    keys = np.load(os.path.join(a.cache, "keys.npy"))
    X = np.load(os.path.join(a.cache, "X_dev.npy")).astype("float64")
    Xc = np.load(os.path.join(a.cache, "X_comm.npy")).astype("float64")
    split = np.load(os.path.join(a.cache, "split.npz"))
    rows = np.where(~np.isin(keys[:, 2], split["withheld"]))[0]         # trained years only
    cells, e_loc, m_loc, gate = epoch_gate(keys[rows], EPOCH_EARLY, EPOCH_MODERN, MIN_EPOCH_YEARS)
    e_rows = [rows[np.asarray(r)] for r in e_loc]
    m_rows = [rows[np.asarray(r)] for r in m_loc]
    rs = route_sets(pd.read_csv(os.path.join(a.cache, "route_years.csv")), keys, EPOCH_EARLY,
                    EPOCH_MODERN)
    same = np.array([rs.get((int(r), int(c)), (frozenset(), frozenset()))[0]
                     == rs.get((int(r), int(c)), (frozenset(), frozenset()))[1] for r, c in cells])
    rng = np.random.default_rng(a.seed)
    matched = np.zeros(len(cells), bool)
    matched[rng.choice(len(cells), int(same.sum()), replace=False)] = True
    res = {"n_gated_cells": int(len(cells)), "n_same_route": int(same.sum())}
    tot = lambda M: M.sum(1, keepdims=True)                              # community total abundance
    for label, M in (("dev_species", X), ("community_total", tot(Xc))):
        d_full, d_a, d_b = split_half_change(M, e_rows, m_rows, keys[:, 2])
        res[label] = {name: budget(d_full[m], d_a[m], d_b[m]) for name, m in
                      (("all", np.ones(len(cells), bool)), ("same_route", same),
                       ("random_same_size", matched))}
    reg = lambda m: {"mean_row": float(cells[m, 0].mean()), "mean_col": float(cells[m, 1].mean())}
    res["region_mix"] = {"all": reg(np.ones(len(cells), bool)), "same_route": reg(same)}
    with open(os.path.join(a.out, "route_ceiling.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "route_ceiling.json"), encoding="utf-8"))
    print(f"O3 route ceiling: {r['n_same_route']} of {r['n_gated_cells']} gated cells kept the same "
          f"route set across epochs")
    for label in ("dev_species", "community_total"):
        for name, b in r[label].items():
            ps = b["pooled_signal_share"]
            print(f"  {label:16s} {name:17s} cells {b['n_cells']:5d} resolvable {b['n_resolvable']:4d} "
                  f"pooled signal share {('-' if ps is None else f'{ps:.3f}')}  "
                  f"real change/cell {b['mean_real_change_per_cell']:.4f}")


if __name__ == "__main__":
    main()
