"""Observation-damage check using E086v's EXISTING pseudo-early 1966-95 copy.

Recover the fixed winter readouts (same selected lambda, same calibration rows),
verify their original scores, then change observations only, never the features.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e069_presence_score import fit_delta, list_ll
from final_winter_score import exclude_test_circles, paired_circle_summary
from final_holdout_prepare import ARMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--damaged", required=True, type=Path)
    a = p.parse_args()
    from research.adopt import record_read
    record_read(str(a.damaged), "FINAL_HELDOUT: fixed-readout pseudo-early observation-damage check, no model selection")
    root = a.root
    d, damaged = np.load(root / "sealed/cbc_circles.npz"), np.load(a.damaged)
    saved = np.load(root / "winter/lists_v1_r64.npz")
    species, te = saved["species"], saved["test_rows"]
    columns = [list(d["species"]).index(s) for s in species]
    if any(s not in damaged["species"] for s in species):
        raise ValueError("degraded extraction changed taxonomy support; cannot treat missing taxa as absences")
    dc = [list(damaged["species"]).index(s) for s in species]
    f0 = root / "features/final"
    cells, years = np.load(f0 / "cells.npy"), np.load(f0 / "years.npy")
    ho, bf = np.load(f0 / "holdout_cells.npy"), np.load(f0 / "buffer_cells.npy")
    look = {tuple(rc): i for i, rc in enumerate(cells)}
    ci = np.array([look.get((r, c), -1) for r, c in zip(d["row"], d["col"])])
    yr = d["year_end"]
    ok = (ci >= 0) & (yr >= 1902) & (yr <= 2024)
    rr, cc = np.maximum(d["row"], 0), np.maximum(d["col"], 0)
    test = ok & ho[rr, cc] & ~bf[rr, cc]
    ph = d["party_hours"]
    train = ok & ~ho[rr, cc] & ~bf[rr, cc] & (yr >= 1996) & np.isfinite(ph) & (ph > 0)
    train = exclude_test_circles(train, test, d["circle_idx"])
    features = {family: np.load(root / f"features/{family}/z_raw_all.npy", mmap_mode="r")
                for family in ("final", "final_v2")}
    for dense in features.values():
        ix = np.flatnonzero(ok)
        finite = np.zeros(len(yr), bool)
        finite[ix] = np.isfinite(dense[ci[ix], yr[ix] - 1902]).all(1)
        train &= finite
    P = d["Y"][:, columns] > 0
    train &= P.sum(1) >= 2
    tr = np.flatnonzero(train)
    target = np.log1p(d["Y"][tr][:, columns] * ((np.median(ph[tr]) / ph[tr])[:, None] ** .8))
    dm = {(str(damaged["circles"][c]), int(y)): i for i, (c, y) in
          enumerate(zip(damaged["circle_idx"], damaged["year_end"]))}
    jj = np.array([dm.get((str(d["circles"][d["circle_idx"][i]]), int(yr[i])), -1) for i in te])
    mask = (yr[te] >= 1966) & (yr[te] <= 1995) & (jj >= 0)
    ii = np.flatnonzero(mask)
    Pdamage = damaged["Y"][jj[ii]][:, dc] > 0
    take = Pdamage.sum(1) >= 2
    ii, Pdamage = ii[take], Pdamage[take]
    matched = te[ii]
    scores, verification = {}, {}
    for arm in ("v1_r64", "v2_r64", "v2_r128"):
        family, cols, _ = ARMS[arm]
        dense = features[family]
        fit = json.loads((root / f"winter/fit_{arm}.json").read_text())
        if fit["calibration_rows"] != len(tr):
            raise ValueError("reconstructed calibration rows differ")
        X = np.asarray(dense[ci[tr], yr[tr] - 1902][:, cols], dtype="float64")
        xm, ym = X.mean(0), target.mean(0)
        xc = X - xm
        A = xc.T @ xc
        B = np.linalg.solve(A + fit["ridge_lambda"] * np.trace(A) / A.shape[0] * np.eye(A.shape[0]), xc.T @ (target - ym))
        predict = lambda z: (z - xm) @ B + ym
        delta = fit_delta(predict(X), P[tr])
        dated = np.asarray(dense[ci[matched], yr[matched] - 1902][:, cols], dtype="float64")
        modern = np.asarray(dense[:, (years >= 2005) & (years <= 2024)].mean(1)[:, cols], dtype="float64")
        eta_b, eta_m = predict(dated), predict(modern[ci[matched]])
        original_b, original_m = list_ll(eta_b, delta, P[matched]), list_ll(eta_m, delta, P[matched])
        prior = np.load(root / f"winter/lists_{arm}.npz")
        error = max(np.max(np.abs(original_b - prior["backcast"][ii])), np.max(np.abs(original_m - prior["modern"][ii])))
        if error > 1e-6:
            raise ValueError(f"failed independent readout reconstruction: {arm}: {error}")
        verification[arm] = float(error)
        scores[arm] = {"original": original_b - original_m,
                       "damaged": list_ll(eta_b, delta, Pdamage) - list_ll(eta_m, delta, Pdamage)}
    values = {}
    for arm, data in scores.items():
        for state in ("original", "damaged"):
            values[f"{arm}/{state}/backcast_minus_modern"] = data[state]
            values[f"{arm}/{state}/gain_minus_v1"] = data[state] - scores["v1_r64"][state]
        values[f"{arm}/damage_shift_in_gain"] = data["damaged"] - data["original"]
    result = {"verification_max_error": verification, "years": [1966, 1995],
              "damage": str(a.damaged), "features_unchanged": True,
              "result": paired_circle_summary(d["circle_idx"][matched], values),
              "interpretation": "Differences between original and damaged rows arise entirely from observations at the same sites/years. No encoder/readout was tuned."}
    (root / "winter/damage_check.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
