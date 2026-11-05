"""Final-candidate CBC confirmation using the established conditional list scorer.

Fits the season-specific ridge readout and winter offsets on modern training cells;
all candidates are compared on identical sealed-species lists at held-out cells.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e069_presence_score import ridge_cv, fit_delta, list_ll
from final_holdout_prepare import ARMS

ERAS = [("1902_1919", 1902, 1919), ("1920_1939", 1920, 1939),
        ("1940_1952", 1940, 1952), ("1953_1965", 1953, 1965),
        ("1966_1985", 1966, 1985), ("1986_1995", 1986, 1995),
        ("1996_2004", 1996, 2004), ("2005_2024", 2005, 2024)]


def exclude_test_circles(train, test, circle):
    """A relocated circle must not calibrate a test circle under another cell label."""
    return train & ~np.isin(circle, np.unique(circle[test]))


def paired_circle_summary(circle, values, boot=1000):
    """Same list-weighted circle bootstrap as e086p, using cluster sums."""
    uc, inv = np.unique(circle, return_inverse=True)
    out = {"lists": len(circle), "circles": len(uc)}
    if len(uc) < 10:
        return out
    n = np.bincount(inv)
    rng = np.random.default_rng(86)
    # Explicit choice matches the established resampling scheme.
    draws = np.stack([np.bincount(rng.choice(len(uc), len(uc)), minlength=len(uc)) for _ in range(boot)])
    out["metrics"] = {}
    for name, v in values.items():
        if not np.isfinite(v).all():
            raise ValueError("non-finite list score")
        sums = np.bincount(inv, weights=v)
        b = (draws @ sums) / (draws @ n)
        out["metrics"][name] = {"value": float(np.mean(v)),
                                  "ci95": np.percentile(b, [2.5, 97.5]).tolist()}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--cbc", required=True, type=Path)
    p.add_argument("--reason", required=True)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    from research.adopt import record_read
    record_read(str(a.cbc), a.reason)
    d = np.load(a.cbc, allow_pickle=False)
    out = a.root / "winter"
    out.mkdir(exist_ok=True)
    f0 = a.root / "features" / "final"
    cells, years = np.load(f0 / "cells.npy"), np.load(f0 / "years.npy")
    ho, bf = np.load(f0 / "holdout_cells.npy"), np.load(f0 / "buffer_cells.npy")
    lookup = {tuple(rc): i for i, rc in enumerate(cells)}
    ci = np.array([lookup.get((r, c), -1) for r, c in zip(d["row"], d["col"])])
    yr = d["year_end"]
    ok = (ci >= 0) & (yr >= years[0]) & (yr <= 2024)
    # Check geometry/finite support common to both candidates before inspecting scores.
    rr, cc = np.maximum(d["row"], 0), np.maximum(d["col"], 0)
    test = ok & ho[rr, cc] & ~bf[rr, cc]
    train = ok & ~ho[rr, cc] & ~bf[rr, cc] & (yr >= 1996)
    ph = d["party_hours"]
    train &= np.isfinite(ph) & (ph > 0)
    cir = d["circle_idx"]
    overlap_circles = sorted(set(cir[train]) & set(cir[test]))
    train = exclude_test_circles(train, test, cir)
    features = {}
    for family in ("final", "final_v2"):
        f = a.root / "features" / family
        if not np.array_equal(np.load(f / "cells.npy"), cells) or not np.array_equal(np.load(f / "years.npy"), years):
            raise ValueError("candidate geometry differs")
        dense = np.load(f / "z_raw_all.npy", mmap_mode="r")
        rows = np.flatnonzero(ok)
        valid = np.isfinite(dense[ci[rows], yr[rows] - years[0]]).all(1)
        finite = np.zeros(len(yr), bool)
        finite[rows] = valid
        train &= finite
        test &= finite
        features[family] = dense
    Y = d["Y"]
    # Fixed calibration-support eligibility, shared by every candidate.
    keep = (Y[train] > 0).sum(0) >= 10
    species = d["species"][keep]
    if len(species) < 5 or "houfin" in species:
        raise ValueError("insufficient or forbidden confirmation species")
    P = Y[:, keep] > 0
    train &= P.sum(1) >= 2
    test &= P.sum(1) >= 2
    tr, te = np.flatnonzero(train), np.flatnonzero(test)
    target = np.log1p(Y[tr][:, keep] * ((np.median(ph[tr]) / ph[tr])[:, None] ** .8))
    groups = (d["row"][tr] // 6) * 100000 + d["col"][tr] // 6
    all_scores = {}
    details = {}
    for arm, (family, cols, _) in ARMS.items():
        saved = out / f"lists_{arm}.npz"
        if saved.exists():
            z = np.load(saved)
            if not np.array_equal(z["test_rows"], te) or not np.array_equal(z["species"], species):
                raise ValueError("saved winter scores belong to a different support set")
            all_scores[arm] = (z["backcast"], z["modern"])
            details[arm] = json.loads((out / f"fit_{arm}.json").read_text())
            continue
        dense = features[family]
        ztr = np.asarray(dense[ci[tr], yr[tr] - years[0]][:, cols], dtype="float64")
        readout, lam = ridge_cv(ztr, target, groups, [1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        delta = fit_delta(readout(ztr), P[tr])
        modern = np.asarray(dense[:, (years >= 2005) & (years <= 2024)].mean(1)[:, cols], dtype="float64")
        back, mod = [], []
        for offset in range(0, len(te), 8192):
            ix = te[offset:offset + 8192]
            z = np.asarray(dense[ci[ix], yr[ix] - years[0]][:, cols], dtype="float64")
            back.append(list_ll(readout(z), delta, P[ix]))
            mod.append(list_ll(readout(modern[ci[ix]]), delta, P[ix]))
        back, mod = np.concatenate(back), np.concatenate(mod)
        np.savez_compressed(saved, test_rows=te, species=species, backcast=back, modern=mod)
        details[arm] = {"ridge_lambda": lam, "calibration_rows": len(tr), "species": len(species)}
        (out / f"fit_{arm}.json").write_text(json.dumps(details[arm], indent=2))
        all_scores[arm] = (back, mod)
        print(f"winter {arm}: fitted {len(tr)} modern lists; scored {len(te)} test lists", flush=True)
    res = {"species": species.tolist(), "calibration_rows": len(tr), "test_rows": len(te),
           "readout": "winter-count ridge with modern training-cell offsets; raw features", "fits": details,
           "relocated_test_circles_removed_from_calibration": len(overlap_circles),
           "holdout_caveat": "sealed taxa excluded directly; completeness may carry indirect response information",
           "eras": {}}
    modern_counts = np.bincount(cir[te][yr[te] >= 2005], minlength=len(d["circles"]))
    refb, refm = all_scores["v1_r64"]
    for era, lo, hi in ERAS:
        m = (yr[te] >= lo) & (yr[te] <= hi)
        counts = np.bincount(cir[te][m], minlength=len(d["circles"]))
        stable = m & (counts[cir[te]] >= 3) & (modern_counts[cir[te]] >= 3)
        effort = m & (d["field_counters"][te] > 0) & (d["field_counters"][te] <= 6)
        metrics = {}
        for arm, (back, mod) in all_scores.items():
            metrics[f"{arm}/backcast_minus_modern"] = back - mod
            metrics[f"{arm}/minus_v1"] = back - refb
            metrics[f"{arm}/gain_minus_v1"] = (back - mod) - (refb - refm)
        res["eras"][era] = {}
        for label, mask in [("primary", m), ("same_circles", stable), ("counters_1_6", effort)]:
            res["eras"][era][label] = paired_circle_summary(cir[te][mask], {k: v[mask] for k, v in metrics.items()}, a.boot)
    (out / "result.json").write_text(json.dumps(res, indent=2, allow_nan=False))
    print(f"winter done: {out / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
