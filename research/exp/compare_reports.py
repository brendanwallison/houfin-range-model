"""Compare two GP-species report.json files: the reproduction gate (Phase 0.3) and old-vs-new reruns.

    python research/exp/compare_reports.py <local report.json> <reference report.json> [--tol 0.02]

Prints, for the primary and for every level/change pair present in BOTH reports, the two medians, the
reference's bootstrap CI, and a verdict:

    SAME       |delta| <= 1e-6                       (float-identical)
    WITHIN_CI  the local median lies inside the reference's 95% CI (CUDA AMP is not bit-reproducible,
               so this is the reproduction criterion, not float equality)
    DIFFERS    outside the CI and |delta| > tol      (look closer)

Exit status 0 when nothing DIFFERS, 1 otherwise, so a job can gate on it.
"""
import argparse
import json
import sys


def _pairs(rep):
    out = {}
    p = rep.get("primary", {})
    if "median" in p:
        out["primary"] = p
    for section in ("level", "change"):
        for group, tab in (rep.get(section) or {}).items():
            for key, v in tab.items():
                if isinstance(v, dict) and "median" in v:
                    out[f"{section}/{group}/{key}"] = v
    return out


def compare(local, ref, tol=0.02):
    a, b = _pairs(local), _pairs(ref)
    rows, n_diff = [], 0
    for k in sorted(set(a) & set(b)):
        ma, mb = float(a[k]["median"]), float(b[k]["median"])
        lo, hi = (b[k].get("median_ci") or [float("nan"), float("nan")])
        d = ma - mb
        if abs(d) <= 1e-6:
            verdict = "SAME"
        elif lo <= ma <= hi or abs(d) <= tol:
            verdict = "WITHIN_CI"
        else:
            verdict = "DIFFERS"
            n_diff += 1
        rows.append((verdict, k, ma, mb, lo, hi))
    only = {"only_local": sorted(set(a) - set(b)), "only_reference": sorted(set(b) - set(a))}
    return rows, n_diff, only


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("local")
    ap.add_argument("reference")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--max-rows", type=int, default=30)
    a = ap.parse_args()
    local = json.load(open(a.local, encoding="utf-8"))
    ref = json.load(open(a.reference, encoding="utf-8"))
    rows, n_diff, only = compare(local, ref, a.tol)
    counts = {}
    for r in rows:
        counts[r[0]] = counts.get(r[0], 0) + 1
    print(f"compared {len(rows)} medians: {counts}; only-local {len(only['only_local'])}, "
          f"only-reference {len(only['only_reference'])}")
    shown = [r for r in rows if r[0] == "DIFFERS"] + [r for r in rows if r[0] != "DIFFERS"]
    for verdict, k, ma, mb, lo, hi in shown[: a.max_rows]:
        print(f"  {verdict:9s} {k:60s} local {ma:+.4f} ref {mb:+.4f} ref CI [{lo:+.4f}, {hi:+.4f}]")
    sys.exit(1 if n_diff else 0)


if __name__ == "__main__":
    main()
