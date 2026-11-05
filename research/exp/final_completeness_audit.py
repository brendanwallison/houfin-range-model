"""Compare the documented completeness reconstruction and training-only sensitivity."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("original", "rebuilt", "clean", "cbc", "out"):
        p.add_argument("--" + flag, required=True)
    a = p.parse_args()
    old, full, clean = [np.load(f) for f in (a.original, a.rebuilt, a.clean)]
    for d in (full, clean):
        if d["bench_n"].shape != old["bench_n"].shape:
            raise ValueError("completeness rows are not aligned")
    out = {"reconstruction": {k: {"exact": bool(np.array_equal(old[k], full[k])),
                                      "max_abs_difference": float(np.max(np.abs(old[k] - full[k])))}
                              for k in ("list_len", "bench_n", "bench_hit")}}
    comp = lambda d: np.divide(d["bench_hit"], d["bench_n"], out=np.full_like(d["bench_n"], np.nan), where=d["bench_n"] > 0)
    c0, c1 = comp(full), comp(clean)
    years = np.load(a.cbc)["year_end"]
    out["training_only_sensitivity"] = {}
    for name, lo, hi in [("1902_1939", 1902, 1939), ("1940_1965", 1940, 1965), ("1966_2024", 1966, 2024)]:
        m = (years >= lo) & (years <= hi) & np.isfinite(c0) & np.isfinite(c1)
        out["training_only_sensitivity"][name] = {"lists": int(m.sum()),
            "median_absolute_difference": float(np.median(np.abs(c0[m] - c1[m]))),
            "fraction_changed": float(np.mean(np.abs(c0[m] - c1[m]) > 1e-7))}
    out["interpretation"] = "Aggregate sensitivity includes sealed, focal, other excluded and unmapped taxa; it does not isolate each source. Direct target exclusion is not full response independence."
    Path(a.out).write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
