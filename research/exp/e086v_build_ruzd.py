"""E086v data step: CBC Ruzicka targets whose WITHHELD-ERA winters come from a pseudo-early (degraded) copy.

    python research/exp/e086v_build_ruzd.py --base cbc_ruzicka_res.npz --cbc <degraded cbc_circles_comm.npz>
        --years 1966 1995 --out cbc_ruzicka_res_deg.npz

Rows outside --years are taken unchanged from --base (E081: party-hours standardized, 1966-2024). Rows inside --years are
rebuilt from the degraded copy, where party-hours are missing (as before 1940): counts standardized by FIELD COUNTERS to
their median over the window's lists with exponent 0.8 (E081's exponent; counters are the only effort measure early
counts have), averaged over the circles of a cell-year, the base file's species. The calibration (a, b) and weight are
the base file's, so only the window's data differ.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True); ap.add_argument("--cbc", required=True)
    ap.add_argument("--years", type=int, nargs=2, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    B = dict(np.load(a.base)); sp = list(B["species"]); assert "houfin" not in sp
    D = np.load(a.cbc, allow_pickle=False); cs = list(D["species"])
    yr, fc = D["year_end"], D["field_counters"].astype(float)
    win = (D["row"] >= 0) & (yr >= a.years[0]) & (yr <= a.years[1]) & np.isfinite(fc) & (fc > 0)
    Y = D["Y"][win][:, [cs.index(s) for s in sp]].astype(float)
    med = np.median(fc[win])
    Ys = Y * ((med / fc[win])[:, None] ** 0.8)
    df = pd.DataFrame(Ys); df["r"], df["c"], df["y"] = D["row"][win], D["col"][win], yr[win]
    g = df.groupby(["r", "c", "y"]).mean()
    keep = (B["year"] < a.years[0]) | (B["year"] > a.years[1])
    out = {k: np.concatenate([B[k][keep], np.array([i[j] for i in g.index], np.int32)])
           for j, k in enumerate(("row", "col", "year"))}
    out["X"] = np.concatenate([B["X"][keep], g.to_numpy().astype(np.float32)])
    np.savez_compressed(a.out, **out, species=B["species"], a=B["a"], b=B["b"], weight=B["weight"])
    info = {"base_rows_kept": int(keep.sum()), "window_rows": int(len(g)), "window_lists": int(win.sum()),
            "median_counters": float(med), "base_window_rows": int((~keep).sum())}
    json.dump(info, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(f"[E086v] {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
