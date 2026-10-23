"""E081 data step: CBC cell-year communities (resident reference species, 1966-2024, effort-standardized) and a FIXED
calibration from winter CBC Ruzicka similarity to DESK's breeding-season similarity, for desk.aux_ruzicka.

    python research/exp/build_cbc_ruzicka.py --cache <research cache> --cbc cbc_circles_comm.npz
        --presence cbc_presence_res.npz --out cbc_ruzicka_res.npz

Species: the presence term's 59 resident reference species (same list). Counts standardized to the median party-hours
with exponent 0.8 (E064's median; E066's T1), averaged over the circles of a grid cell-year, winters ending 1966-2024
with party-hours. Calibration: on cell-years that also have BBS ESK coordinates in TRAINED years (>= 1996) and
training cells, 200k random pairs; OLS of dot(ESK z_i, ESK z_j) on Ruzicka(log1p CBC_i, log1p CBC_j) gives a, b; the
term asks dot(z_i, z_j) ~ a + b * Ruzicka_CBC(i, j). Weight rule (fixed now): weight = 0.03 / residual MSE of that
calibration, aiming at a ~15% loss share at DESK's epoch-150 loss scale (E061: total ~0.17).
"""
import argparse
import json
import os

import numpy as np


def ruz(A, B):
    A, B = np.log1p(A), np.log1p(B)
    den = np.maximum(A, B).sum(1)
    return np.where(den > 0, np.minimum(A, B).sum(1) / np.maximum(den, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--cbc", required=True)
    ap.add_argument("--presence", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    C = a.cache
    keys = np.load(os.path.join(C, "keys.npy")); esk = np.load(os.path.join(C, "esk_annual.npy")).astype(float)
    spl = np.load(os.path.join(C, "split.npz"))
    D = np.load(a.cbc, allow_pickle=False)
    sp = list(np.load(a.presence)["species"])
    cs = list(D["species"])
    assert "houfin" not in sp
    Y = D["Y"][:, [cs.index(s) for s in sp]].astype(float)
    yr, ph = D["year_end"], D["party_hours"]
    ok = (D["row"] >= 0) & (yr >= 1966) & (yr <= 2024) & np.isfinite(ph) & (ph > 0)
    med = np.median(ph[ok])
    Ys = Y[ok] * ((med / ph[ok])[:, None] ** 0.8)
    df = pd.DataFrame(Ys); df["r"], df["c"], df["y"] = D["row"][ok], D["col"][ok], yr[ok]
    g = df.groupby(["r", "c", "y"]).mean()
    rr = np.array([k[0] for k in g.index]); cc = np.array([k[1] for k in g.index]); yy = np.array([k[2] for k in g.index])
    X = g.to_numpy()
    # calibration on cell-years with ESK z in trained years and training cells
    kidx = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(keys) if spl["is_train"][i] and y >= 1996}
    m = np.array([kidx.get((int(r), int(c), int(y)), -1) for r, c, y in zip(rr, cc, yy)])
    have = np.where(m >= 0)[0]
    rng = np.random.default_rng(81)
    I, J = rng.choice(have, 200000), rng.choice(have, 200000)
    keep = I != J
    I, J = I[keep], J[keep]
    rz = ruz(X[I], X[J]); dt = (esk[m[I]] * esk[m[J]]).sum(1)
    ok2 = np.isfinite(rz)
    b, a0 = np.polyfit(rz[ok2], dt[ok2], 1)
    resid = dt[ok2] - (a0 + b * rz[ok2])
    mse = float(np.mean(resid ** 2))
    corr = float(np.corrcoef(rz[ok2], dt[ok2])[0, 1])
    weight = 0.03 / mse
    np.savez_compressed(a.out, year=yy.astype(np.int32), row=rr.astype(np.int32), col=cc.astype(np.int32),
                        X=X.astype(np.float32), species=np.array(sp), a=np.float32(a0), b=np.float32(b),
                        weight=np.float32(weight))
    info = {"cell_years": int(len(X)), "calibration_cell_years": int(len(have)), "a": float(a0), "b": float(b),
            "corr_ruz_cbc_vs_dot_esk": corr, "residual_mse": mse, "weight": weight,
            "dot_esk_var": float(np.var(dt[ok2]))}
    json.dump(info, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(f"[E081] {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
