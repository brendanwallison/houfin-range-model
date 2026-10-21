"""E086p: do CBC / eBird-trained DESKs predict EARLY-CENTURY (1902-1952) CBC lists better than their controls -- at every
CBC circle, including the ~2,000 cells with no BBS route?

    # 1. encode one model from 1902 at every CBC cell (one process per model; reads the cache's own overlay / run dir)
    python research/exp/e086p_early_lists.py encode --cache <research cache> --cbc cbc_circles_comm.npz --out <dir>
    # 2. score an arm against its control
    python research/exp/e086p_early_lists.py score --arm <cache> --arm-ext <dir> --control <cache> --control-ext <dir>
        --cbc cbc_circles_comm.npz --similarity species_similarity.csv --out <dir> --label NAME

encode: DESK z_ema (validate_bbs_routes.desk_z_ema: the model's own output EMA, scanned from 1902) at every cell holding
a CBC list, 1902-2025 -> z_ema_ext.npy (cells x years x L), cells_ext.npy, years_ext.npy.
score (the E069b machinery, per cache): a level ridge readout of the cache's own z_ema (training cell-years, trained years
>= 1996) to the resident community species' log1p counts; species offsets from 2005-2024 lists at non-test cells; per-
list log P(species | record) for the BACKCAST (z at the list's year) and the MODERN placebo (the cell's mean z
2005-2024). Arm - control paired over the same lists, bootstrap over circles. Eras 1902-1919, 1920-1939 (no model has
seen any pre-1940 list: the presence term used 1940+, the Ruzicka term and heads 1966+), 1940-1952, 1953-1965; cell
types: BBS-trained, BBS held-out or buffer, NO BBS route.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e069_presence_score import fit_delta, list_ll, ridge_cv  # noqa: E402

ERAS = (("1902_1919", 1902, 1919), ("1920_1939", 1920, 1939), ("1940_1952", 1940, 1952), ("1953_1965", 1953, 1965))


def encode(a):
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    run_dir = meta["run_dir"]
    os.environ["ESK_DESK_CONFIG"] = meta["overlay"]
    os.environ.setdefault("RESEARCH_OUT", os.path.dirname(os.path.abspath(run_dir)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.validate_bbs_routes import desk_z_ema
    cfg = load_config()
    cfg["paths"]["desk_output_dir"] = run_dir
    D = np.load(a.cbc, allow_pickle=False)
    cells = np.unique(np.stack([D["row"], D["col"]], 1), axis=0)
    years = np.arange(1902, 2026)
    keys = np.array([(r, c, y) for r, c in cells for y in years], "int32")
    Z, info = desk_z_ema(cfg, keys)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "z_ema_ext.npy"), Z.reshape(len(cells), len(years), -1).astype("float32"))
    np.save(os.path.join(a.out, "cells_ext.npy"), cells.astype("int32")); np.save(os.path.join(a.out, "years_ext.npy"), years)
    json.dump({"cache": a.cache, "run_dir": run_dir, "ema": info}, open(os.path.join(a.out, "encode.json"), "w"), indent=1)
    print(f"[e086p] encoded {len(cells)} CBC cells x {len(years)} years from {run_dir}", flush=True)


def per_list(cache, ext, D, cols_sp, cell_type):
    meta = json.load(open(os.path.join(cache, "meta.json")))
    keys = np.load(os.path.join(cache, "keys.npy")); Xc = np.load(os.path.join(cache, "X_comm.npy"))
    zk = np.load(os.path.join(cache, "z_ema.npy")); spl = np.load(os.path.join(cache, "split.npz"))
    Zx = np.load(os.path.join(ext, "z_ema_ext.npy"), mmap_mode="r")
    cx = np.load(os.path.join(ext, "cells_ext.npy")); yx = np.load(os.path.join(ext, "years_ext.npy"))
    comm, cs = list(meta["community"]), list(D["species"])
    tr = spl["is_train"] & (keys[:, 2] >= 1996)
    T = np.log1p(Xc[:, [comm.index(s) for s in cols_sp]])
    ro, _ = ridge_cv(zk[tr].astype(float), T[tr], spl["block_id"][tr], [1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    cix = {(int(r), int(c)): i for i, (r, c) in enumerate(cx)}
    ci = np.array([cix[(int(r), int(c))] for r, c in zip(D["row"], D["col"])])
    yr = D["year_end"]
    P = D["Y"][:, [cs.index(s) for s in cols_sp]] > 0
    ok = (yr >= int(yx[0])) & (yr <= 2024) & (P.sum(1) >= 2)
    zt = lambda rr: np.asarray(Zx[ci[rr], yr[rr] - int(yx[0])], float)
    zmod = np.asarray(Zx[:, (yx >= 2005) & (yx <= 2024)].mean(1), float)
    # circles outside the covariate grid (Canada, coast) have no z: drop their lists (every year of a cell is NaN or not)
    finite_cell = np.isfinite(np.asarray(Zx[:, 0, 0], float)) & np.isfinite(zmod).all(1)
    ok &= finite_cell[ci]
    test = cell_type != "bbs_trained"
    rm = np.where(ok & ~test & (yr >= 2005))[0]                  # offsets from modern lists at BBS-trained cells
    delta = fit_delta(ro(zt(rm)), P[rm])
    out = {}
    for era, e0, e1 in ERAS:
        for ctype in ("bbs_trained", "bbs_heldout", "no_bbs", "all"):
            m = ok & (yr >= e0) & (yr <= e1) & ((cell_type == ctype) if ctype != "all" else True)
            rr = np.where(m)[0]
            out[(era, ctype)] = (rr, list_ll(ro(zt(rr)), delta, P[rr]), list_ll(ro(zmod[ci[rr]]), delta, P[rr]))
    return out


def score(a):
    import pandas as pd
    D = np.load(a.cbc, allow_pickle=False)
    meta = json.load(open(os.path.join(a.arm, "meta.json")))
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    cs = list(D["species"])
    allsp = [s for s in meta["community"] if s in cs]
    res_sp = [s for s in allsp if mig.get(s) in (1.0, 2.0)]
    assert "houfin" not in allsp
    spl = np.load(os.path.join(a.arm, "split.npz")); cells = np.load(os.path.join(a.arm, "cells.npy"))
    bbs = {(int(r), int(c)) for r, c in cells}
    trc = ~spl["holdout"] & ~spl["buffer"]
    cell_type = np.array(["no_bbs" if (int(r), int(c)) not in bbs else ("bbs_trained" if trc[r, c] else "bbs_heldout")
                          for r, c in zip(D["row"], D["col"])])
    res = {"label": a.label, "arm": a.arm, "control": a.control}
    for name, sp in (("residents", res_sp), ("all", allsp)):
        A = per_list(a.arm, a.arm_ext, D, sp, cell_type); C = per_list(a.control, a.control_ext, D, sp, cell_type)
        out = {"species": len(sp)}
        if a.save_lists:   # E086p addendum: per-list arrays for the effort-stratified check
            os.makedirs(a.out, exist_ok=True)
            np.savez_compressed(os.path.join(a.out, f"lists_{a.label}_{name}.npz"),
                                **{f"{k[0]}|{k[1]}|{nm}": v for k in A for nm, v in zip(("rows", "lba", "lma"), A[k])},
                                **{f"{k[0]}|{k[1]}|{nm}": v for k in C for nm, v in zip(("rows_c", "lbc", "lmc"), C[k])})
        for key in A:
            ra, lba, lma = A[key]; rc, lbc, lmc = C[key]
            assert (ra == rc).all()
            circ = D["circle_idx"][ra]; uc = np.unique(circ)
            k = f"{key[0]}|{key[1]}"
            if len(uc) < 10:
                out[k] = {"lists": int(len(ra)), "circles": int(len(uc))}
                continue
            d_back, d_gain = lba - lbc, (lba - lma) - (lbc - lmc)
            idx = {c: np.where(circ == c)[0] for c in uc}
            rng = np.random.default_rng(86)
            bb, bg = [], []
            for _ in range(a.boot):
                pk = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
                bb.append(d_back[pk].mean()); bg.append(d_gain[pk].mean())
            out[k] = {"lists": int(len(ra)), "circles": int(len(uc)),
                      "arm_backcast_minus_modern": float((lba - lma).mean()), "control_backcast_minus_modern": float((lbc - lmc).mean()),
                      "arm_minus_control_backcast": float(d_back.mean()),
                      "arm_minus_control_backcast_ci95": [float(x) for x in np.percentile(bb, [2.5, 97.5])],
                      "arm_minus_control_gain": float(d_gain.mean()),
                      "arm_minus_control_gain_ci95": [float(x) for x in np.percentile(bg, [2.5, 97.5])]}
        res[name] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"e086p_{a.label}.json"), "w"), indent=1)
    for k, v in res["residents"].items():
        if isinstance(v, dict) and "arm_minus_control_gain" in v:
            print(f"[e086p] {a.label} residents {k:24s} lists {v['lists']:5d} circles {v['circles']:4d} | arm-ctl gain "
                  f"{v['arm_minus_control_gain']:+.3f} [{v['arm_minus_control_gain_ci95'][0]:+.3f},{v['arm_minus_control_gain_ci95'][1]:+.3f}] "
                  f"| ctl backcast-modern {v['control_backcast_minus_modern']:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("--cache", required=True); e.add_argument("--cbc", required=True); e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    for k in ("--arm", "--arm-ext", "--control", "--control-ext", "--cbc", "--similarity", "--out"):
        s.add_argument(k, required=True)
    s.add_argument("--label", default="arm"); s.add_argument("--boot", type=int, default=500)
    s.add_argument("--save-lists", action="store_true")
    a = ap.parse_args()
    encode(a) if a.cmd == "encode" else score(a)


if __name__ == "__main__":
    main()
