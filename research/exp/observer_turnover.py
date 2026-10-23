"""E021 (O1, defect A14): how much of the "true change" every change metric counts is OBSERVER turnover?

    ESK_DESK_CONFIG=<overlay> python research/exp/observer_turnover.py --cache <dir> --out <dir>
    python research/exp/observer_turnover.py --summarize <out>

The ABBA split halves years WITHIN an epoch; both halves share that epoch's observers, so a change of observer between
epochs survives the noise subtraction and is counted as real change -- in every ceiling, captured share, correlation
and calibration this project reports. No habitat readout should predict it. Design (from the skeptic's check,
2026-10-09): within the modern epoch, 2005-2014 vs 2016-2025 (same 10-year span for every cell), the place-specific
cross-half change energy for cells whose observer set was unchanged / partly changed / replaced (BBS ObsN of the QC
runs), for dev species (log1p epoch means) and for the reference community in ESK coordinates (r24 and all 64). The
replaced-minus-unchanged excess, against the early -> modern change energy, is the observer share. Observational:
cells that change observers may differ otherwise (reported: the classes' early -> modern energies).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    from change_oracle import group_rows
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import abba_halves, change_noise
    from src.data.preprocess import bbs
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    yr = keys[:, 2]
    runs = bbs.load_run_metadata()                                    # QC runs with ObsN
    ry = pd.read_csv(c("route_years.csv"))
    mm = ry.merge(runs, left_on=["country", "state", "route", "year"],
                  right_on=["CountryNum", "StateNum", "Route", "Year"])
    obs = mm.groupby(["row", "col", "year"])["ObsN"].apply(lambda s: frozenset(s.astype(int))).to_dict()
    obs_of = lambda rows: frozenset().union(*[obs.get((int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2])),
                                                      frozenset()) for i in rows])
    Xd = np.load(c("X_dev.npy")).astype("float64")
    Xc = np.load(c("X_comm.npy")).astype("float64")
    cfg = load_config()
    proj = lambda v: np.asarray(project_points_to_z(np.asarray(v, "float32"), cfg["desk"]["z_dir"], 64), "float64")
    cen = lambda v: v - v.mean(0)

    # early -> modern reference energy (all cells with both epochs; the cache's esk_epochs halves)
    E, full, A, B = group_rows(a.cache, keys, [int(y) for y in split["withheld"]])
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        idx.setdefault(cell, [None, None])[int(ep)] = j
    pairs = np.array([(e, m_) for e, m_ in idx.values() if e is not None and m_ is not None])
    Ys = {h: epoch_mean_observed(Xd, g).astype("float64") for h, g in (("full", full), ("a", A), ("b", B))}
    ds = {h: v[pairs[:, 1]] - v[pairs[:, 0]] for h, v in Ys.items()}
    nz = change_noise(ds["full"], ds["a"], ds["b"])
    sp = nz["resolvable"]
    em_sp = (cen(ds["a"]) * cen(ds["b"])).mean(0)
    sp = sp & (em_sp > 0)
    em_z = {}
    for r in (24, 64):
        za, zb = E["z_a"][:, :r].astype("float64"), E["z_b"][:, :r].astype("float64")
        em_z[r] = float((cen(za[pairs[:, 1]] - za[pairs[:, 0]]) * cen(zb[pairs[:, 1]] - zb[pairs[:, 0]])).sum(1).mean())
    shared = np.array([len(obs_of(full[e]) & obs_of(full[m_])) > 0 for e, m_ in pairs])

    # within-modern, by observer continuity
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    rows_by = {}
    for i in np.where((yr >= 2005) & (yr <= 2025) & (yr != 2015))[0]:
        rows_by.setdefault(cid[i], []).append(i)
    g = {"a1": [], "b1": [], "a2": [], "b2": []}
    kind = []
    for cc, rows in rows_by.items():
        rows = np.array(sorted(rows, key=lambda i: yr[i]))
        r1, r2 = rows[yr[rows] < 2015], rows[yr[rows] > 2015]
        if len(r1) < 4 or len(r2) < 4:
            continue
        o1, o2 = obs_of(r1), obs_of(r2)
        if not o1 or not o2:
            continue
        kind.append("unchanged" if o1 == o2 else ("replaced" if not (o1 & o2) else "partial"))
        h1, h2 = abba_halves(r1, yr[r1]), abba_halves(r2, yr[r2])
        g["a1"].append(h1[0]), g["b1"].append(h1[1]), g["a2"].append(h2[0]), g["b2"].append(h2[1])
    kind = np.array(kind)
    Ym = {k: epoch_mean_observed(Xd, v).astype("float64") for k, v in g.items()}
    Zm = {k: proj(epoch_mean_observed(Xc, v)) for k, v in g.items()}
    da_s, db_s = cen(Ym["a2"] - Ym["a1"]), cen(Ym["b2"] - Ym["b1"])
    rng = np.random.default_rng(0)
    iu, ir = np.where(kind == "unchanged")[0], np.where(kind == "replaced")[0]
    res = {"cache": a.cache, "n_cells_within_modern": {k: int((kind == k).sum()) for k in ("unchanged", "partial",
                                                                                         "replaced")},
           "share_no_shared_observer_early_modern": float(1 - shared.mean()),
           "n_resolvable_species": int(sp.sum()), "species": {}, "esk": {}}
    en_s = {k: float((da_s[kind == k] * db_s[kind == k]).mean(0)[sp].sum()) for k in ("unchanged", "partial",
                                                                                      "replaced")}
    bs = [((da_s[b_] * db_s[b_]).mean(0)[sp].sum() - (da_s[a_] * db_s[a_]).mean(0)[sp].sum()) / em_sp[sp].sum()
          for a_, b_ in ((rng.choice(iu, len(iu)), rng.choice(ir, len(ir))) for _ in range(a.n_boot))]
    res["species"] = {"early_modern_energy": float(em_sp[sp].sum()), "within_modern": en_s,
                      "observer_share": (en_s["replaced"] - en_s["unchanged"]) / float(em_sp[sp].sum()),
                      "observer_share_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    for r in (24, 64):
        za, zb = cen(Zm["a2"][:, :r] - Zm["a1"][:, :r]), cen(Zm["b2"][:, :r] - Zm["b1"][:, :r])
        en = {k: float((za[kind == k] * zb[kind == k]).sum(1).mean()) for k in ("unchanged", "partial", "replaced")}
        bz = [(float((za[b_] * zb[b_]).sum(1).mean()) - float((za[a_] * zb[a_]).sum(1).mean())) / em_z[r]
              for a_, b_ in ((rng.choice(iu, len(iu)), rng.choice(ir, len(ir))) for _ in range(a.n_boot))]
        res["esk"][str(r)] = {"early_modern_energy": em_z[r], "within_modern": en,
                              "observer_share": (en["replaced"] - en["unchanged"]) / em_z[r],
                              "observer_share_ci": [float(np.percentile(bz, 2.5)), float(np.percentile(bz, 97.5))]}
    with open(os.path.join(a.out, "observer_turnover.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "observer_turnover.json"), encoding="utf-8"))
    n = r["n_cells_within_modern"]
    print(f"observer turnover: within-modern cells unchanged {n['unchanged']}, partial {n['partial']}, replaced "
          f"{n['replaced']}; {r['share_no_shared_observer_early_modern']:.0%} of cells share no observer early vs modern")
    s = r["species"]
    w = s["within_modern"]
    print(f"  species ({r['n_resolvable_species']} resolvable): within-modern change energy unchanged {w['unchanged']:.3f} "
          f"/ partial {w['partial']:.3f} / replaced {w['replaced']:.3f}; early->modern {s['early_modern_energy']:.3f}; "
          f"observer share {s['observer_share']:.0%} [{s['observer_share_ci'][0]:.0%}, {s['observer_share_ci'][1]:.0%}]")
    for rk, e in r["esk"].items():
        w = e["within_modern"]
        print(f"  ESK r{rk}: unchanged {w['unchanged']:.4f} / partial {w['partial']:.4f} / replaced {w['replaced']:.4f}; "
              f"early->modern {e['early_modern_energy']:.4f}; observer share {e['observer_share']:.0%} "
              f"[{e['observer_share_ci'][0]:.0%}, {e['observer_share_ci'][1]:.0%}]")


if __name__ == "__main__":
    main()
