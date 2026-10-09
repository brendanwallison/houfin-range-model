"""Change-on-change oracle: is held-out species change predictable from COMMUNITY change at all, and does a
level-fitted readout transfer to change? (E4 / P1 / T4 in one table.)

    python research/exp/change_oracle.py --cache <dir> --out <dir> [--population trained|withheld]
        [--ranks 24 64]
    python research/exp/change_oracle.py --summarize <out>

Cell-epoch level, on the cache's held-out cells, every arm graded against the SAME target. The
features come from split half A and the target from half B (ABBA within each cell-epoch), so survey
noise shared by the community and the species (a poor year depresses both) cannot pass for signal:

    trend       each species' mean change over training cells (the continental trend; no place info)
    F:lvl       beta fitted on LEVELS (cell-epoch log1p means ~ features, training cells, trained
                years), applied to the held-out feature change -- the age model's readout form
    F:lvl+trend the same plus the species' trend
    F:chg       beta fitted on CHANGE (training cells' change ~ their feature change) + intercept --
                the best a linear readout of this feature change can do (an oracle: it sees change)
    F:chg_dev   chg minus its intercept part: the place-specific, community-driven change only

for F in esk (the OBSERVED community's ESK coordinates, half A) and desk_raw / desk_ema (the model's z,
epoch means over the surveyed years), at each rank. esk:chg low => species change is not a linear
function of community change (P1: trends, fronts); esk:chg high but esk:lvl low => level betas do not
transfer to change (space-for-time, T4/B1); esk:lvl high but desk:lvl low => DESK's information is
the bottleneck. Every BLR is ``research/lib/blr`` with one iid block (the age model's prior).
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
from lib import blr, ledger  # noqa: E402

BLOCK = 6


def group_rows(cache, keys, withheld):
    """The esk_project groups and their ABBA halves, rebuilt and checked against esk_epochs.npz."""
    from esk_project import epoch_groups
    from src.community_encoder.train_DESK.validate_bbs_routes import EPOCH_EARLY, EPOCH_MODERN
    from src.community_encoder.train_DESK.validation_core import split_half_groups
    E = dict(np.load(os.path.join(cache, "esk_epochs.npz")))
    groups = epoch_groups(keys, withheld, EPOCH_EARLY, EPOCH_MODERN)
    gk = list(groups)
    glist = [groups[k] for k in gk]
    A, B, ok = split_half_groups(glist, years=keys[:, 2])
    keep = np.where(ok)[0]
    full = [glist[i] for i in keep]
    flat = np.concatenate(full)
    assert np.array_equal(flat, E["rows_flat"]), "groups differ from esk_epochs.npz"
    return E, full, [A[i] for i in keep], [B[i] for i in keep]


def desk_epoch_z(z_all, cell_index, years0, keys, groups, rank):
    zi = [np.asarray(z_all[cell_index[g], keys[g, 2] - years0, :rank], "float64").mean(0)
          for g in map(np.asarray, groups)]
    return np.stack(zi)


def metrics(dp, d_b, nz_b, blocks, n_boot, seed=0):
    from fast_arm import attenuation
    from src.community_encoder.train_DESK.validate_gp_species import block_sums, pooled_skill
    from src.community_encoder.train_DESK.validation_core import captured_share
    res = nz_b["resolvable"]
    _, sa = block_sums((dp - d_b) ** 2, blocks)
    _, sb = block_sums(d_b ** 2, blocks)
    _, sg = block_sums((np.abs(d_b) > 0).astype("float64"), blocks)
    sa[:, ~res], sb[:, ~res] = 0.0, 0.0
    pooled, _ = pooled_skill(sa, sb, n_boot, seed, block_signal=sg)
    _, cap = captured_share(d_b, dp, nz_b)
    att = attenuation(dp, d_b, nz_b)
    fin = np.isfinite(att)
    return {"skill_vs_no_change": pooled.get("median"), "skill_ci": pooled.get("median_ci"),
            "captured_pooled": cap,
            "attenuation_median": float(np.median(att[fin])) if fin.any() else None,
            "attenuation_iqr": ([float(np.percentile(att[fin], q)) for q in (25, 75)]
                                if fin.any() else None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="trained", choices=("trained", "withheld"))
    ap.add_argument("--ranks", type=int, nargs="+", default=(24, 64))
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--exp", default=os.environ.get("RESEARCH_EXP", "adhoc"))
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import change_noise
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    E, full, half_a, half_b = group_rows(a.cache, keys, withheld)
    Xd = np.load(c("X_dev.npy")).astype("float64")
    meta = json.load(open(c("meta.json"), encoding="utf-8"))

    # pair each cell's early and modern group for the population
    want_early_wh = a.population == "withheld"
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh) == want_early_wh:
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    if not len(pairs):
        sys.exit(f"no cell has both epochs in the {a.population} population")
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]]
    buf = split["buffer"][cells[:, 0], cells[:, 1]]
    tr, ev = np.where(~held & ~buf)[0], np.where(held)[0]

    ymean = lambda groups: epoch_mean_observed(Xd, groups).astype("float64")    # log1p epoch means
    Y = {h: ymean(g) for h, g in (("full", full), ("a", half_a), ("b", half_b))}
    d = {h: Y[h][pairs[:, 1]] - Y[h][pairs[:, 0]] for h in Y}
    keep = (Y["full"][:, :] > 0).any(0)                      # species seen somewhere in the pairs
    d = {h: v[:, keep] for h, v in d.items()}
    Yb = Y["b"][:, keep]
    species = np.asarray(meta["dev_species"])[keep]
    nz = change_noise(d["full"][ev], d["a"][ev], d["b"][ev])
    cell_noise_b = (d["a"][ev] - d["b"][ev]) ** 2 / 2.0     # a half's noise is twice the full's
    nz_b = {"resolvable": nz["resolvable"], "cell_noise": cell_noise_b,
            "noise": np.nanmean(cell_noise_b, 0)}
    blocks = (cells[ev, 0] // BLOCK) * 100000 + cells[ev, 1] // BLOCK
    d_b_ev = d["b"][ev]
    res = {"cache": a.cache, "population": a.population, "n_cells_train": int(len(tr)),
           "n_cells_eval": int(len(ev)), "n_species": int(keep.sum()),
           "n_resolvable": int(nz["resolvable"].sum()), "arms": {}}

    trend = d["b"][tr].mean(0)
    res["arms"]["trend"] = metrics(np.broadcast_to(trend, d_b_ev.shape), d_b_ev, nz_b, blocks, a.n_boot)

    # level rows: training cells' cell-epochs from trained years (in tempho the early epoch is withheld)
    lvl_groups = np.unique(np.concatenate([pairs[tr, 1]] + ([pairs[tr, 0]] if a.population == "trained"
                                                           else [])))
    z_all = {"desk_raw": np.load(c("z_raw_all.npy"), mmap_mode="r"),
             "desk_ema": np.load(c("z_ema_all.npy"), mmap_mode="r")}
    ci, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    for rank in a.ranks:
        feats = {"esk": (E["z_a"][:, :rank].astype("float64"))}
        for name, za in z_all.items():
            feats[name] = desk_epoch_z(za, ci, y0, keys, full, rank)
        for fname, Fz in feats.items():
            dF = Fz[pairs[:, 1]] - Fz[pairs[:, 0]]
            m_l = blr.fit(Fz[lvl_groups], Yb[lvl_groups], [(0, rank)])
            lvl = dF[ev] @ m_l["coef"].T
            m_c = blr.fit(dF[tr], d["b"][tr], [(0, rank)])
            chg, _ = blr.predict(m_c, dF[ev])
            chg_dev = (dF[ev] - m_c["xbar"]) @ m_c["coef"].T
            for arm, dp in (("lvl", lvl), ("lvl+trend", lvl + trend), ("chg", chg),
                            ("chg_dev", chg_dev)):
                res["arms"][f"{fname}:{arm}:r{rank}"] = metrics(dp, d_b_ev, nz_b, blocks, a.n_boot)
            ledger.log_eval(a.exp, f"change_oracle/{fname}/r{rank}", a.population, "dev",
                            int(nz["resolvable"].sum()), "")
    with open(os.path.join(a.out, "change_oracle.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "change_oracle.json"), encoding="utf-8"))
    f = lambda v: "   -  " if v is None else f"{v:+.3f}"
    print(f"change oracle [{r['population']}] train cells {r['n_cells_train']}, eval cells "
          f"{r['n_cells_eval']}, species {r['n_species']} ({r['n_resolvable']} resolvable); target = "
          f"half B, features = half A")
    print(f"  {'arm':24s} {'skill':>7s} {'CI':>17s} {'captured':>9s} {'atten':>7s}")
    for k, m in r["arms"].items():
        ci = m.get("skill_ci") or [None, None]
        print(f"  {k:24s} {f(m['skill_vs_no_change'])} [{f(ci[0])},{f(ci[1])}] "
              f"{f(m['captured_pooled'])}   {f(m['attenuation_median'])}")


if __name__ == "__main__":
    main()
