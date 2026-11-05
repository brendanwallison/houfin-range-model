"""Rescore any GRID54 run's refined point on the SDM-baseline scorer (sdm_b0_score.py): T plug-in / marginal /
conditional, S-all, S-real (pseudo-zeros excluded), region and region x era strata, 216 km block-bootstrap intervals
for S-real (overall and by region), and paired differences against a reference run when the designs match.

Normally called by rescore_run.sh STEP (which exports the run's design + predictive with the run's own frozen code).
    python sdm_rescore.py --step step10_pn05 --root /mnt/e/Datasets/houfin/grid54/baselines/rescore [--ref step3_obsroute]
Writes <root>/<step>/rescore.json. Table across steps: --table step3_obsroute step10_pn05 ...
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_b0_score import REGIONS, load_design, paired_bootstrap, step3_lp, summarise  # noqa: E402

RUNS = "/mnt/e/Datasets/houfin/grid54/runs"
MR = "/work/07980/bwa386/ls6/houfin/processed/model_results"


def block_mean_ci(design, v, mask, n=1000, seed=0):
    return paired_bootstrap(design, v, mask, n=n, seed=seed)       # bootstrap of a mean = paired with zero


def load(root, step):
    d = os.path.join(root, step)
    design = load_design(d)
    pr = np.load(os.path.join(d, "pred.npz"))
    pred = {k: pr[k] for k in pr.files}
    man = json.load(open(os.path.join(d, "design_manifest.json")))
    return design, pred, man


def stored_targets(step):
    """The run's own stored held-out numbers (viz_ref/metrics.json holdout) for parity."""
    p = os.path.join(RUNS, step, "out", "viz_ref", "metrics.json")
    if not os.path.exists(p):
        return None
    h = json.load(open(p)).get("holdout") or {}
    t = {}
    if h.get("temporal"):
        t["T_plug"] = h["temporal"]["mean_lpd"]
    if h.get("spatial"):
        t["S_plug"] = h["spatial"]["mean_lpd"]
    oe = h.get("observer_effect") or {}
    if oe.get("temporal"):
        t["T_marg"] = oe["temporal"]["marginal_mean_lpd"]; t["T_cond"] = oe["temporal"]["conditional_mean_lpd"]
    if oe.get("spatial"):
        t["S_marg"] = oe["spatial"]["marginal_mean_lpd"]
    return t


def status(step):
    out = os.path.join(RUNS, step, "out")
    st = {"refinement_done": os.path.exists(os.path.join(out, "REFINEMENT_DONE")),
          "unconverged_flag": os.path.exists(os.path.join(out, "UNCONVERGED_READOUT"))}
    rp = os.path.join(MR, f"refined_float64__age_map_float32_grid54_{step}_quick90", "refine_report.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        st["refine_valid"] = r.get("valid")
        for k in ("final_grad_norm", "grad_norm", "iterations", "n_iter", "reason", "stop_reason"):
            if k in r:
                st[k] = r[k]
    return st


def score(root, step, ref=None):
    design, pred, man = load(root, step)
    lp = step3_lp(design, pred)
    T, S = design["temporal"], design["spatial"]
    Sr = S & ~design["pseudo"]
    got = {"T_plug": lp["plug"][T].mean(), "T_marg": lp["marg"][T].mean(), "T_cond": lp["cond"][T].mean(),
           "S_plug": lp["plug"][S].mean(), "S_marg": lp["marg"][S].mean()}
    tgt = stored_targets(step)
    parity = None
    if tgt:
        parity = {k: float(got[k] - tgt[k]) for k in tgt}
    res = {"step": step, "status": status(step), "manifest": man, "scores": {k: float(v) for k, v in got.items()},
           "parity_vs_stored": parity,
           "parity_pass": None if parity is None else all(abs(v) < 1e-3 for v in parity.values()),
           "summary": summarise(design, lp, pred["mu"])}
    ci = {"S_real_marg": block_mean_ci(design, lp["marg"], Sr), "T_marg": block_mean_ci(design, lp["marg"], T),
          "T_cond": block_mean_ci(design, lp["cond"], T)}
    for rn, lo, hi in REGIONS:
        m = Sr & (design["lon"] >= lo) & (design["lon"] < hi)
        ci[f"S_real_marg|{rn}"] = block_mean_ci(design, lp["marg"], m)
    res["block_ci"] = ci
    if ref and ref != step and os.path.exists(os.path.join(root, ref, "pred.npz")):
        rd, rp_, rman = load(root, ref)
        same = rman["design_sha256"] == man["design_sha256"]
        res["paired_vs_ref"] = {"ref": ref, "same_design": same}
        if same:
            rlp = step3_lp(rd, rp_)
            pv = {}
            for name, m, kind in (("T_marg", T, "marg"), ("T_cond", T, "cond"), ("S_real_marg", Sr, "marg"),
                                  ("S_all_marg", S, "marg")):
                pv[name] = paired_bootstrap(design, lp[kind] - rlp[kind], m)
            for rn, lo, hi in REGIONS:
                m = Sr & (design["lon"] >= lo) & (design["lon"] < hi)
                pv[f"S_real_marg|{rn}"] = paired_bootstrap(design, lp["marg"] - rlp["marg"], m)
            res["paired_vs_ref"]["diff"] = pv
    json.dump(res, open(os.path.join(root, step, "rescore.json"), "w"), indent=1)
    return res


def table(root, steps):
    print("| step | status | parity | T plug | T marg | T cond | S-all marg | S-real marg [90% CI] | west | plains | midwest | east | vs ref S-real [CI] | vs ref T marg [CI] |")
    print("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|")
    for s in steps:
        p = os.path.join(root, s, "rescore.json")
        if not os.path.exists(p):
            print(f"| {s} | missing |"); continue
        r = json.load(open(p))
        sc, su, ci = r["scores"], r["summary"], r["block_ci"]
        st = r["status"]
        flag = "UNCONVERGED" if st.get("unconverged_flag") or not st.get("refinement_done") else ("valid" if st.get("refine_valid") else str(st.get("refine_valid")))
        par = "n/a" if r["parity_pass"] is None else ("ok" if r["parity_pass"] else "FAIL " + json.dumps({k: round(v, 4) for k, v in r["parity_vs_stored"].items()}))
        reg = su["S_real"]["by_region"]
        c = ci["S_real_marg"]
        pv = (r.get("paired_vs_ref") or {}).get("diff")
        if pv:
            a, b = pv["S_real_marg"], pv["T_marg"]
            ref_s = f"{a['mean']:+.3f} [{a['lo90']:+.3f}, {a['hi90']:+.3f}]"
            ref_t = f"{b['mean']:+.3f} [{b['lo90']:+.3f}, {b['hi90']:+.3f}]"
        else:
            same = (r.get("paired_vs_ref") or {}).get("same_design")
            ref_s = ref_t = "--" if same is None else ("design differs" if same is False else "--")
        print(f"| {s} | {flag} | {par} | {sc['T_plug']:.3f} | {sc['T_marg']:.3f} | {sc['T_cond']:.3f} | {sc['S_marg']:.3f} | "
              f"{su['S_real']['marg']:.3f} [{c['lo90']:.3f}, {c['hi90']:.3f}] | "
              + " | ".join(f"{reg[k]['marg']:.3f}" if k in reg else "--" for k in ("west", "plains", "midwest", "east"))
              + f" | {ref_s} | {ref_t} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/Datasets/houfin/grid54/baselines/rescore")
    ap.add_argument("--step")
    ap.add_argument("--ref", default="step3_obsroute")
    ap.add_argument("--table", nargs="*")
    a = ap.parse_args()
    if a.step:
        r = score(a.root, a.step, a.ref)
        print(json.dumps({k: r[k] for k in ("step", "status", "scores", "parity_vs_stored", "parity_pass")}, indent=1))
    if a.table:
        table(a.root, a.table)


if __name__ == "__main__":
    main()
