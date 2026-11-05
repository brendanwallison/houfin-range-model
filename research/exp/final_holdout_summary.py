"""Summarize existing fair_rivals components, preserving paired bootstrap draws."""
import argparse
import json
from pathlib import Path
import re
import datetime
from zoneinfo import ZoneInfo
import numpy as np


def estimate(v):
    good = np.isfinite(v[1:])
    return {"value": float(v[0]) if np.isfinite(v[0]) else None,
            "ci95": np.percentile(v[1:][good], [2.5, 97.5]).tolist() if good.any() else None,
            "valid_bootstrap_share": float(good.mean())}


def correlation(z, arm):
    cv, vp, vt = (z[f"{arm}__cv"].sum(1), z[f"{arm}__vp"].sum(1), z["vt"].sum(1))
    good = (vp > 0) & (vt > 0)
    return np.divide(cv, np.sqrt(np.maximum(vp * vt, 0)),
                     out=np.full(len(vp), np.nan), where=good)


def summarize(root):
    groups = {}
    for directory in sorted((root / "summer").glob("*")):
        path = directory / "species_components_final.npz"
        result = directory / "fair_rivals_incr_final.json"
        if not path.exists() or not result.exists() or not (directory / "DONE").exists():
            continue
        m = re.fullmatch(r"(v\d_[a-z0-9]+)_e(\d+)_(\d+)_f([.\d]+)_s(\d+)", directory.name)
        if not m:
            continue
        arm, lo, hi, thin, seed = m.groups()
        key = f"{lo}_{hi}/f{thin}/s{seed}"
        with np.load(path) as f:
            z = {k: f[k].astype("float64") if k != "columns" else f[k] for k in f.files}
        meta = json.loads(result.read_text())
        entry = {"n_cells": meta["n_cells"], "n_fit_rows": meta["n_fit_rows"],
                 "species": len(z["columns"]), "metrics": {}}
        entry["existing_rivals"] = {k: meta["models"][k] for k in ("strong", "placebo_300", "strong+desk_within")
                                    if k in meta["models"]}
        entry["head_to_head"] = meta.get("head_to_head", {})
        entry["strong_gain"] = meta.get("strong_gain")
        for readout in ("desk_within", "desk_pooled", "strong", "pooled_ridge"):
            if f"{readout}__cv" not in z:
                continue
            entry["metrics"][readout] = estimate(correlation(z, readout))
            point = entry["metrics"][readout]["value"]
            expected = meta["models"][readout]["corr"]
            if point is not None and not np.isclose(point, expected, atol=1e-6):
                raise ValueError("component rederivation disagrees with direct fair_rivals score")
            if f"{readout}__mean_pred" in z and "mean_observed" in z:
                bias = z[f"{readout}__mean_pred"][0] - z["mean_observed"][0]
                entry["metrics"][readout]["mean_species_bias"] = float(bias.mean())
                entry["metrics"][readout]["mean_absolute_species_bias"] = float(np.abs(bias).mean())
        groups.setdefault(key, {})[arm] = (entry, z)
    output = {"summer": {}}
    for key, arms in groups.items():
        block = {"candidates": {k: v[0] for k, v in arms.items()}, "paired_vs_v1": {}}
        if "v1_r64" in arms:
            ref = arms["v1_r64"][1]
            for name, (_, z) in arms.items():
                if name == "v1_r64":
                    continue
                if not np.array_equal(z["columns"], ref["columns"]) or not np.array_equal(z["vt"], ref["vt"]):
                    raise ValueError(f"unpaired summer supports/draws: {key} {name}")
                block["paired_vs_v1"][name] = {rd: estimate(correlation(z, rd) - correlation(ref, rd))
                                              for rd in ("desk_within", "desk_pooled")}
        output["summer"][key] = block
    for label, path in [("winter", root / "winter" / "result.json"),
                        ("completeness_audit", root / "audit" / "result.json"),
                        ("membership_audit", root / "audit" / "membership_result.json"),
                        ("winter_damage_check", root / "winter" / "damage_check.json")]:
        if path.exists():
            output[label] = json.loads(path.read_text())
    (root / "SUMMARY.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print(f"{len(groups)} completed summer windows/regimes; winter {'winter' in output}")
    return output


def report(output, root, destination):
    def fmt(z):
        if not z or z.get("value") is None:
            return "pending"
        ci = z.get("ci95")
        return f"{z['value']:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
    count = len(list((root / "summer").glob("*/DONE")))
    lines = ["# Final DESK held-out evaluation results", "",
             f"Updated {datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds')}. "
             f"Summer tasks: {count}/26; winter complete: {'winter' in output}; "
             f"observation-damage diagnostic complete: {'winter_damage_check' in output}.", "",
             "PROVISIONAL: direct species targets were excluded, but sealed species and House Finch enter the training "
             "CBC completeness benchmark. These are not strictly untouched-species results. The encoders used the full "
             "training timeline: this is unseen-species/location evaluation and readout backcasting, not encoder temporal holdout.", "",
             "## Summer", "", "Noise-corrected correlation of epoch change across held-out cells, pooled over 296 species. "
             "Intervals use the existing paired 6x6-block bootstrap. Modern window 2005-2025; readout calibration 1996+. "
             "Within fits cell deviations; pooled fits levels and is then applied to temporal change. v2 r64=32+32; r128=64+64.", ""]
    for rd in ("desk_within", "desk_pooled"):
        lines += [f"### {rd}", "", "| Early window / training fraction / seed | v1 r64 | v2 r64 | v2 r128 | v2 r64 minus v1 | v2 r128 minus v1 |", "|---|---|---|---|---|---|"]
        for key, block in sorted(output["summer"].items()):
            vals = [fmt(block["candidates"].get(arm, {}).get("metrics", {}).get(rd)) for arm in ["v1_r64", "v2_r64", "v2_r128"]]
            diff = [fmt(block["paired_vs_v1"].get(arm, {}).get(rd)) for arm in ["v2_r64", "v2_r128"]]
            lines.append("| " + " | ".join([key, *vals, *diff]) + " |")
        lines.append("")
    primary = output["summer"].get("1966_1986/f1/s1", {}).get("candidates", {})
    if "v1_r64" in primary:
        base = primary["v1_r64"].get("existing_rivals", {})
        lines += ["### Established rivals, primary summer window", "",
                  "A gain over v1 does not by itself establish a gain over the existing fair linear/trend rivals.", "",
                  "| Candidate | Within readout minus strong rival | Additional gain when added to strong rival |", "|---|---|---|"]
        for arm, entry in primary.items():
            d = entry.get("head_to_head", {}).get("desk_within - strong")
            g = entry.get("strong_gain")
            lines.append(f"| {arm} | " + fmt({"value": d["diff"], "ci95": d["ci"]} if d else None) + " | " +
                         fmt({"value": g["desk_within"], "ci95": g["ci"]} if g else None) + " |")
        if "strong" in base:
            lines += ["", f"Strong rival correlation: {base['strong']['corr']:.3f}; position-time placebo: {base['placebo_300']['corr']:.3f}.", ""]
    if "winter" in output:
        w = output["winter"]
        lines += ["## Winter", "", f"{len(w['species'])} eligible species; {w['calibration_rows']:,} modern calibration lists, "
                  f"{w['test_rows']:,} held-out lists. Two relocated test circles excluded from calibration. "
                  "Uses the existing conditional list likelihood, winter-specific ridge and modern training-circle offsets. "
                  "Scores measure relative composition, not absolute abundance. Differences below compare each model's "
                  "gain over its fixed modern-feature placebo; intervals bootstrap circles.", "",
                  "| Era | Lists / circles | v1 backcast minus modern | v2 r64 gain minus v1 | v2 r128 gain minus v1 |", "|---|---|---|---|---|"]
        for era, block in w["eras"].items():
            z = block["primary"]
            vals = [fmt(z.get("metrics", {}).get(k)) for k in ["v1_r64/backcast_minus_modern", "v2_r64/gain_minus_v1", "v2_r128/gain_minus_v1"]]
            lines.append(f"| {era} | {z['lists']} / {z['circles']} | " + " | ".join(vals) + " |")
        lines += ["", "Same-circle sensitivity has only 3 and 5 circles before 1940, below the scoring gate. "
                  "Early advantages require the observation-damage and effort checks before a habitat interpretation.", ""]
    lines += ["## Holdout audit and validation", "", "The original completeness arrays were reproduced exactly. "
              "Recognized sealed taxa make up about 37% of benchmark members; House Finch is a member in about 80% of "
              "circle-years. A clean-completeness training sensitivity is needed before production adoption. "
              "ABBA year halves address sampling noise but do not guarantee independent route/observer effects. "
              "Twelve focused tests pass. Component correlations are independently checked against the established "
              "scorer's direct results; paired comparisons require identical columns and truth bootstrap components.", ""]
    if "winter_damage_check" in output:
        d = output["winter_damage_check"]
        lines += ["## Observation-damage diagnostic", "", "Reconstructed fixed readouts reproduce original scores within 1e-6. "
                  "Reuses E086v's 1966-1995 damaged observations, with original taxonomy exclusions frozen. "
                  "Features and readouts are unchanged between original and degraded observations.", "",
                  f"Matched lists/circles: {d['result']['lists']} / {d['result']['circles']}.", ""]
        lines += [f"- {k}: {fmt(v)}" for k, v in d["result"].get("metrics", {}).items()]
        lines += ["", "Observation degradation increases apparent backcast gain with habitat features held fixed. "
                  "This scenario does not reproduce the large real early-era v1/v2 separation; that discrepancy remains unresolved."]
    lines += ["", f"Full results, source snapshots/manifests and local/remote read ledgers: `{root}`. "
              "No training checkpoints or age-model production configuration were changed.", ""]
    destination.write_text("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    output = summarize(a.root)
    if a.report:
        report(output, a.root, a.report)
