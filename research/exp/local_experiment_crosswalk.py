"""Index original sandbox summaries, saved parameters and terminal optimizer traces.

Descriptions and convergence flags are evidence fields, never scientific verdicts.
Missing original per-run source hashes stay unknown rather than being imputed from
the latest version of a frequently edited runner.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


FLAGS = ("form", "field_alpha", "decouple", "estab", "estab_gate", "estab_z", "field_z",
         "edited_inputs", "real_rates", "survanchor", "floor", "dd_info", "bbl_info",
         "vr_alpha_late", "late_from", "train_until", "free_n50")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_for(path, root):
    value = json.loads(path.read_text())
    stem = path.name.removesuffix("_summary.json")
    trace, params = path.with_name(stem + ".jsonl"), path.with_name(stem + "_params.npz")
    last = None
    if trace.is_file():
        for line in reversed(trace.read_text().splitlines()):
            try:
                candidate = json.loads(line)
                if "eval" in candidate:
                    last = candidate
                    break
            except ValueError:
                continue
    row = {"artifact": path.relative_to(root).as_posix(), "summary_sha256": digest(path),
           "original_source_identity": "unknown unless separate original run manifest exists",
           "flags_declared": {key: value[key] for key in FLAGS if key in value},
           "objective": value.get("end_neg_logp"), "bbs_nll": value.get("bbs_nll_end"),
           "optimizer_converged_flag": value.get("converged"), "optimizer_message": value.get("message"),
           "n_eval": value.get("n_eval"), "terminal_evaluation": last,
           "params_present": params.is_file(), "params_sha256": digest(params) if params.is_file() else None,
           "never_reached": value.get("never_reached_end")}
    for key in ("lambda_split_end", "establishment", "disease", "abundance_end"):
        if key not in value:
            continue
        if key == "lambda_split_end":
            row["occupied_east_lambda"] = {period: {"early": info.get("occupied_east", {}).get("q10_50_90_y0"),
                                                    "modern": info.get("occupied_east", {}).get("q10_50_90_2010")}
                                           for period, info in value[key].items()}
        elif key == "abundance_end":
            row["heldout_nll"] = {kind: -sum(info.get(field, 0.) for info in value[key].values())
                                   for kind, field in (("blocks", "loglik_heldout_blocks"),
                                                       ("future", "loglik_heldout_2021"))}
        elif key == "disease" and value[key]:
            row["east_disease_toll"] = value[key].get("by_region", {}).get("east", {}).get("2011-2025", {}).get("realised_mortality")
        elif key == "establishment":
            row["establishment_declared"] = value[key]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()
    rows, errors = [], []
    for path in sorted(args.sandbox.glob("**/*_summary.json")):
        try:
            rows.append(row_for(path, args.sandbox))
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            errors.append({"artifact": path.relative_to(args.sandbox).as_posix(), "error": str(exc)})
    report = {"scope": "original local sandbox artifacts; mixed bases/units, not a ranking",
              "rows": rows, "errors": errors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        columns = ["artifact", "objective", "bbs_nll", "optimizer_converged_flag", "n_eval",
                   "terminal_gradient_norm", "params_present", "never_reached", "form", "field_z",
                   "real_rates", "decouple", "train_until", "free_n50", "summary_sha256", "params_sha256"]
        with args.csv_out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                flat = {key: row.get(key) for key in columns}
                flat.update({key: row["flags_declared"].get(key) for key in
                             ("form", "field_z", "real_rates", "decouple", "train_until", "free_n50")})
                flat["terminal_gradient_norm"] = (row["terminal_evaluation"] or {}).get("grad_norm")
                writer.writerow(flat)
    print(json.dumps({"summaries": len(rows), "errors": len(errors),
                      "with_parameter_checkpoints": sum(row["params_present"] for row in rows),
                      "with_terminal_trace": sum(row["terminal_evaluation"] is not None for row in rows)}))


if __name__ == "__main__":
    main()
