"""Create compact auditable evidence from completed fits and matched readouts."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--readouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case, overlay in json.loads((args.campaign / "campaign_manifest.json").read_text())["cases"]:
        fit_path, readout_path = args.campaign / case / "result.json", args.readouts / case / "readout.json"
        if not fit_path.is_file() or not readout_path.is_file():
            continue
        fit, readout = json.loads(fit_path.read_text()), json.loads(readout_path.read_text())
        params = args.campaign / case / "best_params.npz"
        identity = hashlib.sha256(params.read_bytes()).hexdigest()
        if readout["params_sha256"] != identity:
            raise ValueError(f"Readout/fit checkpoint mismatch for {case}")
        abund = readout["scored_replay"]["abundance"].values()
        scores = {name: -sum(row[key] for row in abund) for name, key in
                  (("regional_bbs_nll", "loglik"), ("future_development_nll", "loglik_heldout_2021"),
                   ("spatial_development_nll", "loglik_heldout_blocks"))}
        row = {"case": case, "overlay": overlay, "params_sha256": identity,
               "fit_result_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
               "readout_sha256": hashlib.sha256(readout_path.read_bytes()).hexdigest(),
               "objective": fit["objective"], "data_nll": fit["data_nll"],
               "total_data_nll": sum(fit["data_nll"].values()),
               "prior_and_regularizer_cost": fit["prior_and_regularizer_cost"],
               "gradient_l2": fit["gradient_l2"], "gradient_max": fit["gradient_max"],
               "optimizer_success": fit["optimizer_success"], "reason": fit["reason"],
               "stationarity_demonstrated": fit["stationarity_demonstrated"],
               "diagnostic_population_max_error": readout["diagnostic_population_max_error"],
               "mortality_numpy_reconstruction_max_error": readout["mortality_numpy_reconstruction_max_error"],
               "prediction": scores, "regions": readout["regions"], "movement": readout["movement"],
               "front_absolute_0p01": readout["scored_replay"]["curves"]["abs_0.01_birds_km2"]["arrival_offset"]}
        rows.append(row)
    payload = {"status": "exploratory capped first-start comparisons; no stationarity or identification claim",
               "field_order": ["Sa", "Fmax", "K", "Sj_ratio"], "rows": rows,
               "score_scope": "total data NLL includes all configured count rows plus nests; regional BBS and development scores use real BBS at 25-49N",
               "validation": "diagnostic scan reproduces original states; mortality independently reconstructed at pre-reproduction density",
               "readout_interpretations": readout["interpretation"] if rows else {},
               "summary_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"cases": [r["case"] for r in rows], "out": str(args.out)}))


if __name__ == "__main__":
    main()
