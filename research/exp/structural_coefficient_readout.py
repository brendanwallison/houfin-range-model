"""Independent coefficient-geometry accounting for the registered F0/F2 arms."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for directory in sorted(args.campaign.iterdir()):
        if not directory.is_dir() or not (directory / "result.json").is_file():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        cfg = manifest["config"]
        if cfg["population_model"].get("readout", {}).get("split", False):
            raise ValueError("This independent check assumes the registered unsplit readout")
        spec = cfg["population_model"]["manifold_prior"]
        path = directory / "best_params.npz"
        with np.load(path) as params:
            amplitudes = np.logaddexp(params["w_scale_raw"], 0.)
            shared = params["manifold_factor"]
            beta = shared * amplitudes
            norms = {"demography": float(np.linalg.norm(shared))}
            if spec["form"] == "capacity_demography":
                if spec["fixed_capacity_correlation"] != 0.:
                    raise ValueError("Only the registered independent F2 contrast is implemented")
                beta[:, 2] = params["capacity_factor"][:, 0] * amplitudes[2]
                norms["capacity"] = float(np.linalg.norm(params["capacity_factor"]))
            elif spec["form"] != "exchangeable" or spec["fixed_coupling"] != 1.:
                raise ValueError("Not the registered single-field F0 model")
        norm = np.linalg.norm(beta, axis=0)
        cosine = beta.T @ beta / (norm[:, None] * norm[None, :])
        with Path(cfg["input_dir"], "metadata.pkl").open("rb") as f:
            meta = pickle.load(f)
        shape = tuple(int(meta[k]) for k in ("time", "N_land", "M"))
        z = np.memmap(Path(cfg["input_dir"], meta["z_gathered_path"]), dtype=np.float32, mode="r", shape=shape)
        years = np.asarray(meta["years"])
        deltas = {}
        for start, end in ((1966, 2010), (1993, 2010)):
            a, b = [int(np.flatnonzero(years == year)[0]) for year in (start, end)]
            # Centering and intercepts cancel: this is the actual link-predictor change.
            delta = (np.asarray(z[b], np.float64) - np.asarray(z[a], np.float64)) @ beta
            deltas[f"{start}-{end}"] = {"all_land_q10_50_90_by_field": np.quantile(delta, [.1, .5, .9], axis=0).tolist(),
                                        "all_land_rms_by_field": np.sqrt((delta**2).mean(0)).tolist()}
        rows.append({"case": directory.name, "params_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "amplitudes": amplitudes.tolist(), "factor_norms": norms,
                     "effective_coefficient_norms": norm.tolist(), "effective_coefficient_cosines": cosine.tolist(),
                     "link_predictor_deltas": deltas})
    result = {"field_order": ["Sa", "Fmax", "K", "Sj_ratio"], "rows": rows,
              "scope": "Independent reconstruction of registered unsplit F0/F2 effective coefficients; all-land predictor deltas, not changes in vital-rate units",
              "readout_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": [r["case"] for r in rows], "out": str(args.out)}))


if __name__ == "__main__":
    main()
