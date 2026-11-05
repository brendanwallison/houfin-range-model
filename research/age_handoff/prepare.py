"""Generate isolated final-DESK age-model configs against the frozen age-model source."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / "config/overlays"
CFG.mkdir(parents=True, exist_ok=True)
P = "${HOUFIN_PROCESSED}"
for family, width, kernel in [("v1", 64, "ruzicka"), ("v2", 128, "ruzicka_sum")]:
    tag = f"fd_{family}"
    cube = f"cube_desk_final{'_v2' if family == 'v2' else ''}_mean4_diag"
    config = {"_extends": "map_build_vl_r64.json",
              "_comment": "Diagnostic final-candidate mean FEATURES, shared hashed coordinates; existing contaminated-completeness candidates, not production adoption.",
              "path_features": {"input_dir": f"{P}/encoder/{cube}", "output_dir": f"{P}/latent_avian_paths_{tag}"},
              "raw_z_dir": f"{P}/latent_avian_paths_{tag}", "input_dir": f"{P}/model_inputs/numpyro_input_{tag}",
              "latent_dim": width, "source_latent_dim": width,
              "kernel_contract": {"kernel": kernel, "centered": False, "feature_prior": "isotropic"}}
    (CFG / f"map_build_{tag}.json").write_text(json.dumps(config, indent=2) + "\n")

for arm, source, source_width, rank, columns, kernel in [
    ("fd_control64", "cbc1902_r64", 64, 64, None, "ruzicka"),
    ("fd_v1_r64", "fd_v1", 64, 64, None, "ruzicka"),
    ("fd_v2_r128", "fd_v2", 128, 128, None, "ruzicka_sum"),
    ("fd_v2_r64", "fd_v2", 128, 64, list(range(32)) + list(range(64, 96)), "ruzicka_sum")]:
    config = {"_extends": "map_run80h_temporal.json",
              "_comment": "Frozen correlated spatial/temporal readout and biology; equal isotropic coefficient variance across summer/winter. Only encoder representation and declared width/columns differ.",
              "input_dir": f"{P}/model_inputs/numpyro_input_{source}",
              "latent_dim": source_width, "source_latent_dim": source_width,
              "kernel_contract": {"kernel": kernel, "centered": False, "feature_prior": "isotropic"},
              "population_model": {"readout": {"rank": rank, "feature_columns": columns}},
              "run_names": {"map": f"age_map_{{precision}}_{arm}", "vis_map": f"age_map_{{precision}}_{arm}"}}
    (CFG / f"map_run_{arm}.json").write_text(json.dumps(config, indent=2) + "\n")
print("generated", len(list(CFG.glob("*.json"))), "handoff overlays")
