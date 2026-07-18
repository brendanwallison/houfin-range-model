"""One-shot cache of the reprojected eBird community stack (``E``).

``load_tifs_structured`` reprojects every eBird weekly raster onto the model grid
on *every* ESK/DESK/amplitude call — the same expensive work repeated by ~4
consumers. This caches the reprojected ``(H, W, S*T)`` stack once (NaN preserved —
masking downstream relies on it) plus its metadata, and offers a drop-in
``load_ebird_stack`` returning the same ``(stack, meta)`` contract (cache hit → load;
miss → build/fallback). ``meta`` carries ``species`` (block order) so per-species
BBS anomalies align to the right 52-week block.

    python -m src.community_encoder.train_DESK.ebird_cache      # build the cache
"""
import os

import numpy as np

from src.config_utils import load_config, load_data_config
from src.community_encoder.train_DESK.ebird_io import load_tifs_structured  # torch-free


def _cache_path(config):
    return config.get("bbs", {}).get("ebird_stack_cache") \
        or os.path.join(config["paths"]["desk_output_dir"], "ebird_stack.npz")


def build_ebird_cache(config=None, out_path=None):
    """Reproject the eBird stack once and cache it (NaN preserved). Returns the path."""
    config = load_config(config) if not isinstance(config, dict) else config
    out_path = out_path or _cache_path(config)
    target_res_m = load_data_config()["grid"]["target_res_m"]
    stack, meta = load_tifs_structured(
        config["paths"]["ebird_folder"],
        config["esk"].get("pattern", "*_abundance_median_2023-*.tif"),
        target_res_m=target_res_m,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(  # NOT compressed: preserves NaN + fast to memory-map on load
        out_path, stack=stack.astype(np.float32),
        n_species=meta["n_species"], n_weeks=meta["n_weeks"],
        native_res_m=meta["native_res_m"], target_res_m=meta["target_res_m"] or -1,
        species=np.array(meta["species"], dtype=object),
    )
    print(f"[ebird_cache] cached stack {stack.shape} "
          f"({meta['n_species']} sp x {meta['n_weeks']} wk) -> {out_path}")
    return out_path


def load_ebird_stack(config=None):
    """Return ``(stack (H,W,S*T), meta)`` from the cache, else build/fallback.

    Contract matches ``load_tifs_structured`` (NaN preserved); ``meta`` adds
    ``species``. Consumers that only need the stack ignore the extra key.
    """
    config = load_config(config) if not isinstance(config, dict) else config
    path = _cache_path(config)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        meta = {"n_species": int(z["n_species"]), "n_weeks": int(z["n_weeks"]),
                "native_res_m": float(z["native_res_m"]),
                "target_res_m": (float(z["target_res_m"]) if float(z["target_res_m"]) > 0 else None),
                "species": list(z["species"])}
        return z["stack"], meta
    target_res_m = load_data_config()["grid"]["target_res_m"]
    return load_tifs_structured(
        config["paths"]["ebird_folder"],
        config["esk"].get("pattern", "*_abundance_median_2023-*.tif"),
        target_res_m=target_res_m,
    )


if __name__ == "__main__":
    build_ebird_cache()
