import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
import matplotlib.pyplot as plt


def load_mask(mask_path: str) -> np.ndarray:
    mask = np.load(mask_path)
    return mask.astype(bool)


def load_latent_matrix(path: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
    latent = np.load(path)
    if latent.ndim == 3:
        if mask is None:
            return latent.reshape(-1, latent.shape[-1])
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
        return latent[mask].reshape(-1, latent.shape[-1])
    return latent


def load_yearly_latent_slice(cube_path: str, year: int = 2023) -> np.ndarray:
    cube = np.load(cube_path)
    if cube.ndim == 3:
        return cube
    raise ValueError(f"Expected a 3D latent cube, found shape {cube.shape}")


def load_response_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def align_response_to_mask(response_map: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat_mask = mask.ravel()
    flat_response = response_map.ravel()
    values = flat_response[flat_mask]
    return values, np.where(flat_mask)[0]


def build_output_dir(base_dir: str, name: str) -> str:
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_image(path: str, fig=None) -> None:
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
