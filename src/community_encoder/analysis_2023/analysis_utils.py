"""Small I/O and plotting helpers shared by the 2023 single-year diagnostics.

Load masks, latent (Z) matrices, and response rasters into aligned arrays, and
save figures. Z is the ESK/DESK latent habitat-quality space (ESK = kernel-PCA on
eBird community similarity; DESK = autoencoder predicting Z from environment);
these are analysis utilities over Z, not part of encoder training.
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
import matplotlib.pyplot as plt


def load_mask(mask_path: str) -> np.ndarray:
    """Load a saved .npy mask and return it as a boolean array."""
    mask = np.load(mask_path)
    return mask.astype(bool)


def load_latent_matrix(path: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Load a latent array as a 2-D (n_pixels, latent_dim) matrix.

    A 3-D (H, W, D) cube is flattened to rows; if a 2-D ``mask`` is given, only
    masked pixels are kept. An already-2-D array is returned unchanged.
    """
    latent = np.load(path)
    if latent.ndim == 3:
        if mask is None:
            return latent.reshape(-1, latent.shape[-1])
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
        return latent[mask].reshape(-1, latent.shape[-1])
    return latent


def load_yearly_latent_slice(cube_path: str, year: int = 2023) -> np.ndarray:
    """Load a 3-D (H, W, D) latent cube and return it as-is.

    Raises if the array is not 3-D. ``year`` is currently unused.
    """
    cube = np.load(cube_path)
    if cube.ndim == 3:
        return cube
    raise ValueError(f"Expected a 3D latent cube, found shape {cube.shape}")


def load_response_raster(path: str) -> np.ndarray:
    """Read band 1 of a raster as a float32 2-D array."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def align_response_to_mask(response_map: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract response values at masked pixels.

    Returns the values under ``mask`` (flattened) and their flat indices into the
    raveled grid.
    """
    flat_mask = mask.ravel()
    flat_response = response_map.ravel()
    values = flat_response[flat_mask]
    return values, np.where(flat_mask)[0]


def build_output_dir(base_dir: str, name: str) -> str:
    """Create ``base_dir/name`` (if needed) and return the path."""
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_image(path: str, fig=None) -> None:
    """Tight-layout, save (dpi=200), and close the figure (current figure if none)."""
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
