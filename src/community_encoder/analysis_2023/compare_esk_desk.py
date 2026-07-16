"""Diagnostic comparing the ESK and DESK latent spaces for a single year.

ESK is the kernel-PCA embedding of eBird community similarity (Z); DESK is the
autoencoder that predicts Z from environmental covariates. This scatters and
histograms the two Z matrices to check how closely DESK reproduces ESK.
"""
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from .analysis_utils import build_output_dir, load_latent_matrix, load_mask, save_image
from .config_utils import load_config


def compare_esk_desk(config: Optional[Union[Dict[str, Any], str, os.PathLike]] = None) -> Dict[str, Any]:
    """Load ESK and DESK Z under the shared mask, compare, and save a figure.

    Resolves paths from the ``single_year_analysis`` config block, keeps rows
    finite in both, correlates dim 0, and writes ``esk_desk_comparison.png``
    (dim-0 scatter + pointwise-distance histogram). Returns {out_dir, corr_dim0}.
    """
    if config is None:
        config = load_config()
    elif isinstance(config, (str, os.PathLike)):
        config = load_config(config)

    analysis_cfg = config.get("single_year_analysis", {})
    out_dir = analysis_cfg.get("comparison_output_dir") or build_output_dir(analysis_cfg.get("output_dir") or "", "esk_desk_comparison")
    os.makedirs(out_dir, exist_ok=True)

    esk_path = analysis_cfg.get("esk_feature_path") or analysis_cfg.get("esk_z_path")
    desk_path = analysis_cfg.get("desk_feature_path") or analysis_cfg.get("desk_z_path")
    mask_path = analysis_cfg.get("mask_path")

    if not esk_path or not desk_path or not mask_path:
        raise ValueError("esk_feature_path/esk_z_path, desk_feature_path/desk_z_path, and mask_path must be set for comparison")

    mask = load_mask(mask_path)
    esk = load_latent_matrix(esk_path, mask=mask)
    desk = load_latent_matrix(desk_path, mask=mask)

    if esk.shape[0] != desk.shape[0]:
        raise ValueError(f"ESK and DESK row counts do not match: {esk.shape[0]} vs {desk.shape[0]}")

    valid = np.isfinite(esk).all(axis=1) & np.isfinite(desk).all(axis=1)
    esk = esk[valid]
    desk = desk[valid]

    corr = np.corrcoef(esk[:, 0], desk[:, 0])[0, 1] if min(esk.shape[1], desk.shape[1]) > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(esk[:, 0], desk[:, 0], alpha=0.2)
    axes[0].set_xlabel("ESK dim 0")
    axes[0].set_ylabel("DESK dim 0")
    axes[0].set_title("ESK vs DESK first dimension")

    axes[1].hist(np.linalg.norm(esk - desk, axis=1), bins=40)
    axes[1].set_xlabel("Pointwise distance")
    axes[1].set_title("Latent-space distance")
    save_image(os.path.join(out_dir, "esk_desk_comparison.png"), fig)

    return {"out_dir": out_dir, "corr_dim0": float(corr)}
