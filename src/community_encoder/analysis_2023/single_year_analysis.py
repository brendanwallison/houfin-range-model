"""Single-year (2023) regression of a response raster on the latent Z.

Fits a response field (e.g. House Finch relative abundance) from the ESK/DESK
latent habitat space via closed-form Bayesian linear regression, saving
fit/residual diagnostics.
"""
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from .analysis_utils import (
    align_response_to_mask,
    build_output_dir,
    load_latent_matrix,
    load_mask,
    load_response_raster,
    save_image,
)
from .config_utils import load_config


def run_bayesian_regression(z: np.ndarray, y: np.ndarray, out_dir: str) -> Dict[str, float]:
    """Closed-form Bayesian (ridge) regression of ``y`` on latent ``z``.

    Fits the posterior-mean weights, saves ``regression_fit.png`` (predicted-vs-
    observed + residual histogram) to ``out_dir``, and returns {rmse, r2}.
    """
    alpha = 1.0
    sigma2 = np.var(y) * 0.1 + 1e-6

    A = alpha * np.eye(z.shape[1]) + (z.T @ z) / sigma2
    A_inv = np.linalg.inv(A)
    w_mean = A_inv @ (z.T @ y) / sigma2

    y_pred = z @ w_mean
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    r2 = float(1.0 - np.var(residuals) / np.var(y))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred, y, alpha=0.3)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "--", color="black")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Regression Fit")

    axes[1].hist(residuals, bins=40)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")
    save_image(os.path.join(out_dir, "regression_fit.png"), fig)

    return {"rmse": rmse, "r2": r2}


def run_single_year_analysis(config: Optional[Union[Dict[str, Any], str, os.PathLike]] = None) -> Dict[str, Any]:
    """Load Z under the mask, align a response raster, and run the regression.

    Resolves paths from the ``single_year_analysis``/``paths`` config, keeps
    finite aligned rows, and calls :func:`run_bayesian_regression`. Returns
    {out_dir, rmse, r2}.
    """
    if config is None:
        config = load_config()
    elif isinstance(config, (str, os.PathLike)):
        config = load_config(config)

    analysis_cfg = config.get("single_year_analysis", {})
    paths = config.get("paths", {})

    z_path = (
        analysis_cfg.get("esk_feature_path")
        or analysis_cfg.get("esk_z_path")
        or os.path.join(paths.get("esk_output_dir", ""), "Z.npy")
    )
    mask_path = analysis_cfg.get("mask_path") or os.path.join(paths.get("esk_output_dir", ""), "valid_mask.npy")
    response_path = analysis_cfg.get("response_path")
    out_dir = analysis_cfg.get("output_dir") or build_output_dir(paths.get("desk_output_dir", ""), "single_year_analysis")
    os.makedirs(out_dir, exist_ok=True)

    if not response_path:
        raise ValueError("A response raster path is required for single-year analysis")

    mask = load_mask(mask_path)
    z = load_latent_matrix(z_path, mask=mask)
    response = load_response_raster(response_path)

    y, _ = align_response_to_mask(response, mask)
    valid = np.isfinite(y) & np.all(np.isfinite(z), axis=1)
    z_use = z[valid]
    y_use = y[valid]

    if z_use.shape[0] == 0:
        raise ValueError("No valid aligned rows remain for regression")

    metrics = run_bayesian_regression(z_use, y_use, out_dir)
    return {"out_dir": out_dir, **metrics}
