"""Train DESK: a semi-supervised autoencoder predicting ESK's Z from covariates.

DESK ("Deep ESK") learns to reconstruct the ESK kernel-PCA latent Z -- the
habitat-similarity "ground truth" (for ``off``/``validate`` this is the eBird 2023
spatial Z) -- from covariates that exist for every year, so Z can be extrapolated
across the whole timeline. Trained with three losses: a stabilizing MSE against the
ESK Z where it is known, a metric loss preserving Ruzicka-similarity relationships,
and an autoencoder reconstruction over labeled + unlabeled years.

**Grid-native.** The model (``MultiStreamAutoencoder``) maps a covariate grid
``(B,H,W,C)`` -> latent grid, so its optional spatial residual conv can see each
cell's neighbours. Training therefore operates on whole-year grids, not a shuffled
bag of pixels: the supervised losses gather the valid pixels of the single labelled
year's grid (2023), and every other year's ``state_{year}.npz`` grid feeds the
reconstruction loss (and gives the spatial conv many unlabelled spatial examples to
regularise its filters against -- the labelled signal exists for only one grid).

N-stream: reads ``state_{year}.npz`` (climate/land-use/HYDE/soil/elevation) via
``state_schema.json`` (``covariate_io``). The ``enrich`` mode's multi-year
supervised points are added in ``spacetime_enrich``.
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config_utils import load_config
from . import covariate_io as cio
from .ebird_cache import load_ebird_stack
from .model_arch import MultiStreamAutoencoder


def compute_valid_mask(ebird_stack, cov_stack, z_mask):
    """Intersect finite eBird, finite covariates (all channels), and the ESK-Z mask."""
    m_ebird = np.any(~np.isnan(ebird_stack), axis=-1)
    m_cov = np.all(~np.isnan(cov_stack), axis=-1)
    final = m_ebird & m_cov & z_mask
    print(f"[mask] eBird {m_ebird.sum()} & cov {m_cov.sum()} & Z {z_mask.sum()} "
          f"-> {final.sum()} supervised pixels")
    return final


def _split_mask(mask, train_frac=0.8, seed=0):
    """Split a boolean grid mask's True cells into (train, val) boolean grid masks."""
    ys, xs = np.where(mask)
    g = np.random.default_rng(seed)
    perm = g.permutation(len(ys))
    cut = int(train_frac * len(ys))
    tr = np.zeros_like(mask); va = np.zeros_like(mask)
    tr[ys[perm[:cut]], xs[perm[:cut]]] = True
    va[ys[perm[cut:]], xs[perm[cut:]]] = True
    return tr, va


def true_kernel_loss(z_pred, x_raw, num_pairs=4096):
    """MSE between the dot product in Z and the Ruzicka similarity in raw X, over
    ``num_pairs`` random pairs drawn from the supplied (valid) pixel set."""
    B = z_pred.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)
    idx = torch.randint(0, B, (2, num_pairs), device=z_pred.device)
    i, j = idx[0], idx[1]
    xi, xj = x_raw[i], x_raw[j]
    sum_plus = xi + xj
    diff_abs = torch.abs(xi - xj)
    numerator = 0.5 * torch.sum(sum_plus - diff_abs, dim=1)
    denominator = 0.5 * torch.sum(sum_plus + diff_abs, dim=1)
    valid = denominator > 1e-3
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)
    sim_true = numerator[valid] / (denominator[valid] + 1e-8)
    zi, zj = z_pred[i][valid], z_pred[j][valid]
    sim_pred = (zi * zj).sum(dim=1)
    return F.mse_loss(sim_pred, sim_true)


def _load_hist_grids(states_dir, schema, mu, sd, exclude_year):
    """Preload every ``state_{year}.npz`` grid (except the labelled year) as
    normalized ``(H,W,C)`` tensors + validity masks for the reconstruction loss."""
    grids, masks, years = [], [], []
    for fp in sorted(glob.glob(os.path.join(states_dir, "state_*.npz"))):
        yr = int(os.path.basename(fp).split("_")[1].split(".")[0])
        if yr == exclude_year:
            continue
        covn, m = cio.norm_grid(cio.load_state_stack(yr, states_dir, schema), mu, sd)
        grids.append(torch.tensor(covn)); masks.append(torch.tensor(m)); years.append(yr)
    if not grids:
        return None, None, []
    print(f"[Historical] {len(years)} year grids ({years[0]}..{years[-1]})")
    return torch.stack(grids), torch.stack(masks), years


def train_model_semisup(covn2023, mask_cov, mask_sup_tr, mask_sup_val, z_ref, x_raw_grid,
                        hist_grids, hist_masks, stream_dims, latent_dim, spatial_kernel=3,
                        epochs=100, lr=1e-3, batch_years=8, weights=None, seed=0):
    """Train the N-stream grid DESK autoencoder semi-supervised; return the fitted model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = weights or {"stabilizing": 1.0, "metric": 5.0, "reconstruction": 0.1}

    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    # Supervised (labelled) year grid: forwarded every step (small; dropout varies it).
    cov2023 = covn2023[None].to(device)                 # (1,H,W,C)
    m_cov = mask_cov[None].to(device)                   # (1,H,W)
    z_ref_t = torch.tensor(z_ref, device=device)        # (H,W,L)
    x_t = torch.tensor(x_raw_grid, device=device)       # (H,W, S*T)
    m_tr = torch.as_tensor(mask_sup_tr, device=device).bool()
    m_val = torch.as_tensor(mask_sup_val, device=device).bool()
    m_cov2023 = torch.as_tensor(mask_cov, device=device).bool()

    n_hist = 0 if hist_grids is None else hist_grids.shape[0]
    g = torch.Generator().manual_seed(seed)
    print(f"--- Training grid DESK (spatial_kernel={spatial_kernel}, {n_hist} hist years) ---")

    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_hist, generator=g) if n_hist else torch.zeros(1, dtype=torch.long)
        total_rh, steps = 0.0, 0
        for b0 in range(0, max(n_hist, 1), batch_years):
            steps += 1
            opt.zero_grad()

            # Supervised losses on the labelled (2023) grid.
            z2023, recon2023 = model(cov2023, m_cov)     # (1,H,W,L), (1,H,W,C)
            z_flat = z2023[0]
            loss_stab = torch.mean(torch.sum((z_flat[m_tr] - z_ref_t[m_tr]) ** 2, dim=1))
            loss_true = true_kernel_loss(z_flat[m_tr], x_t[m_tr])
            loss_recon_s = F.mse_loss(recon2023[0][m_cov2023], cov2023[0][m_cov2023])

            # Reconstruction on a batch of unlabelled year grids.
            if n_hist:
                sel = order[b0:b0 + batch_years]
                xb = hist_grids[sel].to(device); mb = hist_masks[sel].to(device).bool()
                _, recon_h = model(xb, mb)
                loss_recon_h = F.mse_loss(recon_h[mb], xb[mb])
            else:
                loss_recon_h = torch.tensor(0.0, device=device)

            loss = (weights["stabilizing"] * loss_stab
                    + weights["metric"] * loss_true
                    + weights["reconstruction"] * (loss_recon_s + loss_recon_h))
            loss.backward()
            opt.step()
            total_rh += loss_recon_h.item()

        model.eval()
        with torch.no_grad():
            z2023, _ = model(cov2023, m_cov)
            zf = z2023[0]
            stab_val = torch.mean(torch.sum((zf[m_val] - z_ref_t[m_val]) ** 2, dim=1)).item()
            cos = F.cosine_similarity(zf[m_val], z_ref_t[m_val]).mean().item()
            true_val = true_kernel_loss(zf[m_val], x_t[m_val]).item()
            gpar = float(model.gamma.detach()) if spatial_kernel > 0 else 0.0
            scheduler.step(stab_val)
            print(f"Ep {ep:03d} | Stab(val) {stab_val:.4f} | True(val) {true_val:.4f} | "
                  f"Rec(H) {total_rh / max(steps,1):.4f} | Cos {cos:.3f} | gamma {gpar:+.4f}")
    return model


def prepare_supervised(cov_stack, ebird_stack, z_flat, z_mask, mu, sd, out_dir):
    """Build the labelled year's grid tensors: normalized covariate grid + cov mask,
    supervised mask (eBird & cov & Z), ESK-Z grid, and raw eBird grid."""
    H, W, _ = cov_stack.shape
    mask_sup = compute_valid_mask(ebird_stack, cov_stack, z_mask)
    np.save(os.path.join(out_dir, "training_mask.npy"), mask_sup)
    covn, mask_cov = cio.norm_grid(cov_stack, mu, sd)
    z_grid = np.zeros((H, W, z_flat.shape[1]), dtype="float32")
    z_grid[z_mask] = z_flat
    x_grid = np.nan_to_num(ebird_stack, nan=0.0).astype("float32")
    return covn, mask_cov, mask_sup, z_grid, x_grid


def run_desk_experiment(config=None):
    """Driver: load N-stream states + ESK Z, prepare grids, train DESK, save model+meta.

    Trains the ``off``/``validate`` model (eBird 2023 spatial ESK Z target, single
    labeled year). The ``enrich`` multi-year path is added by ``spacetime_enrich``.
    """
    config = load_config(config) if not isinstance(config, dict) else config
    paths, desk_cfg = config["paths"], config["desk"]
    out_dir = paths["desk_output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    states_dir = os.path.join(paths["hist_dir"], "yearly_states")
    schema = cio.load_schema(states_dir)
    label_year = int(desk_cfg.get("label_year", 2023))
    spatial_kernel = int(desk_cfg.get("spatial_conv", {}).get("kernel", 3)) \
        if desk_cfg.get("spatial_conv", {}).get("enabled", True) else 0

    ebird_stack, _ = load_ebird_stack(config)
    cov_stack = cio.load_state_stack(label_year, states_dir, schema)

    z_dir = desk_cfg["z_dir"]
    try:
        z_mask = np.load(os.path.join(z_dir, "valid_mask.npy"))
        z_flat = np.load(os.path.join(z_dir, "Z.npy"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ESK Z.npy/valid_mask.npy not in {z_dir}") from exc
    # ESK saves Z at the max swept latent_dim. Optionally truncate to desk.latent_dim:
    # kernel-PCA columns are eigenvalue-ordered, so Z[:, :k] IS the exact dim-k
    # embedding (no ESK re-run needed). Unset -> use all columns.
    ld = desk_cfg.get("latent_dim")
    if ld and z_flat.shape[1] > int(ld):
        print(f"[desk] truncating ESK Z {z_flat.shape[1]} -> {int(ld)} dims (top eigen-components)")
        z_flat = z_flat[:, :int(ld)]

    # Normalization stats: fit on the supervised (labelled) pixels, exactly as before,
    # then applied to every grid (labelled + historical) and frozen for the cube.
    mask_sup0 = compute_valid_mask(ebird_stack, cov_stack, z_mask)
    mu, sd = cio.fit_norm(cov_stack[mask_sup0].astype("float32"))

    covn, mask_cov, mask_sup, z_grid, x_grid = prepare_supervised(
        cov_stack, ebird_stack, z_flat, z_mask, mu, sd, out_dir)
    mask_cov_t = torch.tensor(mask_cov)
    m_tr, m_val = _split_mask(mask_sup, desk_cfg.get("train_val_split", 0.8))

    hist_grids, hist_masks, _ = _load_hist_grids(states_dir, schema, mu, sd, label_year)

    stream_dims = cio.stream_dims(schema)
    model = train_model_semisup(
        torch.tensor(covn), mask_cov_t, m_tr, m_val, z_grid, x_grid,
        hist_grids, hist_masks, stream_dims, latent_dim=z_grid.shape[2],
        spatial_kernel=spatial_kernel,
        epochs=desk_cfg.get("epochs", 100), lr=desk_cfg.get("lr", 1e-3),
        batch_years=desk_cfg.get("batch_years", 8),
        weights=desk_cfg.get("weights"))

    torch.save(model.state_dict(), os.path.join(out_dir, "env_model_semisup.pth"))
    np.savez(os.path.join(out_dir, "desk_meta.npz"),
             mu=mu, sd=sd, stream_dims=np.array(stream_dims, int),
             latent_dim=z_grid.shape[2], label_year=label_year,
             spatial_kernel=spatial_kernel, schema=json.dumps(schema))
    print(f"[desk] saved model + desk_meta.npz -> {out_dir} (spatial_kernel={spatial_kernel})")


if __name__ == "__main__":
    run_desk_experiment()
