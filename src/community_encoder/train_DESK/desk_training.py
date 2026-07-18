"""Train DESK: a semi-supervised autoencoder predicting ESK's Z from covariates.

DESK ("Dynamic ESK") learns to reconstruct the ESK kernel-PCA latent Z -- the
habitat-similarity "ground truth" (for ``off``/``validate`` this is the eBird 2023
spatial Z) -- from covariates that exist for every year, so Z can be extrapolated
across the whole timeline. Trained with three losses: a stabilizing MSE against the
ESK Z where it is known, a metric loss preserving Ruzicka-similarity relationships,
and an autoencoder reconstruction over labeled + unlabeled years.

N-stream: reads the continental ``state_{year}.npz`` (climate/land-use/HYDE/soil/
elevation) via ``state_schema.json`` (``covariate_io``) and trains an N-branch
``MultiStreamAutoencoder`` -- replacing the deprecated 2-stream PRISM/BUI wiring.
The ``enrich`` mode's multi-year supervised points are added in ``spacetime_enrich``.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


class PixelDataset(Dataset):
    """Supervised rows: (covariate C-vector, ESK-Z target, raw eBird community)."""

    def __init__(self, cov, z, x, split="train", train_val_split=0.8, seed=0):
        self.cov, self.z, self.x = cov, z, x
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(cov.shape[0], generator=g)
        cut = int(train_val_split * cov.shape[0])
        self.idx = perm[:cut] if split == "train" else perm[cut:]
        print(f"[Supervised-{split}] N={len(self.idx)}")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.cov[j], self.z[j], self.x[j]


class HistoricalDataset(Dataset):
    """Unsupervised covariate bag (all sampled years) for the reconstruction loss."""

    def __init__(self, path, schema, mu, sd):
        bag = cio.transform_flat(np.load(path), schema)
        cov = cio.apply_norm(bag, mu, sd)
        self.cov = torch.tensor(cov, dtype=torch.float32)
        print(f"[Historical] N={len(self.cov)}")

    def __len__(self):
        return len(self.cov)

    def __getitem__(self, idx):
        return self.cov[idx]


def true_kernel_loss(z_pred, x_raw, num_pairs=4096):
    """MSE between the dot product in Z and the Ruzicka similarity in raw X."""
    B = z_pred.shape[0]
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


def _streams(cov_batch, schema):
    """Split a (B, C) covariate batch into the per-stream tensors the model expects."""
    return cio.split_streams(cov_batch, schema)


def train_model_semisup(train_ds, val_ds, hist_ds, schema, stream_dims, latent_dim,
                        epochs=50, lr=1e-3, batch_size=4096, weights=None):
    """Train the N-stream DESK autoencoder semi-supervised; return the fitted model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = weights or {"stabilizing": 1.0, "metric": 5.0, "reconstruction": 0.1}

    model = MultiStreamAutoencoder(stream_dims, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    loader_sup = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_unsup = DataLoader(hist_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    iter_unsup = iter(loader_unsup)
    print("--- Starting Semi-Supervised Training (N-stream) ---")

    for ep in range(1, epochs + 1):
        model.train()
        total_recon_hist, steps = 0.0, 0
        for cov, z_ref, x_raw in loader_sup:
            steps += 1
            cov, z_ref, x_raw = cov.to(device), z_ref.to(device), x_raw.to(device)
            try:
                cov_hist = next(iter_unsup)
            except StopIteration:
                iter_unsup = iter(loader_unsup)
                cov_hist = next(iter_unsup)
            cov_hist = cov_hist.to(device)

            opt.zero_grad()
            z_pred, recon = model(*_streams(cov, schema))
            loss_stab = torch.mean(torch.sum((z_pred - z_ref) ** 2, dim=1))
            loss_true = true_kernel_loss(z_pred, x_raw)
            loss_recon_sup = F.mse_loss(recon, cov)

            _, recon_hist = model(*_streams(cov_hist, schema))
            loss_recon_hist = F.mse_loss(recon_hist, cov_hist)

            loss = (weights["stabilizing"] * loss_stab
                    + weights["metric"] * loss_true
                    + weights["reconstruction"] * (loss_recon_sup + loss_recon_hist))
            loss.backward()
            opt.step()
            total_recon_hist += loss_recon_hist.item()

        model.eval()
        with torch.no_grad():
            cov_v, z_v, x_v = next(iter(DataLoader(val_ds, batch_size=batch_size)))
            cov_v, z_v, x_v = cov_v.to(device), z_v.to(device), x_v.to(device)
            z_pred, recon = model(*_streams(cov_v, schema))
            loss_recon = F.mse_loss(recon, cov_v).item()
            loss_stab = torch.mean(torch.sum((z_pred - z_v) ** 2, dim=1)).item()
            loss_true = true_kernel_loss(z_pred, x_v).item()
            cos = F.cosine_similarity(z_pred, z_v).mean().item()
            scheduler.step(loss_stab)
            print(f"Ep {ep:03d} | Stab {loss_stab:.4f} | True {loss_true:.4f} | "
                  f"Rec(S/H) {loss_recon:.4f}/{total_recon_hist / max(steps,1):.4f} | Cos {cos:.3f}")
    return model


def prepare_supervised(cov_stack, ebird_stack, z_flat, z_mask, out_dir):
    """Flatten covariate/eBird/Z stacks to aligned per-pixel tensors on a shared mask."""
    H, W, _ = cov_stack.shape
    mask = compute_valid_mask(ebird_stack, cov_stack, z_mask)
    np.save(os.path.join(out_dir, "training_mask.npy"), mask)

    cov = cov_stack[mask].astype("float32")
    x = np.nan_to_num(ebird_stack[mask], nan=0.0).astype("float32")
    Z_grid = np.zeros((H, W, z_flat.shape[1]), dtype="float32")
    Z_grid[z_mask] = z_flat
    z = Z_grid[mask]
    return cov, z, x, mask


def run_desk_experiment(config=None):
    """Driver: load N-stream states + ESK Z, prepare tensors, train DESK, save model+meta.

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

    ebird_stack, _ = load_ebird_stack(config)
    cov_stack = cio.load_state_stack(label_year, states_dir, schema)

    z_dir = desk_cfg["z_dir"]
    try:
        z_mask = np.load(os.path.join(z_dir, "valid_mask.npy"))
        z_flat = np.load(os.path.join(z_dir, "Z.npy"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ESK Z.npy/valid_mask.npy not in {z_dir}") from exc

    cov, z, x, _ = prepare_supervised(cov_stack, ebird_stack, z_flat, z_mask, out_dir)
    mu, sd = cio.fit_norm(cov)
    cov_n = cio.apply_norm(cov, mu, sd)

    tv = desk_cfg.get("train_val_split", 0.8)
    train_ds = PixelDataset(torch.tensor(cov_n), torch.tensor(z), torch.tensor(x), "train", tv)
    val_ds = PixelDataset(torch.tensor(cov_n), torch.tensor(z), torch.tensor(x), "val", tv)
    hist_ds = HistoricalDataset(os.path.join(paths["hist_dir"], "history_vectors.npy"),
                                schema, mu, sd)

    stream_dims = cio.stream_dims(schema)
    model = train_model_semisup(
        train_ds, val_ds, hist_ds, schema, stream_dims, latent_dim=z.shape[1],
        epochs=desk_cfg.get("epochs", 100), lr=desk_cfg.get("lr", 1e-3),
        batch_size=desk_cfg.get("batch_size", 4096),
        weights=desk_cfg.get("weights"))

    torch.save(model.state_dict(), os.path.join(out_dir, "env_model_semisup.pth"))
    np.savez(os.path.join(out_dir, "desk_meta.npz"),
             mu=mu, sd=sd, stream_dims=np.array(stream_dims, int),
             latent_dim=z.shape[1], label_year=label_year,
             schema=json.dumps(schema))
    print(f"[desk] saved model + desk_meta.npz -> {out_dir}")


if __name__ == "__main__":
    run_desk_experiment()
