import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config_utils import load_config
from .esk_kernel import load_tifs_structured
from .model_arch import BMLPBlock, MultiInputAutoencoder


def compute_strict_mask(ebird_stack, prism_stack, bui_stack):
    """Implements the strict union/intersection logic."""
    print("Computing strict validity mask...")

    mask_ebird = np.any(~np.isnan(ebird_stack), axis=-1)
    print(f" -> eBird valid pixels: {np.sum(mask_ebird)}")

    mask_prism = np.all(~np.isnan(prism_stack), axis=-1)
    print(f" -> PRISM valid pixels: {np.sum(mask_prism)}")

    mask_bui = np.all(~np.isnan(bui_stack), axis=-1)
    print(f" -> BUI valid pixels:   {np.sum(mask_bui)}")

    final_mask = mask_ebird & mask_prism & mask_bui
    print(f" -> FINAL INTERSECTION: {np.sum(final_mask)}")
    return final_mask


class PixelDataset(Dataset):
    """Standard Supervised Dataset (2023 Data)."""

    def __init__(self, p_flat, b_flat, z_flat, x_flat, split="train", train_val_split=0.8):
        self.p_flat = p_flat
        self.b_flat = b_flat
        self.z_flat = z_flat
        self.x_flat = x_flat

        N = self.p_flat.shape[0]
        indices = torch.randperm(N)
        split_idx = int(train_val_split * N)

        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        if split == "train":
            self.idx = train_idx
        else:
            self.idx = val_idx

        print(f"[Supervised-{split}] Dataset ready. N={len(self.idx)}")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        idx = self.idx[i]
        return self.p_flat[idx], self.b_flat[idx], self.z_flat[idx], self.x_flat[idx]


class HistoricalDataset(Dataset):
    """Unsupervised Dataset (1900-2023 Bag of Vectors)."""

    def __init__(self, path, stats, p_dim=84):
        print(f"Loading historical bag from {path}...")
        raw_data = np.load(path)
        self.data = torch.tensor(raw_data, dtype=torch.float32)

        self.p_data = self.data[:, :p_dim]
        self.b_data = self.data[:, p_dim:]

        print("Normalizing history with 2023 stats...")
        self.p_data = (self.p_data - stats["p_mu"]) / (stats["p_sd"] + 1e-6)
        self.b_data = (self.b_data**0.1 - stats["b_mu"]) / (stats["b_sd"] + 1e-6)

        print(f"[Historical] Dataset ready. N={len(self.p_data)}")

    def __len__(self):
        return len(self.p_data)

    def __getitem__(self, idx):
        return self.p_data[idx], self.b_data[idx]


def true_kernel_loss(z_pred, x_raw, num_pairs=4096):
    """Computes loss between the dot product in Z and the Ruzicka similarity in X."""
    B = z_pred.shape[0]
    idx = torch.randint(0, B, (2, num_pairs), device=z_pred.device)
    i, j = idx[0], idx[1]

    xi, xj = x_raw[i], x_raw[j]

    sum_plus = xi + xj
    diff_abs = torch.abs(xi - xj)
    numerator = 0.5 * torch.sum(sum_plus - diff_abs, dim=1)
    denominator = 0.5 * torch.sum(sum_plus + diff_abs, dim=1)

    valid_pairs = denominator > 1e-3
    if valid_pairs.sum() == 0:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)

    sim_true = numerator[valid_pairs] / (denominator[valid_pairs] + 1e-8)
    zi, zj = z_pred[i][valid_pairs], z_pred[j][valid_pairs]
    sim_pred = (zi * zj).sum(dim=1)

    return F.mse_loss(sim_pred, sim_true)


def train_model_semisup(train_ds, val_ds, hist_ds, dims, epochs=50, lr=1e-3, batch_size=4096, weights=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if weights is None:
        weights = {"stabilizing": 1.0, "metric": 5.0, "reconstruction": 0.1}

    model = MultiInputAutoencoder(dims["p"], dims["b"], dims["z"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    loader_sup = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_unsup = DataLoader(hist_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    iter_unsup = iter(loader_unsup)

    print("--- Starting Semi-Supervised Training ---")

    for ep in range(1, epochs + 1):
        model.train()
        total_recon_hist = 0.0
        steps = 0

        for p, b, z_ref, x_raw in loader_sup:
            steps += 1
            p, b, z_ref, x_raw = p.to(device), b.to(device), z_ref.to(device), x_raw.to(device)

            try:
                p_hist, b_hist = next(iter_unsup)
            except StopIteration:
                iter_unsup = iter(loader_unsup)
                p_hist, b_hist = next(iter_unsup)
            p_hist, b_hist = p_hist.to(device), b_hist.to(device)

            opt.zero_grad()

            z_pred, recon = model(p, b)
            loss_stab = torch.mean(torch.sum((z_pred - z_ref) ** 2, dim=1))
            loss_true = true_kernel_loss(z_pred, x_raw)
            target_sup = torch.cat([p, b], dim=1)
            loss_recon_sup = F.mse_loss(recon, target_sup)

            _, recon_hist = model(p_hist, b_hist)
            target_hist = torch.cat([p_hist, b_hist], dim=1)
            loss_recon_hist = F.mse_loss(recon_hist, target_hist)

            loss_final = (
                weights["stabilizing"] * loss_stab
                + weights["metric"] * loss_true
                + weights["reconstruction"] * (loss_recon_sup + loss_recon_hist)
            )

            loss_final.backward()
            opt.step()
            total_recon_hist += loss_recon_hist.item()

        model.eval()
        with torch.no_grad():
            val_iter = iter(DataLoader(val_ds, batch_size=batch_size))
            p_v, b_v, z_v, x_v = next(val_iter)
            p_v, b_v, z_v, x_v = p_v.to(device), b_v.to(device), z_v.to(device), x_v.to(device)

            z_pred, recon = model(p_v, b_v)
            target = torch.cat([p_v, b_v], dim=1)
            loss_recon = F.mse_loss(recon, target).item()
            loss_stab = torch.mean(torch.sum((z_pred - z_v) ** 2, dim=1)).item()
            loss_true = true_kernel_loss(z_pred, x_v).item()
            cos_stab = F.cosine_similarity(z_pred, z_v).mean().item()

            avg_hist = total_recon_hist / steps
            scheduler.step(loss_stab)

            print(
                f"Ep {ep:03d} | Stab: {loss_stab:.4f} | True: {loss_true:.4f} | "
                f"Rec(Sup/His): {loss_recon:.4f}/{avg_hist:.4f} | Cos: {cos_stab:.3f}"
            )

    return model


def prepare_supervised_tensors(prism_stack, bui_stack, ebird_stack, z_old_flat, old_mask, out_dir):
    H, W, _ = prism_stack.shape

    strict_data_mask = compute_strict_mask(ebird_stack, prism_stack, bui_stack)
    final_valid_mask = strict_data_mask & old_mask

    print(f"Final Intersection Mask: {final_valid_mask.sum()} pixels")
    np.save(os.path.join(out_dir, "training_mask.npy"), final_valid_mask)

    p_flat = torch.tensor(prism_stack[final_valid_mask], dtype=torch.float32)
    b_flat = torch.tensor(bui_stack[final_valid_mask], dtype=torch.float32)

    x_flat_np = ebird_stack[final_valid_mask]
    x_flat_np = np.nan_to_num(x_flat_np, nan=0.0)
    x_flat = torch.tensor(x_flat_np, dtype=torch.float32)

    Z_grid = np.zeros((H, W, z_old_flat.shape[1]), dtype=np.float32)
    Z_grid[old_mask] = z_old_flat
    z_flat = torch.tensor(Z_grid[final_valid_mask], dtype=torch.float32)

    return p_flat, b_flat, z_flat, x_flat, final_valid_mask


def run_desk_experiment(config=None):
    if config is None:
        config = load_config()
    elif isinstance(config, (str, os.PathLike)):
        config = load_config(config)

    paths = config["paths"]
    desk_cfg = config["desk"]

    ebird_stack, _ = load_tifs_structured(paths["ebird_folder"], "*_abundance_median_2023-*.tif")

    print("Loading 2023 EMA State...")
    state_2023 = np.load(os.path.join(paths["hist_dir"], "state_2023_bio_ema10.npz"))
    prism_stack = state_2023["prism"]
    bui_stack = state_2023["bui"]

    z_dir = desk_cfg["z_dir"]
    try:
        old_mask = np.load(os.path.join(z_dir, "valid_mask.npy"))
        z_old_flat = np.load(os.path.join(z_dir, "Z.npy"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Could not find Z.npy or valid_mask.npy in {z_dir}") from exc

    out_dir = paths["desk_output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    p_flat, b_flat, z_flat, x_flat, _ = prepare_supervised_tensors(
        prism_stack,
        bui_stack,
        ebird_stack,
        z_old_flat,
        old_mask,
        out_dir,
    )

    p_mu, p_sd = p_flat.mean(0), p_flat.std(0)
    b_mu, b_sd = (b_flat**0.1).mean(0), (b_flat**0.1).std(0)

    p_flat = (p_flat - p_mu) / (p_sd + 1e-6)
    b_flat = (b_flat**0.1 - b_mu) / (b_sd + 1e-6)

    stats_dict = {"p_mu": p_mu, "p_sd": p_sd, "b_mu": b_mu, "b_sd": b_sd}

    train_ds = PixelDataset(p_flat, b_flat, z_flat, x_flat, split="train", train_val_split=desk_cfg.get("train_val_split", 0.8))
    val_ds = PixelDataset(p_flat, b_flat, z_flat, x_flat, split="val", train_val_split=desk_cfg.get("train_val_split", 0.8))

    hist_ds = HistoricalDataset(
        os.path.join(paths["hist_dir"], "history_vectors_bio_ema10.npy"),
        stats=stats_dict,
        p_dim=p_flat.shape[1],
    )

    dims = {"p": p_flat.shape[1], "b": b_flat.shape[1], "z": z_flat.shape[1]}
    weights = desk_cfg.get("weights", {"stabilizing": 1.0, "metric": 5.0, "reconstruction": 0.1})
    model = train_model_semisup(
        train_ds,
        val_ds,
        hist_ds,
        dims,
        epochs=desk_cfg.get("epochs", 100),
        lr=desk_cfg.get("lr", 1e-3),
        batch_size=desk_cfg.get("batch_size", 4096),
        weights=weights,
    )

    torch.save(model.state_dict(), os.path.join(out_dir, "env_model_semisup.pth"))


if __name__ == "__main__":
    run_desk_experiment()
