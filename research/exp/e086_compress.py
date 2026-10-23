"""E086 V3: a JOINT basis by compressing the sum-of-Ruzickas features DESK predicts, at k dims, as a research cache.

    python research/exp/e086_compress.py --cache <late cache> --head-ckpt <resume_checkpoint.pt.completed>
        --amps <fast_arm sum results.json> --k 24 --z-rank 24 --out <new cache dir>

Stacks the amplitude-weighted blocks the sum readout uses -- sqrt(a_B) z[:z_rank], sqrt(a_h) H_h [z, private_h] --
so their Gram matrix IS the readout's median kernel; takes the UNCENTRED principal directions (the ESK convention:
the kernel is an uncentred dot product) of the DESK-PREDICTED stack over every cell x year in the cache (covariate
only: no observation enters), and writes a cache whose z_*_all / z_* are the top-k scores, ordered by predicted
energy. Every other cache file is symlinked. fast_arm --rank k on it is the iso readout of the joint basis.
The projection is fitted on z_raw (what the readout reads) and the SAME one applied to z_ema. Records the energy captured at k.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--head-ckpt", required=True)
    ap.add_argument("--amps", required=True, help="fast_arm results.json of the sum readout (block_amplitude_medians)")
    ap.add_argument("--k", type=int, required=True, help="components kept; 0 = all (the exact fixed-metric equivalent)")
    ap.add_argument("--z-rank", type=int, default=24)
    ap.add_argument("--order", choices=("energy", "within"), default="energy",
                    help="E086c: rank directions by total uncentred energy (V3) or by WITHIN-cell (over-years) energy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    st = torch.load(a.head_ckpt, map_location="cpu", weights_only=False)
    heads = [h["weight"].numpy().astype("float64") for h in st["metric_heads"]]
    amps = json.load(open(a.amps))["block_amplitude_medians"]
    amps = list(amps.values()) if isinstance(amps, dict) else list(amps)
    assert len(amps) >= 1 + len(heads), amps
    j = lambda f: os.path.join(a.cache, f)

    def stack(kind):
        z = np.asarray(np.load(j(f"z_{kind}_all.npy"), mmap_mode="r"), "float64")
        pp = j(f"zpriv_{kind}_all.npy")
        p = np.asarray(np.load(pp, mmap_mode="r"), "float64") if os.path.exists(pp) else None
        C, Y, L = z.shape
        blocks = [np.sqrt(amps[0]) * z[..., : a.z_rank]]
        a0 = 0
        for h, Wh in enumerate(heads):
            pk = Wh.shape[1] - L
            inp = np.concatenate([z, p[..., a0:a0 + pk]], -1) if pk > 0 else z
            a0 += max(pk, 0)
            blocks.append(np.sqrt(amps[1 + h]) * inp @ Wh.T)
        return np.concatenate(blocks, -1)

    F = stack("raw")   # the readout (fast_arm default) reads raw z; amplitudes were fitted on it
    C, Y, D = F.shape
    Ff = F.reshape(-1, D)
    Ff = Ff[np.isfinite(Ff).all(1)]
    if a.order == "within":                       # E086c: deviations of each cell from its own mean over years
        Fw = F - np.nanmean(F, axis=1, keepdims=True)
        Ff = Fw.reshape(-1, D)
        Ff = Ff[np.isfinite(Ff).all(1)]
    G = Ff.T @ Ff / len(Ff)                       # uncentred second moment (of levels, or of within-cell deviations)
    ev, V = np.linalg.eigh(G)
    order = np.argsort(ev)[::-1]
    ev, V = ev[order], V[:, order]
    if a.k <= 0:
        a.k = D
    Vk = V[:, : a.k]
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.cache):
        if f.startswith(("z_ema", "z_raw", "zpriv_")) or os.path.exists(os.path.join(a.out, f)):
            continue
        os.symlink(os.path.abspath(j(f)), os.path.join(a.out, f))
    ci = np.load(j("key_cell_index.npy")); keys = np.load(j("keys.npy")); years = np.load(j("years.npy"))
    yi = keys[:, 2] - years[0]
    for kind, Fk in (("raw", F), ("ema", stack("ema"))):
        S = (Fk @ Vk).astype("float32")
        np.save(os.path.join(a.out, f"z_{kind}_all.npy"), S)
        np.save(os.path.join(a.out, f"z_{kind}.npy"), S[ci, yi])
    info = {"source_cache": a.cache, "head_ckpt": a.head_ckpt, "amps": amps[: 1 + len(heads)], "k": a.k,
            "z_rank": a.z_rank, "stack_dim": int(D), "order": a.order, "energy_at_k": float(ev[: a.k].sum() / ev.sum()),
            "energy_cum": (np.cumsum(ev) / ev.sum()).tolist(),
            "block_dims": [a.z_rank] + [int(W.shape[0]) for W in heads]}
    json.dump(info, open(os.path.join(a.out, "compress.json"), "w"), indent=1)
    print(f"[e086-compress] {os.path.basename(a.cache.rstrip('/'))}: stack {D} dims -> k={a.k}, energy "
          f"{info['energy_at_k']:.3f} (24: {info['energy_cum'][23]:.3f}, 32: {info['energy_cum'][31]:.3f})")


if __name__ == "__main__":
    main()
