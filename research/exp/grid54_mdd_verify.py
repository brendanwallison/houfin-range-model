"""Verify the 54 km kernel-scale inputs (juvenile mean dispersal distance 200 / 110 km, shape .468) against the 330 km
obsroute input: everything but the juvenile kernels / Z_disp identical; realized MDD, band masses and mean distances
from the stored kernels; Z_disp change summary.

    python research/exp/grid54_mdd_verify.py 200 110      (writes grid54/grid54_mdd_verify.json)
"""
import json
import os
import pickle
import sys

import numpy as np

P = "/work/07980/bwa386/ls6/houfin/processed/grid54"
OLD = f"{P}/model_inputs/numpyro_input_s64_grid54_obsroute"
KERNEL_KEYS = {"juvenile_fft_kernel_stack", "juvenile_edge_correction_stack", "juvenile_kernel_labels", "K",
               "dispersal_spec", "z_disp_gathered_path", "z_gathered_path", "ingest_id", "population_model_spec",
               "path_feature_meta"}


def meta(d):
    with open(os.path.join(d, "metadata.pkl"), "rb") as fh:
        return pickle.load(fh)


def kernel_summary(m, cell=54.0):
    ks = np.real(np.fft.ifft2(np.asarray(m["juvenile_fft_kernel_stack"])))
    K, Ly, Lx = ks.shape
    nb = K // 4
    dy = np.minimum(np.arange(Ly), Ly - np.arange(Ly)); dx = np.minimum(np.arange(Lx), Lx - np.arange(Lx))
    r = cell * np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    tot = ks.sum(0)
    out = {"mass": float(tot.sum()), "realized_mdd_km": float((tot * r).sum() / tot.sum()),
           "self_cell": float(tot[0, 0]), "beyond_1000km": float(tot[r > 1000].sum()), "bands": []}
    for b in range(nb):
        kb = ks[[d * nb + b for d in range(4)]].sum(0)
        out["bands"].append({"mass": float(kb.sum()), "mean_km": float((kb * r).sum() / kb.sum())})
    return out


def main():
    a = meta(OLD)
    res = {"330": kernel_summary(a)}
    for mdd in sys.argv[1:]:
        d = f"{P}/model_inputs/numpyro_input_s64_grid54_mdd{mdd}_obsroute"
        b = meta(d)
        diff = []
        for k in sorted(set(a) | set(b)):
            if k in KERNEL_KEYS:
                continue
            va, vb = a.get(k), b.get(k)
            try:
                same = (np.array_equal(np.asarray(va), np.asarray(vb)) if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray)
                        else va == vb)
            except Exception:
                same = False
            if not same:
                diff.append(k)
        za = np.memmap(os.path.join(OLD, a["z_gathered_path"]), dtype="float32", mode="r", shape=(a["time"], a["N_land"], a["M"]))
        zb = np.memmap(os.path.join(d, b["z_gathered_path"]), dtype="float32", mode="r", shape=(b["time"], b["N_land"], b["M"]))
        zda = np.memmap(os.path.join(OLD, a["z_disp_gathered_path"]), dtype="float32", mode="r", shape=(a["time"], a["N_land"], a["K"], a["M"]))
        zdb = np.memmap(os.path.join(d, b["z_disp_gathered_path"]), dtype="float32", mode="r", shape=(b["time"], b["N_land"], b["K"], b["M"]))
        corr = {}
        for t in (0, 64, 123):
            x, y = np.asarray(zda[t]), np.asarray(zdb[t])
            corr[t] = [float(np.corrcoef(x[:, k].ravel(), y[:, k].ravel())[0, 1]) for k in range(a["K"])]
        res[mdd] = dict(kernel_summary(b), metadata_keys_differing_beyond_kernels=diff,
                        Z_identical=bool(all(np.array_equal(za[t], zb[t]) for t in range(a["time"]))),
                        dispersal_spec=b["dispersal_spec"], labels_same=list(a["juvenile_kernel_labels"]) == list(b["juvenile_kernel_labels"]),
                        adult_kernel_identical=bool(np.array_equal(np.asarray(a["adult_fft_kernel"]), np.asarray(b["adult_fft_kernel"]))),
                        Zdisp_corr_vs_330_by_kernel=corr)
    with open(f"{P}/grid54_mdd_verify.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "Zdisp_corr_vs_330_by_kernel"} for k, v in res.items()}, indent=1))
    for k, v in res.items():
        if "Zdisp_corr_vs_330_by_kernel" in v:
            print(k, "Z_disp corr vs 330 (min over kernels, by year):", {t: round(min(c), 4) for t, c in v["Zdisp_corr_vs_330_by_kernel"].items()})


if __name__ == "__main__":
    main()
