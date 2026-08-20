"""Does the model predict the DIRECTION of community change better than interpolation?

The single-pair diagnostic inside the trainer said no: model dir-cos 0.48 against 0.51 for
inverse-distance interpolation of the targets. But it rested on 36 validation cells, because it
compares 1966 with 2025 and 1966 is BBS's launch year -- 351 cells surveyed, 225 of them also
surveyed in 2025. A 0.03 difference on 36 cells decides nothing.

This panel fixes the sample and adds a test the single pair cannot run:

* **Epochs, not one pair.** 1967/1985/2005/2025, every pair reported separately. Each cell uses
  its own nearest actual survey within ``tol`` years of the epoch, and the model is read at that
  same real year -- no averaging, no interpolation in time. Union over the six pairs is ~2,900
  cells against 225.
* **Curvature.** A 1967->2025 difference is a chord: it cannot tell monotone growth from
  rise-then-fall. Second differences over three consecutive epochs can, and for House Finch that
  is the eastern story -- expansion, then decline after conjunctivitis arrives in the 1990s.
  Nothing currently measured can fail that test.
* **A real bar in every cell.** Inverse-distance interpolation of the targets, computed per
  pair, so the comparison is local rather than one pass/fail for the whole record.

Spacing is not arbitrary: ~19 years is about two output-EMA half-lives at the learned 10.5 y, the
shortest interval over which the EMA is not dominating the predicted change.

What this CANNOT do: treat the six pairs as independent evidence. They share cells and nest in
time (1967->2025 contains 1985->2005), so they are reported per pair and never pooled. Note also
that the adjacent intervals summing to the full span is an identity, not a check -- both sides
are differences of the same vectors -- so it is not reported.

    python scripts/run_encoder.py validate-epochs
"""
import itertools
import json
import os

import numpy as np
import torch

DEFAULT_EPOCHS = (1967, 1985, 2005, 2025)
DEFAULT_TOL = 2


def nearest_survey(pip, supervise, epochs, tol):
    """``{epoch: {(row,col): year}}`` -- each cell's surveyed year closest to each epoch.

    Ties break to the EARLIER year, so the choice cannot depend on dict ordering. Only rows
    flagged ``supervise`` count, since duplicates exist for the kernel only.
    """
    sup = supervise if supervise is not None else np.ones(len(pip), bool)
    out = {int(e): {} for e in epochs}
    order = np.lexsort((pip[:, 2],))                       # ascending year -> earlier wins ties
    for i in order:
        if not sup[i]:
            continue
        r, c, y = int(pip[i, 0]), int(pip[i, 1]), int(pip[i, 2])
        for e in out:
            if abs(y - e) > tol:
                continue
            prev = out[e].get((r, c))
            if prev is None or abs(y - e) < abs(prev - e):
                out[e][(r, c)] = y
    return out


def _idw_at(cells_wanted, train_years, z_of, k=8, power=2.0):
    """Inverse-distance estimate at ``cells_wanted`` from ``train_years`` (cell -> year).

    ``z_of(cell, year)`` supplies the target vector. Each training cell contributes the value at
    ITS OWN nearest year to the epoch, which is the same rule the model side uses.
    """
    from scipy.spatial import cKDTree

    if len(train_years) < k or not cells_wanted:
        return None
    tr_cells = sorted(train_years)
    tr_rc = np.array(tr_cells, dtype=float)
    src = np.stack([z_of(c, train_years[c]) for c in tr_cells])
    dist, idx = cKDTree(tr_rc).query(np.array(cells_wanted, dtype=float), k=k)
    w = 1.0 / np.maximum(dist, 1e-6) ** power
    w = w / w.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkl->nl", w, src[idx])


def _dcos(dp, dt):
    from .desk_training import median_dir_cos
    return median_dir_cos(torch.as_tensor(np.asarray(dp), dtype=torch.float32),
                          torch.as_tensor(np.asarray(dt), dtype=torch.float32))


def run_epoch_directions(config=None, epochs=None, tol=None, out_dir=None):
    from src.config_utils import load_config, target_points_dir
    from .desk_training import (blocked_holdout, load_point_set, supervised_cells)
    from .esk_kernel import project_points_to_z
    from .validate_bbs_routes import desk_z_ema

    config = load_config(config) if not isinstance(config, dict) else config
    desk_cfg = config["desk"]
    ecfg = (desk_cfg.get("epoch_directions") or {})
    epochs = tuple(int(e) for e in (epochs or ecfg.get("epochs") or DEFAULT_EPOCHS))
    tol = int(DEFAULT_TOL if tol is None else tol) if ecfg.get("tol") is None else int(ecfg["tol"])
    z_dir, _cfg = desk_cfg["z_dir"], desk_cfg.get("trend", {})
    points_dir = target_points_dir(config)

    X, pip, _w, supervise = load_point_set(points_dir)
    latent_dim = int(desk_cfg.get("latent_dim")
                     or np.load(os.path.join(z_dir, "Z.npy")).shape[1])
    z_obs = project_points_to_z(X, z_dir, latent_dim)
    if z_obs is None:
        raise FileNotFoundError(f"need the ESK projection in {z_dir}; re-run spacetime-esk")

    import rasterio
    from src.config_utils import load_data_config
    with rasterio.open(load_data_config()["grid"]["ref_raster"]) as src:
        H, W = src.height, src.width
    spatial_kernel = int(desk_cfg.get("spatial_conv", {}).get("kernel", 3)) \
        if desk_cfg.get("spatial_conv", {}).get("enabled", True) else 0
    sup_cells = supervised_cells(pip, supervise, (H, W))
    holdout, buffer_mask = blocked_holdout(
        sup_cells, block_cells=int(_cfg.get("block_cells", 12)),
        holdout_frac=float(_cfg.get("holdout_frac", 0.15)),
        buffer_cells=spatial_kernel // 2, seed=int(_cfg.get("seed", 0)))
    print(f"[epochs] split reproduced: {int(sup_cells.sum()):,} supervised, "
          f"{int(holdout.sum()):,} val, {int(buffer_mask.sum()):,} buffer", flush=True)

    # target vector lookup, by (cell, year)
    row_of = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(pip)}
    def z_target(cell, year):
        return z_obs[row_of[(cell[0], cell[1], year)]]

    near = nearest_survey(pip, supervise, epochs, tol)
    for e in epochs:
        print(f"[epochs] {e}+/-{tol}: {len(near[e]):,} cells with a survey", flush=True)

    val_of = {e: {c: y for c, y in near[e].items() if holdout[c]} for e in epochs}
    trn_of = {e: {c: y for c, y in near[e].items()
                  if not holdout[c] and not buffer_mask[c]} for e in epochs}

    # One desk_z_ema call for every (cell, year) the panel needs: it re-encodes the whole
    # 1940..max span per call, so asking once is the difference between minutes and an hour.
    wanted = sorted({(c[0], c[1], y) for e in epochs
                     for d in (val_of[e], trn_of[e]) for c, y in d.items()})
    keys = np.array(wanted, dtype=np.int64)
    print(f"[epochs] encoding {len(keys):,} cell-years through the saved DESK + output EMA",
          flush=True)
    z_model_rows = desk_z_ema(config, keys)
    m_of = {k: z_model_rows[i] for i, k in enumerate(wanted)}

    results = {"epochs": list(epochs), "tol": tol, "pairs": {}, "curvature": {}}
    print("\n  pair            cells   model   idw    null   verdict")
    for a, b in itertools.combinations(epochs, 2):
        cells = sorted(set(val_of[a]) & set(val_of[b]))
        if len(cells) < 10:
            print(f"  {a}->{b}   {len(cells):>6}   (too few cells)")
            continue
        dt = np.stack([z_target(c, val_of[b][c]) - z_target(c, val_of[a][c]) for c in cells])
        dm = np.stack([m_of[(c[0], c[1], val_of[b][c])] - m_of[(c[0], c[1], val_of[a][c])]
                       for c in cells])
        ia = _idw_at(cells, trn_of[a], z_target)
        ib = _idw_at(cells, trn_of[b], z_target)
        di = (ib - ia) if (ia is not None and ib is not None) else None
        rng = np.random.default_rng(0)
        null = _dcos(dm, dt[rng.permutation(len(cells))])
        mc, ic = _dcos(dm, dt), (_dcos(di, dt) if di is not None else float("nan"))
        verdict = "model" if mc > ic + 0.02 else ("idw" if ic > mc + 0.02 else "tie")
        print(f"  {a}->{b}   {len(cells):>6}   {mc:5.2f}  {ic:5.2f}  {null:5.2f}   {verdict}")
        results["pairs"][f"{a}_{b}"] = {"n": len(cells), "model_dir_cos": mc,
                                        "idw_dir_cos": ic, "null_dir_cos": null,
                                        "verdict": verdict}

    # Curvature: does the model reproduce a CHANGE in direction (expansion then decline)?
    print("\n  curvature (second difference over three epochs)")
    for a, b, c3 in zip(epochs, epochs[1:], epochs[2:]):
        cells = sorted(set(val_of[a]) & set(val_of[b]) & set(val_of[c3]))
        if len(cells) < 10:
            print(f"  {a}/{b}/{c3}  {len(cells):>6}  (too few cells)")
            continue
        def second(get):
            d1 = np.stack([get(cc, b) - get(cc, a) for cc in cells])
            d2 = np.stack([get(cc, c3) - get(cc, b) for cc in cells])
            return d2 - d1
        gt = lambda cc, e: z_target(cc, val_of[e][cc])
        gm = lambda cc, e: m_of[(cc[0], cc[1], val_of[e][cc])]
        st, sm = second(gt), second(gm)
        val = _dcos(sm, st)
        print(f"  {a}/{b}/{c3}  {len(cells):>6}  model {val:5.2f}")
        results["curvature"][f"{a}_{b}_{c3}"] = {"n": len(cells), "model_dir_cos": val}

    out_dir = out_dir or config["paths"]["desk_output_dir"]
    path = os.path.join(out_dir, "epoch_directions.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n[epochs] wrote {path}")
    return results


if __name__ == "__main__":
    run_epoch_directions()
