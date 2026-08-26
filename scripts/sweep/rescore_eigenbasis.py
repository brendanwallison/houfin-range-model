"""Recompute the eigenbasis diagnostics from saved checkpoints, with an error bar.

The nesting gap separated the 19 stage-1 configurations by 18% -- against ~7% for the kernel, and
with the baseline ranking 16th of 19 rather than 3rd -- but it was measured on ONE fixed batch, so
none of that spread could be told from sampling noise. Reporting a more discriminating number
without its error bar is how the single-draw kernel estimate misled earlier in this sweep.

This gets the error bar without retraining: it reloads each run's checkpoint, reconstructs the
supervised quantity, and scores several INDEPENDENT held-out batches. ~86 grid forwards per run,
minutes rather than the ~17 GPU-hours a rerun would cost.

    python scripts/sweep/rescore_eigenbasis.py $HOUFIN_PROCESSED/sweeps/desk_hp/sweep_t0_f100_*

**Two honest caveats about what this measures.**

1. The checkpoints were saved at each run's UNSMOOTHED best epoch, which is not the epoch the
   smoothed ranking picks. For a question about the geometry of the learned basis that is probably
   acceptable -- but it is a different epoch, not the same one, and any comparison against the
   smoothed kernel ranking inherits that mismatch.
2. It reconstructs ``z_ema`` via ``apply_output_ema``, the numpy twin of the trainer's torch scan
   (``tests/test_output_ema.py`` asserts they agree). That is the SUPERVISED quantity and the one
   the in-training diagnostic used, so these numbers are comparable with the recorded ones -- and
   deliberately not ``z_raw``, which ``validate_spacetime.encode_points`` would have given more
   cheaply but which is a different tensor.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.community_encoder.train_DESK import covariate_io as cio          # noqa: E402
from src.community_encoder.train_DESK.desk_training import (              # noqa: E402
    apply_output_ema, load_point_set, spacetime_metric_pool)
from src.community_encoder.train_DESK.eigenbasis_diag import (            # noqa: E402
    eigenbasis_report, ruzicka_gram)
from src.community_encoder.train_DESK.esk_kernel import project_points_to_z  # noqa: E402
from src.community_encoder.train_DESK.model_arch import (                 # noqa: E402
    MultiStreamAutoencoder, hidden_width_from_meta)
from src.config_utils import load_config, target_points_dir               # noqa: E402


def _z_ema_window(run_dir, states_dir, dm, device):
    """Forward every year of the window and apply the learned output EMA. ``(T,H,W,L)``, years.

    The whole window is required, not just the years holding val points: the output EMA is a
    CAUSAL scan from ``ema_warmup_start``, so a year's value depends on every year before it.
    Encoding points year-by-year would give ``z_raw`` instead -- a different tensor.
    """
    import torch

    schema = json.loads(str(dm["schema"]))
    cio.assert_schema_compatible(schema, cio.load_schema(states_dir),
                                 context="rescore_eigenbasis")
    model = MultiStreamAutoencoder(
        [int(d) for d in dm["stream_dims"]], int(dm["latent_dim"]),
        int(dm["spatial_kernel"]) if "spatial_kernel" in dm else 0,
        hidden_width=hidden_width_from_meta(dm),
        mlp_expansion=int(dm["mlp_expansion"]) if "mlp_expansion" in dm else 4)
    model.load_state_dict(torch.load(os.path.join(run_dir, "env_model_semisup.pth"),
                                     map_location=device))
    model.to(device).eval()
    mu, sd = dm["mu"].astype("float32"), dm["sd"].astype("float32")
    start, end = int(dm["ema_warmup_start"]), int(dm["label_year"])
    zs, kept = [], []
    for y in range(start, end + 1):
        try:
            cov = cio.load_state_stack(y, states_dir, schema)
        except FileNotFoundError:
            continue
        covn, valid = cio.norm_grid(cov, mu, sd)
        with torch.no_grad():
            zz, _ = model(torch.tensor(covn[None], dtype=torch.float32, device=device),
                          torch.tensor(valid[None], device=device))
        zs.append(zz[0].float().cpu().numpy())
        kept.append(y)
    z_raw = np.stack(zs, 0)
    hl = float(dm["ema_half_life"])
    if not np.isfinite(hl) or hl <= 0:
        raise SystemExit(f"{run_dir}: desk_meta has no usable ema_half_life ({hl}); cannot "
                         f"reconstruct the supervised z_ema")
    return apply_output_ema(z_raw, hl), kept


def rescore(run_dir, cfg, draws, batch, ranks, seed=0):
    """Every eigenbasis diagnostic for one run, over ``draws`` independent held-out batches."""
    import torch

    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    z_ema, years = _z_ema_window(run_dir, states_dir, dm, device)
    yi = {y: i for i, y in enumerate(years)}

    # The val pool, rebuilt from the masks the run itself saved -- so it is the same held-out
    # cells the run was scored on, not a fresh split.
    tr_mask = np.load(os.path.join(run_dir, "training_mask.npy"))
    holdout = np.load(os.path.join(run_dir, "holdout_cells.npy"))
    m_val = tr_mask & holdout
    Xp, pip, _w, sup = load_point_set(target_points_dir(cfg))
    vy, vf, vx = spacetime_metric_pool(pip, Xp, sup, m_val, holdout.shape[1], exclude_years=())
    keep = np.array([int(y) in yi for y in vy], dtype=bool)
    vy, vf, vx = vy[keep], vf[keep], vx[keep]
    if len(vy) < 16:
        return {"run": os.path.basename(run_dir), "error": f"only {len(vy)} val cell-years"}

    z_flat = z_ema.reshape(z_ema.shape[0], -1, z_ema.shape[-1])
    ti = np.array([yi[int(y)] for y in vy])
    z_pts = z_flat[ti, vf]                                     # (N, L)
    ref = project_points_to_z(vx, cfg["desk"]["z_dir"], int(dm["latent_dim"]))

    reps = []
    for d in range(draws):
        idx = (np.random.default_rng(seed + 7919 * d).choice(len(vy), batch, replace=False)
               if len(vy) > batch else np.arange(len(vy)))
        reps.append(eigenbasis_report(z_pts[idx], vx[idx],
                                      z_ref=(None if ref is None else ref[idx]),
                                      ranks=ranks,
                                      gram=ruzicka_gram(vx[idx])))
    gaps = [r["nesting_gap"] for r in reps if "nesting_gap" in r]
    out = {"run": os.path.basename(run_dir),
           "n_val_cell_years": int(len(vy)), "draws": draws, "batch": int(batch),
           "nesting": float(np.mean([r["nesting"]["nesting_loss"] for r in reps])),
           "nesting_sd": float(np.std([r["nesting"]["nesting_loss"] for r in reps], ddof=1))
           if draws > 1 else 0.0,
           "ratio": float(np.mean([r["nesting"]["operator_metric_ratio"] for r in reps])),
           "max_offdiag": float(np.mean([r["orthogonality"]["max_offdiag"] for r in reps])),
           "inversions": int(np.median([r["spectrum"]["inversions"] for r in reps])),
           "first_inversion": reps[0]["spectrum"]["worst_inversion_at"],
           "disagreement": float(np.mean([r["spectrum"]["estimator_disagreement"]
                                          for r in reps])),
           "best_epoch_of_checkpoint": int(dm["best_epoch"]) if "best_epoch" in dm else None}
    if gaps:
        out["nesting_gap"] = float(np.mean(gaps))
        out["nesting_gap_sd"] = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
        for r in ranks:
            vals = [rep["subspace_vs_ref"].get(r) for rep in reps
                    if r in rep.get("subspace_vs_ref", {})]
            if vals:
                out[f"subspace_r{r}"] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+")
    ap.add_argument("--draws", type=int, default=6,
                    help="independent held-out batches (default 6; the sd estimate is +-32%% "
                         "at 6, enough to compare against an 18%% between-config spread)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--ranks", default="8,16,24,32,48,64")
    ap.add_argument("--out", default=None, help="write the rows as JSON here too")
    args = ap.parse_args()
    ranks = tuple(int(r) for r in args.ranks.split(","))
    cfg = load_config()

    rows = []
    for d in args.run_dir:
        if not os.path.isfile(os.path.join(d, "desk_meta.npz")):
            print(f"skip {os.path.basename(d)}: no desk_meta.npz", file=sys.stderr)
            continue
        print(f"scoring {os.path.basename(d)} ...", file=sys.stderr, flush=True)
        try:
            rows.append(rescore(d, cfg, args.draws, args.batch, ranks))
        except Exception as exc:                      # one bad run must not lose the rest
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append({"run": os.path.basename(d), "error": f"{type(exc).__name__}: {exc}"})
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("no run scored successfully", file=sys.stderr)
        return 1

    print(f"\n{len(ok)} run(s), {args.draws} independent batches of {args.batch} held-out "
          f"cell-years each")
    print("checkpoints are from each run's UNSMOOTHED best epoch, which is not the epoch the "
          "smoothed\nkernel ranking selects -- a different epoch, not the same one\n")
    has_gap = [r for r in ok if "nesting_gap" in r]
    key = "nesting_gap" if has_gap else "nesting"
    print(f"{'run':<26} {'ep':>5} {key:>11} {'+-':>8} {'ratio':>7} {'offdiag':>8} "
          f"{'inv':>4} {'sub@24':>7}")
    print("-" * 84)
    for r in sorted(ok, key=lambda x: x.get(key, 1e9)):
        print(f"{r['run']:<26} {r.get('best_epoch_of_checkpoint') or 0:>5} "
              f"{r.get(key, float('nan')):>11.5f} "
              f"{r.get(key + '_sd', float('nan')):>8.5f} {r['ratio']:>7.3f} "
              f"{r['max_offdiag']:>8.3f} {r['inversions']:>4} "
              f"{r.get('subspace_r24', float('nan')):>7.3f}")
    vals = [r[key] for r in ok if np.isfinite(r.get(key, np.nan))]
    sds = [r[key + "_sd"] for r in ok if np.isfinite(r.get(key + "_sd", np.nan))
           and r[key + "_sd"] > 0]
    if len(vals) > 1 and sds:
        spread = 100 * (max(vals) / min(vals) - 1)
        rel = 100 * float(np.median(sds)) / abs(float(np.median(vals)))
        se = rel / max(args.draws, 1) ** 0.5
        print(f"\nspread across runs {spread:.1f}%; per-batch sd {rel:.1f}%, "
              f"standard error of the mean {se:.2f}%")
        if spread > 3 * se:
            print("  The spread is well above this diagnostic's own noise, so the ordering is a "
                  "real\n  difference between configurations.")
        else:
            print("  The spread is NOT clearly above the noise. Do not read the ordering; raise "
                  "--draws\n  or --batch, or accept that these configurations are not separated "
                  "by this measure.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
