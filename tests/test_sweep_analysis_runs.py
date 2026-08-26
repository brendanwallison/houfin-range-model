"""Execution tests for the sweep analysis scripts: they must actually RUN, on realistic input.

Every other test of these scripts asserts on their SOURCE -- that a message exists, that a formula
is not the old one. Those are useful for pinning down reasoning that cannot be exercised locally,
and they are also how a KeyError shipped: a dict key was renamed, the print statement was not, 607
tests passed, and `analyze.py --stage 1 --smooth 5` crashed on the first real invocation after the
header line had already been printed.

So this file executes the entry points end to end on synthetic runs shaped like real ones, and
asserts on their OUTPUT. Cheap -- no cluster, no GPU, no model -- and it covers the one failure mode
source assertions structurally cannot.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_run(root, run_id, *, metric_weight=5.0, n_epochs=500, peak=120, floor=0.0068,
              with_eig=True, with_sd=True, ranks=(8, 16, 24, 32, 48, 64), best_rank=8,
              seed=0):
    """One run directory shaped like a real 500-epoch stage-1 run.

    Deliberately includes the awkward features of the real data: a curve that turns over well
    before the end (so the end-of-training value is NOT the optimum), per-epoch noise large enough
    that the raw argmin is a lucky draw, and an inverted rank curve.
    """
    d = os.path.join(root, run_id)
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(1, n_epochs + 1):
        base = floor * (1 + 1.4 * np.exp(-e / (peak / 2.5)) + 0.22 * max(0, e - peak) / n_epochs)
        v = float(base * (1 + rng.normal(0, 0.06)))
        r = {"epoch": e, "kernel_val": v, "kernel_train": v * 0.72,
             "zmse_val": float(0.21 + 0.35 * np.exp(-e / 40)), "zmse_train": 0.19,
             "kernel_val_sp": v, "kernel_val_spt": float("nan"),
             "zmse_val_sp": 0.21, "zmse_val_spt": float("nan"), "zmse_val_anchor": 0.21,
             "zmse_val_yearout": float("nan"), "lr": 1e-3, "half_life": 10.3,
             "epoch_seconds": 6.1, "selection_metric": "val_kernel", "eval_every": 1}
        if with_sd:
            r["kernel_val_sd"] = v * 0.0112
            r["kernel_val_draws"] = 8
        # an INVERTED rank curve: best at best_rank, worse at full rank
        for rk in ranks:
            pen = abs(ranks.index(rk) - ranks.index(best_rank)) / max(len(ranks) - 1, 1)
            r[f"kernel_val_ema_r{rk}"] = float(v * (1 + 0.05 * pen))
            r[f"kernel_val_raw_r{rk}"] = float(v * (1 + 0.05 * pen) * 1.06)
        if with_eig and e % 10 == 0:
            r.update({"eig_nesting": -1.8, "eig_nesting_gap": 0.49, "eig_nesting_ref": -2.29,
                      "eig_spectrum_inversions": 26, "eig_first_inversion": 4,
                      "eig_max_offdiag": 0.43, "eig_subspace_r24": 0.51,
                      "eig_spectrum_descending": False, "eig_estimator_disagreement": 18.8,
                      "eig_nesting_ratio": 0.63})
        rows.append(r)
    with open(os.path.join(d, "train_trajectory.jsonl"), "w") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    be = int(min(rows[20:], key=lambda r: r["kernel_val"])["epoch"])
    json.dump({"best_epoch": be, "selection_metric": "val_kernel", "restored_best": True,
               "best_val_kernel": rows[be - 1]["kernel_val"],
               "best_val_zmse": rows[be - 1]["zmse_val"],
               "epochs_run": n_epochs, "epochs_budget": n_epochs, "warmup_epochs": 20,
               "min_lr_frac": 0.05, "selection_smooth": 0, "metric_weight": metric_weight,
               "metric_pairs": 65536, "eval_kernel_pairs": 65536, "eval_kernel_draws": 8,
               "n_train_cell_years": 80415, "n_params": 2500000},
              open(os.path.join(d, "run_summary.json"), "w"))
    ho = np.zeros((60, 60), bool); ho[:, :8] = True
    bf = np.zeros((60, 60), bool); bf[:, 8:11] = True
    np.save(os.path.join(d, "holdout_cells.npy"), ho)
    np.save(os.path.join(d, "buffer_cells.npy"), bf)
    np.savez(os.path.join(d, "desk_meta.npz"), best_epoch=be, val_cells=532, train_cells=2791,
             buffer_cells=579, train_dropped_cells=0)
    for f in ("env_model_semisup.pth", "output_ema.pth"):
        open(os.path.join(d, f), "w").write("x")
    return d


def _stage1_root(tmp_path):
    root = str(tmp_path / "sweeps" / "hp")
    os.makedirs(root, exist_ok=True)
    for i, (cfg, w, br) in enumerate([("base", 5.0, 8), ("hl4", 5.0, 8), ("mw20", 20.0, 16),
                                      ("mw40", 40.0, 16), ("mw60", 60.0, 16), ("w64", 5.0, 16)]):
        _make_run(root, f"sweep_t0_f100_{cfg}", metric_weight=w, best_rank=br, seed=i)
    return root


def _rank_rows(text):
    """Rows of the RANKING table only.

    Section-aware on purpose: the eigenbasis table's rows also begin with a configuration name, so
    a filter on the line prefix silently mixes the two tables and reads a column from the wrong
    one. That is how the first version of these tests failed -- on '4.0%' where a kernel value was
    expected.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines)
                     if l.startswith("config") and "vs base" in l) + 2
    except StopIteration:
        return []
    out = []
    for l in lines[start:]:
        if not l.strip() or l.startswith(("threshold", "=", "NOTE", "WARNING", "rank ")):
            break
        out.append(l.split())
    return out


def _run(*args):
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep", "analyze.py"),
                          *args], capture_output=True, text=True, cwd=REPO)
    return out


def test_stage1_runs_to_completion_unsmoothed(tmp_path):
    """The plain path must produce a full table, not a header and a traceback."""
    out = _run("--root", _stage1_root(tmp_path), "--stage", "1")
    assert out.returncode == 0, out.stderr
    assert "Traceback" not in out.stderr, out.stderr
    for token in ("estimator noise floor", "config", "vs base", "(baseline)",
                  "eigenbasis diagnostics", "--smooth 5"):
        assert token in out.stdout, token
    # every configuration reaches the table, not just the ones before the crash
    for cfg in ("base", "hl4", "mw20", "mw40", "mw60", "w64"):
        assert cfg in out.stdout, cfg


def test_stage1_runs_to_completion_smoothed(tmp_path):
    """The --smooth path is the one that crashed with KeyError: 'tail' after the header."""
    out = _run("--root", _stage1_root(tmp_path), "--stage", "1", "--smooth", "5")
    assert out.returncode == 0, out.stderr
    assert "Traceback" not in out.stderr, out.stderr
    assert "TRAILING MEDIAN" in out.stdout
    assert "rank stability, raw argmin vs" in out.stdout
    # and the min_delta note must NOT fire: with smoothing the epochs differ by design
    assert "min_delta rejected" not in out.stdout
    for cfg in ("base", "hl4", "mw40", "mw60", "w64"):
        assert cfg in out.stdout, cfg


def test_the_smoothed_ranking_differs_from_the_raw_one_on_noisy_data(tmp_path):
    """If smoothing changed nothing on data this noisy, it would not be doing its job."""
    root = _stage1_root(tmp_path)
    raw = _run("--root", root, "--stage", "1").stdout
    sm = _run("--root", root, "--stage", "1", "--smooth", "5").stdout

    assert _rank_rows(raw) and _rank_rows(sm)
    eps_raw = {r[0]: r[1] for r in _rank_rows(raw)}
    eps_sm = {r[0]: r[1] for r in _rank_rows(sm)}
    assert eps_raw != eps_sm, "smoothing did not change any selected epoch"
    # and the smoothed value must be >= the raw one for every config: a median cannot beat a min
    val_raw = {r[0]: float(r[2]) for r in _rank_rows(raw)}
    val_sm = {r[0]: float(r[2]) for r in _rank_rows(sm)}
    for cfg in val_raw:
        assert val_sm[cfg] >= val_raw[cfg] - 1e-12, (cfg, val_raw[cfg], val_sm[cfg])


def test_the_end_of_training_column_is_not_the_optimum(tmp_path):
    """The fixture peaks well before the end, so those two columns must differ.

    This is the property that made the old robustness check meaningless, so it is worth pinning:
    if a fixture were built where the curve never turned over, the bug would be invisible again.
    """
    root = _stage1_root(tmp_path)
    out = _run("--root", root, "--stage", "1", "--smooth", "5").stdout
    rows = _rank_rows(out)
    assert rows
    for r in rows:
        kernel, endval = float(r[2]), float(r[3])
        assert endval > kernel, (r[0], kernel, endval)
    assert "over-training resistance" in out


def test_check_run_runs_to_completion(tmp_path):
    """Same execution guarantee for the per-run checker."""
    root = _stage1_root(tmp_path)
    d = os.path.join(root, "sweep_t0_f100_base")
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "sweep", "check_run.py"),
                          d], capture_output=True, text=True, cwd=REPO)
    assert "Traceback" not in out.stderr, out.stderr
    for token in ("trajectory has", "estimator noise", "rank curve", "s/epoch"):
        assert token in out.stdout, token


def test_stage2_runs_to_completion(tmp_path):
    """stage 2 has no execution test either, and it formats a table the same way."""
    root = str(tmp_path / "sweeps" / "hp2")
    os.makedirs(root, exist_ok=True)
    for t in ("t0", "t1975"):
        for f in ("f100", "f70"):
            for cfg in ("base", "hl4"):
                _make_run(root, f"sweep_{t}_{f}_{cfg}")
    out = _run("--root", root, "--stage", "2")
    assert out.returncode == 0, out.stderr
    assert "Traceback" not in out.stderr, out.stderr
    assert "best EPOCH by configuration" in out.stdout
    assert "winner per cell" in out.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
