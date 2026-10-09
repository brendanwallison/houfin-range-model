"""research/exp/fast_arm.py on a synthetic cache with the real cache's layout.

Half the dev species are linear readouts of a drifting z (what the fast arm can know), half are pure
noise. The readouts must score positive change skill with an attenuation slope near 1; the noise
species must not be resolvable at all. The same builder is the planted-calibration generator's seed.
"""
import json
import os
import sys

import numpy as np
import pytest


def make_cache(root, n_side=18, years=range(1966, 2026, 2), L=8, n_dev=8, withheld=(), seed=0):
    from src.community_encoder.train_DESK.validation_core import row_splits
    rng = np.random.default_rng(seed)
    cells = np.array([(r, c) for r in range(n_side) for c in range(n_side)], "int32")
    all_years = np.arange(1940, 2026)
    base = rng.normal(size=(len(cells), L))
    drift = rng.normal(size=(len(cells), L))
    frac = np.clip((all_years - 1966) / (2025 - 1966), 0, 1)
    z_all = (base[:, None, :] + frac[None, :, None] * drift[:, None, :]).astype("float32")
    keys = np.array([(r, c, y) for r, c in cells for y in years], "int32")
    ci = np.repeat(np.arange(len(cells)), len(list(years)))
    z = z_all[ci, keys[:, 2] - 1940]
    W = rng.normal(size=(L, n_dev)) * 0.6
    lin = 1.0 + z @ W
    lin[:, n_dev // 2:] = 1.0 + 0.5 * rng.normal(size=(len(keys), n_dev - n_dev // 2))
    lam = np.exp(np.clip(lin, -3, 4))
    X_dev = rng.poisson(lam).astype("float32")
    ho = np.zeros((n_side, n_side), bool)
    ho[:6, 12:] = True
    ho[12:, :6] = True
    bf = np.zeros_like(ho)
    is_train, group, block_id = row_splits(keys, ho, bf, 6, list(withheld), list(withheld))
    os.makedirs(root, exist_ok=True)
    for name, arr in (("keys", keys), ("X_dev", X_dev), ("cells", cells), ("years", all_years),
                      ("z_raw_all", z_all), ("z_ema_all", z_all), ("key_cell_index", ci)):
        np.save(os.path.join(root, f"{name}.npy"), arr)
    np.savez(os.path.join(root, "split.npz"), holdout=ho, buffer=bf, is_train=is_train,
             group=group, block_id=block_id, withheld=np.array(list(withheld), "int32"),
             common=np.array(list(withheld), "int32"))
    json.dump({"dev_species": [f"s{i}" for i in range(n_dev)], "run_dir": "synthetic"},
              open(os.path.join(root, "meta.json"), "w"))
    return n_dev


def _run(monkeypatch, cache, out, *extra):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "exp"))
    import fast_arm
    from research.lib import ledger
    monkeypatch.setattr(ledger, "LEDGER", __import__("pathlib").Path(out) / "ledger.csv")
    monkeypatch.setattr(sys, "argv", ["fast_arm", "--cache", cache, "--out", out, "--n-boot",
                                      "50", "--rank", "8", *extra])
    fast_arm.main()
    return json.load(open(os.path.join(out, "results.json")))


def test_readout_species_score_change_and_noise_species_are_not_resolvable(tmp_path, monkeypatch):
    cache = str(tmp_path / "cache")
    n_dev = make_cache(cache)
    r = _run(monkeypatch, cache, str(tmp_path / "out"))
    ch = r["change"]["space"]
    assert ch["n_resolvable"] == n_dev // 2
    assert ch["skill_vs_no_change"]["median"] > 0.3
    assert 0.6 < ch["attenuation_median"] < 1.4
    import pandas as pd
    tab = pd.read_csv(tmp_path / "out" / "per_species.csv")
    assert tab["resolvable_space"].to_numpy(bool)[: n_dev // 2].all()
    assert r["level"]["space"]["vs_intercept"]["median"] > 0.0
    assert r["level"]["space"]["vs_persistence"] is None          # held-out cells: no own data


def test_split_kernel_sees_that_the_signal_is_in_the_change(tmp_path, monkeypatch):
    """B1 on a planted case: species read out the drift; zbar alone cannot express it. The split
    kernel's within-cell amplitude must not collapse."""
    cache = str(tmp_path / "cache")
    make_cache(cache, seed=2)
    r = _run(monkeypatch, cache, str(tmp_path / "out"), "--kernel", "split")
    a_bar, a_dz = r["block_amplitude_medians"]
    assert a_dz > 0.1 * a_bar
    assert r["change"]["space"]["skill_vs_no_change"]["median"] > 0.3


def test_time_group_uses_leave_one_out_and_persistence(tmp_path, monkeypatch):
    cache = str(tmp_path / "cache")
    make_cache(cache, withheld=range(1966, 1976), seed=3)
    r = _run(monkeypatch, cache, str(tmp_path / "out"))
    assert set(r["change"]) == {"space_time", "time"}
    assert r["level"]["time"]["vs_persistence"] is not None
    assert r["change"]["time"]["n_cells"] > 0
