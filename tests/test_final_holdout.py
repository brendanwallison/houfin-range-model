import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "exp"))
from final_winter_score import paired_circle_summary, exclude_test_circles


def test_relocated_test_circle_cannot_enter_calibration():
    train = np.array([True, True, False, False])
    test = np.array([False, False, True, True])
    circles = np.array([1, 2, 1, 3])
    np.testing.assert_array_equal(exclude_test_circles(train, test, circles), [False, True, False, False])
from final_holdout_prepare import adapters


def test_circle_bootstrap_preserves_paired_constant_gain():
    circle = np.repeat(np.arange(12), np.arange(1, 13))
    result = paired_circle_summary(circle, {"gain": np.full(len(circle), .25)}, boot=100)
    assert result["metrics"]["gain"]["value"] == .25
    np.testing.assert_allclose(result["metrics"]["gain"]["ci95"], [.25, .25])
    assert "metrics" not in paired_circle_summary(circle[circle < 3], {}, boot=100)


def test_confirmation_read_is_logged_before_loading(tmp_path, monkeypatch):
    from research import adopt
    monkeypatch.setattr(adopt, "LOG", tmp_path / "reads.csv")
    (tmp_path / "sealed").mkdir()
    np.save(tmp_path / "sealed" / "X_sealed.npy", np.array([[2, 3]]))
    with pytest.raises(SystemExit):
        adopt.load_sealed(tmp_path, None)
    assert not adopt.LOG.exists()
    np.testing.assert_array_equal(adopt.load_sealed(tmp_path, "FINAL_HELDOUT planted test read"), [[2, 3]])
    assert "FINAL_HELDOUT planted test read" in adopt.LOG.read_text()


def test_adapter_remaps_union_cells_and_selects_season_halves(tmp_path):
    base, features, out = [tmp_path / s for s in ("base", "features", "adapters")]
    base.mkdir()
    (base / "sealed").mkdir()
    keys = np.array([(1, 1, y) for y in (1966, 1967, 2005, 2006)] +
                    [(2, 2, y) for y in (1966, 1967, 2005, 2006)])
    np.save(base / "keys.npy", keys)
    for name in ("F_cov.npy", "F_cov_raw.npy", "esk_annual.npy"):
        np.save(base / name, np.zeros((len(keys), 3)))
    np.savez(base / "split.npz", block_id=np.arange(len(keys)))
    (base / "meta.json").write_text(json.dumps({"sealed_species": ["a"], "dev_species": []}))
    cells, years = np.array([[0, 0], [1, 1], [2, 2]]), np.arange(1902, 2026)
    for family, width in [("final", 64), ("final_v2", 128)]:
        f = features / family
        f.mkdir(parents=True)
        z = np.broadcast_to(np.arange(width), (len(cells), len(years), width)).copy().astype("float32")
        z += np.arange(len(cells))[:, None, None] * 1000
        np.save(f / "z_raw_all.npy", z)
        np.save(f / "cells.npy", cells)
        np.save(f / "years.npy", years)
        for n in ("holdout_cells.npy", "buffer_cells.npy"):
            np.save(f / n, np.zeros((3, 3), bool))
        (f / "DONE.json").write_text("{}")
    adapters(base, features, out)
    z = np.load(out / "v2_r64" / "z_raw.npy")
    np.testing.assert_array_equal(z[0], np.r_[0:32, 64:96] + 1000)
    np.testing.assert_array_equal(z[-1], np.r_[0:32, 64:96] + 2000)
    np.testing.assert_array_equal(np.load(out / "v2_r64" / "key_cell_index.npy"), [1] * 4 + [2] * 4)
    assert np.load(out / "v2_r64" / "split.npz")["is_train"].sum() == 4


def test_fit_cache_preserves_fit_and_separates_changed_responses(tmp_path, monkeypatch):
    import final_cached_fair as cache
    original = cache.blr.fit
    monkeypatch.setattr(cache.blr, "fit", original)
    monkeypatch.setenv("BLR_DEVICE", "cpu")
    monkeypatch.setenv("BLR_MAX_ITER", "5")
    monkeypatch.setenv("BLR_STARTS", "0.5")
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(20, 3)), rng.normal(size=(20, 2))
    cache.install_blr_cache(tmp_path)
    first = cache.blr.fit(X, Y, [(0, 3)])
    second = cache.blr.fit(X, Y, [(0, 3)])
    np.testing.assert_array_equal(first["coef"], second["coef"])
    cache.blr.fit(X, Y + rng.normal(size=Y.shape), [(0, 3)])
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_noise_corrected_summary_does_not_invent_signal_for_bad_denominators():
    from final_holdout_summary import correlation, estimate
    z = {"vt": np.array([[4.], [4.], [-1.], [0.]]),
         "desk__vp": np.array([[4.], [4.], [4.], [4.]]),
         "desk__cv": np.array([[2.], [2.], [2.], [2.]])}
    score = correlation(z, "desk")
    assert score[0] == .5
    assert np.isnan(score[2:]).all()
    assert estimate(score)["valid_bootstrap_share"] == 1 / 3
