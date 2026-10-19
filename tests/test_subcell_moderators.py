import json

import numpy as np

from src.config_utils import load_config
from src.model.data_loading import _subcell_moderator


def test_overlay_extends_is_recursive_and_child_wins(tmp_path):
    base = tmp_path / "base.json"
    parent = tmp_path / "parent.json"
    child = tmp_path / "child.json"
    base.write_text(json.dumps({"a": {"x": 0, "z": 4}, "base": True}))
    parent.write_text(json.dumps({"a": {"x": 1, "y": 2}}))
    child.write_text(json.dumps({"_extends": "parent.json", "a": {"x": 3}}))
    got = load_config(child, default_name=str(base), env_var=None)
    assert got == {"a": {"x": 3, "y": 2, "z": 4}, "base": True}


def test_builtup_deficit_moderator_is_targeted_and_rms_scaled(tmp_path):
    # Two slices suffice here: the helper sees their area-weighted cell mean.
    u = np.array([[[-2.0, -1.0], [1.0, 2.0]],
                  [[-1.0, 0.0], [2.0, 3.0]]])
    a = np.array([0.5, 0.5])
    meta = {"time": 2, "N_land": 2, "years": np.array([2000, 2001])}
    got = _subcell_moderator(
        str(tmp_path), meta,
        {"covariates": {"builtup": -1.0}, "positive_only": True}, u, a)
    assert np.all(got[:, 1] == 0.0)
    assert np.all(got[:, 0] > 0.0)
    np.testing.assert_allclose(np.sqrt(np.mean(got * got)), 1.0)


def test_named_covariates_combine_before_selecting_deficit_tail(tmp_path):
    x = np.array([
        [[-2.0, 0.0], [1.0, 2.0]],
        [[0.0, 2.0], [-1.0, 0.0]],
    ])
    np.savez(tmp_path / "drift_covariates.npz", x=x,
             names=np.array(["tmin_q10", "snow_q50"]), years=np.array([2000, 2001]))
    meta = {"time": 2, "N_land": 2, "years": np.array([2000, 2001])}
    got = _subcell_moderator(
        str(tmp_path), meta,
        {"covariates": {"tmin_q10": -1.0, "snow_q50": 1.0},
         "positive_only": True},
        np.zeros((2, 2, 1)), np.ones(1))
    assert got[0, 0] > 0 and got[0, 1] > 0
    assert got[1, 0] > 0 and got[1, 1] > 0
    np.testing.assert_allclose(np.sqrt(np.mean(got * got)), 1.0)
