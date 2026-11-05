import json
import math

import pytest

from research.exp.local_refine_selected import select_candidate


def candidate(tmp_path, name, objective, config=None):
    root = tmp_path / name
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"config":config or {"x":1}, "objective_measure":"constrained",
                                                   "inputs":[{"sha256":"same","bytes":1}],"sources":{}}))
    (root / "result.json").write_text(json.dumps({"state":"finished","objective":objective,"gradient_max":20.}))
    (root / "best_params.npz").write_bytes(name.encode())
    return root


def test_selects_best_start_of_same_model_without_claiming_convergence(tmp_path):
    a, b = candidate(tmp_path,"a",3.), candidate(tmp_path,"b",2.)
    selected, rows = select_candidate([a,b])
    assert selected["directory"] == str(b) and selected["gradient_max"] == 20.
    assert len(rows) == 2


def test_refuses_cross_model_posterior_ranking(tmp_path):
    a, b = candidate(tmp_path,"a",3.), candidate(tmp_path,"b",2.,{"x":2})
    with pytest.raises(ValueError,match="differ"):
        select_candidate([a,b])


def test_refuses_nonfinite_candidate(tmp_path):
    with pytest.raises(ValueError,match="Nonfinite"):
        select_candidate([candidate(tmp_path,"a",math.nan)])
