"""Report plumbing for validate_spacetime."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.validate_spacetime import (
    RECON_ARRAY_KEYS, report_scalars)


def test_new_reconstruction_scalars_reach_the_report():
    """Regression. The key filter was an ALLOW-LIST, so the interpolation bar was computed,
    never listed, and could never print -- the printed summary reads this filtered dict, not the
    original. Any new scalar must survive by default."""
    recon = {"n": 10, "median_err_desk": 0.4, "frac_desk_beats_nochange": 0.63,
             "median_err_idw": 0.42, "frac_desk_beats_idw": 0.51,   # the two that went missing
             "some_future_bar": 1.23,
             **{k: np.zeros(10) for k in RECON_ARRAY_KEYS}}
    kept = report_scalars(recon)
    for k in ("median_err_idw", "frac_desk_beats_idw", "some_future_bar", "n"):
        assert k in kept, k


def test_the_per_point_arrays_stay_out_of_the_json():
    """They are large and go to the .npz for viz; in the report they would bloat it and break
    json.dumps."""
    import json
    kept = report_scalars({"n": 3, **{k: np.zeros(3) for k in RECON_ARRAY_KEYS}})
    assert set(kept) == {"n"}
    json.dumps(kept)


def test_every_desk_z_ema_call_site_unpacks_two_values():
    """A static check, because this bug class cannot be caught dynamically here.

    desk_z_ema returns ``(Z, metadata)``. Two call sites -- both added this session, in modules
    that need the full data tree to run -- bound the tuple to a single name and died only on
    TACC, one of them after burning two minutes of GPU time. There is no local fixture that
    would exercise them, so the contract is checked in the source instead.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "desk_z_ema"):
                continue
            tgt = node.targets[0]
            if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, ("desk_z_ema returns (Z, metadata); these bind it to one name: "
                          + ", ".join(offenders))
