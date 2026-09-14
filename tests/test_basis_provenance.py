"""The basis a DESK checkpoint trained in must follow it to every consumer.

WHAT WENT WRONG, and why none of the existing guards saw it. Commit 0288236 pointed
``desk.z_dir`` and ``latent_cube.z_dir`` at a newly built basis while the production
checkpoint stayed where it was, one day older than that basis. For three weeks the config
described a system that did not exist on disk, and two full validation rounds ran under it.

Nothing could detect it:

  * ``desk_meta.npz`` recorded normalization, architecture, dropout and best epoch -- but not
    the basis, so the checkpoint could not self-report.
  * ``validate_spacetime``'s ``recent_basis_residual`` LOOKS like the check. It projects points
    through ``desk.z_dir`` and compares them to ``Z.npy`` loaded from that same directory, so it
    is identically zero for ANY basis. Its own docstring said it "confirms the basis matches".
  * The kernel contract in both bases' ``meta.json`` agrees, correctly, because both ARE valid
    Nystrom factorizations of the same Ruzicka kernel.

And the mismatch is not benign. Measured on 4,000 real points across the two bases: Gram
relative difference 0.016, median ``||z_A - z_B||`` at 1.087x ``||z||``, Procrustes residual
0.089. Same kernel, different coordinates, related by a rotation. Every metric built on
``||z - z'||`` then measures that rotation.

These tests pin the three properties that close it, each on the trap rather than the symptom.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.community_encoder.train_DESK.model_arch import (  # noqa: E402
    basis_dir_from_meta, check_basis_matches)


def _meta(tmp_path, **kw):
    """A desk_meta.npz stand-in: np.savez round-tripped, so dtypes match production."""
    p = tmp_path / "desk_meta.npz"
    np.savez(p, **kw)
    return np.load(p, allow_pickle=True)


def test_a_matching_basis_passes_and_returns_what_it_matched(tmp_path):
    dm = _meta(tmp_path, esk_basis_dir="/w/encoder/esk/spacetime")
    assert check_basis_matches(dm, "/w/encoder/esk/spacetime", context="t") == \
        "/w/encoder/esk/spacetime"


def test_a_different_basis_is_refused_and_the_error_names_both(tmp_path):
    """The exact production failure: checkpoint in one basis, config supplying another."""
    dm = _meta(tmp_path, esk_basis_dir="/w/encoder/esk/spacetime")
    with pytest.raises(ValueError) as e:
        check_basis_matches(dm, "/w/encoder/esk_balanced/spacetime", context="validate")
    msg = str(e.value)
    assert "esk/spacetime" in msg and "esk_balanced/spacetime" in msg, \
        "an operator cannot act on a mismatch that does not name both sides"


def test_a_trailing_slash_is_not_a_mismatch(tmp_path):
    """Paths arrive from a config and from os.path.join; normalization must not raise here.

    A guard that fires on a path spelling rather than a path would be turned off within a day,
    and then the real mismatch goes with it.
    """
    dm = _meta(tmp_path, esk_basis_dir="/w/encoder/esk/spacetime/")
    assert check_basis_matches(dm, "/w/encoder/esk//spacetime", context="t") is not None


def test_an_unrecorded_basis_warns_rather_than_raising(tmp_path, capsys):
    """Every checkpoint built before 2026-09-14 lacks the key, including the production one.

    Refusing them would strand the model this project is actually fitting. The requirement is
    that the output says the basis is UNVERIFIED -- the failure mode being guarded against is a
    reader inferring "no error, so the basis matched".
    """
    dm = _meta(tmp_path, latent_dim=64)
    assert basis_dir_from_meta(dm) is None
    assert check_basis_matches(dm, "/w/encoder/esk/spacetime", context="cube") is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "cannot be verified" in out
    assert "/w/encoder/esk/spacetime" in out, "the warning must name the basis being assumed"


def test_the_trainer_records_the_basis_it_projected_targets_through():
    """Pinned against the source, because the write and the read are in different modules.

    A test that only round-trips through `_meta` above would pass forever if desk_training
    stopped writing the key -- which is the state this whole failure came from.
    """
    src = os.path.join(os.path.dirname(__file__), "..", "src", "community_encoder",
                       "train_DESK", "desk_training.py")
    body = open(src, encoding="utf-8").read()
    assert "esk_basis_dir=str(z_dir)" in body, \
        "desk_meta.npz must record the ESK basis; without it a checkpoint's z has no frame"


@pytest.mark.parametrize("consumer", [
    "src/community_encoder/build_final_z_cube.py",
    "src/community_encoder/train_DESK/validate_spacetime.py",
    "src/community_encoder/train_DESK/validate_bbs_routes.py",
])
def test_every_desk_meta_consumer_checks_the_basis(consumer):
    """All three readers, not just the one that broke.

    The cube, the spacetime validator and the route validator each load desk_meta.npz and each
    compare z-coordinates. A guard on one of them leaves the other two able to produce a
    confident, wrong number.
    """
    path = os.path.join(os.path.dirname(__file__), "..", consumer)
    assert "check_basis_matches" in open(path, encoding="utf-8").read(), \
        f"{consumer} loads desk_meta.npz and compares z-coordinates, so it must verify the basis"


def test_the_committed_config_names_one_basis_everywhere():
    """The four independent copies of the basis path must agree in the config as shipped.

    This is the state that was wrong on disk for three weeks. `build_spacetime_cube` now
    enforces it at run time; this asserts the committed default already satisfies it, so the
    guard is not shipped pre-tripped.
    """
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "esk_desk_config.json")
    cfg = json.loads(open(cfg_path, encoding="utf-8").read())
    seen = {
        "desk.z_dir": cfg["desk"]["z_dir"],
        "latent_cube.z_dir": cfg["latent_cube"]["z_dir"],
        "latent_cube.z_ref_path": os.path.dirname(cfg["latent_cube"]["z_ref_path"]),
        "latent_cube.mask_ref_path": os.path.dirname(cfg["latent_cube"]["mask_ref_path"]),
    }
    assert len(set(seen.values())) == 1, f"config names more than one ESK basis: {seen}"


def test_the_cube_refuses_a_config_that_names_two_bases(tmp_path):
    """The run-time guard, driven through the real entry point rather than a copy of it.

    It must fire BEFORE any directory is created or any array read, since the whole point is to
    stop a cube that would be labelled with one basis and backfilled from another.
    """
    torch = pytest.importorskip("torch")                          # noqa: F841
    from src.community_encoder.build_final_z_cube import build_spacetime_cube
    out = tmp_path / "cube"
    cfg = {
        "paths": {"desk_output_dir": str(tmp_path / "desk")},
        "desk": {"z_dir": "/w/encoder/esk/spacetime"},
        "latent_cube": {
            "data_dir": str(tmp_path), "hist_dir": str(tmp_path),
            "z_dir": "/w/encoder/esk_balanced/spacetime",          # <- the disagreement
            "model_path": str(tmp_path / "m.pth"),
            "z_ref_path": "/w/encoder/esk/spacetime/Z.npy",
            "mask_ref_path": "/w/encoder/esk/spacetime/valid_mask.npy",
            "water_mask_path": str(tmp_path / "ocean.tif"),
            "output_dir": str(out),
        },
    }
    with pytest.raises(ValueError, match="named inconsistently"):
        build_spacetime_cube(cfg)
    assert not out.exists(), "the guard must fire before the cube directory is created"


def _basis_paths(cfg):
    """The four independent copies of the ESK basis path, as a dict of key -> directory."""
    return {
        "desk.z_dir": cfg["desk"]["z_dir"],
        "latent_cube.z_dir": cfg["latent_cube"]["z_dir"],
        "latent_cube.z_ref_path": os.path.dirname(cfg["latent_cube"]["z_ref_path"]),
        "latent_cube.mask_ref_path": os.path.dirname(cfg["latent_cube"]["mask_ref_path"]),
    }


def test_every_overlay_still_names_one_basis_AFTER_merging_onto_the_base():
    """The invariant is on the MERGED config, which is the only thing any stage ever sees.

    Checking an overlay in isolation gets this wrong in both directions: an overlay that pins
    only `desk.z_dir` (the tempho runs, which never build a cube) looks incomplete while being
    perfectly consistent, and an overlay that moves all four to a basis the BASE disagrees with
    looks fine while being the original defect.

    That defect, precisely: 0288236 moved both z_dirs and left z_ref_path/mask_ref_path behind.
    z_dir is only a provenance LABEL in the cube builder while z_ref_path is what the static
    backfill actually reads, so the keys it moved and the key that determines content were
    disjoint -- the cube would have been labelled with one basis and filled from another.

    Iterating every overlay rather than a named list is deliberate: the next one to get this
    wrong does not exist yet. The merge is the real `_deep_merge`, so this cannot drift from
    what `load_config` does.
    """
    from src.config_utils import _deep_merge
    root = os.path.join(os.path.dirname(__file__), "..")
    base = json.loads(open(os.path.join(root, "config", "esk_desk_config.json"),
                           encoding="utf-8").read())
    d = os.path.join(root, "config", "overlays")
    checked = []
    for name in sorted(n for n in os.listdir(d) if n.endswith(".json")):
        o = json.loads(open(os.path.join(d, name), encoding="utf-8").read())
        if "desk" not in o and "latent_cube" not in o:
            continue                                  # not an encoder overlay (e.g. map_*_z)
        seen = _basis_paths(_deep_merge(base, o))
        assert len(set(seen.values())) == 1, \
            f"{name} merged onto the base names more than one ESK basis: {seen}"
        checked.append(name)
    assert checked, "no overlay exercises this; the guard would be silently untested"


def test_the_tempho_overlays_pin_their_basis_rather_than_inheriting_it():
    """The three temporal-holdout runs must name the basis they were actually run on.

    They were run on esk_balanced (checkpoints 2026-08-21, graded 2026-09-01). Inheriting the
    basis from the base config is how the production checkpoint came to sit on a different one
    silently -- an overlay whose job is to reproduce a specific run cannot let the basis move
    underneath it when production later moves.
    """
    d = os.path.join(os.path.dirname(__file__), "..", "config", "overlays")
    for yr in (1975, 1985, 1995):
        o = json.loads(open(os.path.join(d, f"desk_tempho_{yr}.json"), encoding="utf-8").read())
        z = (o.get("desk") or {}).get("z_dir")
        assert z and "esk_balanced" in z, \
            f"desk_tempho_{yr}.json must pin the basis it ran on, got {z!r}"
