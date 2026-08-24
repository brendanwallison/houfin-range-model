"""Tests for the DESK architecture's width knobs and their checkpoint round-trip.

``hidden_width`` and ``mlp_expansion`` set ``state_dict`` SHAPES, so they are persisted in
``desk_meta.npz`` and read back everywhere the net is rebuilt for inference. Making
``hidden_width`` accept a per-stream sequence turns that into a scalar-or-list distinction, and
the failure mode is specific: ``np.savez`` stores an int as a 0-d array and a list as a 1-d
one, and ``int()`` on a 1-d array raises only for length > 1. So a one-stream list converts
silently while a six-stream list crashes at inference -- two GPU-hours after the run that
produced it, in a stage that only runs on the cluster.
"""
import ast
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.model_arch import (  # noqa: E402
    MultiStreamAutoencoder, hidden_width_from_meta, resolve_hidden_widths)

DIMS = [240, 33, 3, 24, 16, 3]          # climate, landuse, hyde, bui, soil, elevation
LATENT = 64


def _save_and_reload(tmp_path, model, name):
    """Persist the way ``desk_training`` does, then rebuild from the meta and load the weights.

    This is the contract that matters: the meta must be sufficient to reconstruct a net the
    checkpoint fits. Anything less and the cube stage rebuilds a differently-sized net and the
    load fails -- after the training that produced the checkpoint has already been paid for.
    """
    path = tmp_path / f"{name}.npz"
    np.savez(path,
             hidden_width=(np.int64(model.hidden_widths[0]) if model.hidden_width is not None
                           else np.array(model.hidden_widths, dtype=np.int64)),
             mlp_expansion=int(model.mlp_expansion),
             stream_dims=np.array(DIMS, int), latent_dim=LATENT, spatial_kernel=5)
    dm = np.load(path, allow_pickle=True)
    rebuilt = MultiStreamAutoencoder(
        [int(d) for d in dm["stream_dims"]], int(dm["latent_dim"]),
        int(dm["spatial_kernel"]),
        hidden_width=hidden_width_from_meta(dm),
        mlp_expansion=int(dm["mlp_expansion"]))
    rebuilt.load_state_dict(model.state_dict())        # raises on any shape disagreement
    return rebuilt


def test_a_uniform_width_still_persists_as_a_scalar(tmp_path):
    """A uniform net's meta must hold a SCALAR, so nothing that predates the list form breaks.

    Persisting every net as an array would make new uniform checkpoints unreadable by any
    reader still doing ``int(dm["hidden_width"])`` -- including anything outside this repo, and
    including an older checkout used to re-run a comparison.
    """
    m = MultiStreamAutoencoder(DIMS, LATENT, 5, dropout=0.1, hidden_width=128)
    assert m.hidden_width == 128 and m.hidden_widths == [128] * 6
    path = tmp_path / "u.npz"
    np.savez(path, hidden_width=np.int64(m.hidden_widths[0]))
    dm = np.load(path)
    assert np.asarray(dm["hidden_width"]).ndim == 0, "uniform must persist as a 0-d scalar"
    assert hidden_width_from_meta(dm) == 128
    assert int(dm["hidden_width"]) == 128, "the pre-existing int() reader must still work"
    _save_and_reload(tmp_path, m, "u2")
    print("uniform width persists and reloads as a scalar")


def test_per_stream_widths_round_trip_through_the_meta(tmp_path):
    """The list form must survive ``np.savez`` and rebuild a net the checkpoint fits."""
    widths = [256, 128, 64, 64, 64, 64]
    m = MultiStreamAutoencoder(DIMS, LATENT, 5, dropout=0.1, hidden_width=widths)
    assert m.hidden_widths == widths
    # hidden_width is None, not one of the widths: there is no single width to report, and
    # returning one of them would be a number that silently misdescribes the net.
    assert m.hidden_width is None
    rebuilt = _save_and_reload(tmp_path, m, "ps")
    assert rebuilt.hidden_widths == widths
    # functional at the reconstructed width
    x = torch.randn(1, 4, 5, sum(DIMS)); msk = torch.ones(1, 4, 5, dtype=torch.bool)
    rebuilt.eval()
    with torch.no_grad():
        z, rec = rebuilt(x, msk)
    assert z.shape == (1, 4, 5, LATENT) and rec.shape == x.shape
    print(f"per-stream widths {widths} round-trip and rebuild")


def test_an_old_checkpoint_with_no_width_key_still_loads(tmp_path):
    """A meta predating the capacity knob has no ``hidden_width`` at all.

    It must resolve to the historical default ``max(128, latent_dim*4)``, not to an error and
    not to a guess -- those checkpoints exist and the comparisons that used them have to stay
    reproducible.
    """
    path = tmp_path / "old.npz"
    np.savez(path, stream_dims=np.array(DIMS, int), latent_dim=LATENT)
    dm = np.load(path)
    assert hidden_width_from_meta(dm) is None
    m = MultiStreamAutoencoder(DIMS, LATENT, 5, hidden_width=hidden_width_from_meta(dm))
    assert m.hidden_widths == [max(128, LATENT * 4)] * 6 == [256] * 6
    print("a pre-capacity-knob meta resolves to the historical default")


def test_a_mismatched_width_rebuild_is_rejected_not_silently_accepted(tmp_path):
    """Loading per-stream weights into a uniform net must RAISE.

    This is the whole reason the widths are persisted. If the shapes happened to be compatible
    the load would succeed and every channel would carry another channel's weights -- the same
    invisible class as the 94-of-96 species-column misalignment.
    """
    ps = MultiStreamAutoencoder(DIMS, LATENT, 5, hidden_width=[256, 128, 64, 64, 64, 64])
    uni = MultiStreamAutoencoder(DIMS, LATENT, 5, hidden_width=128)
    with pytest.raises(RuntimeError):
        uni.load_state_dict(ps.state_dict())
    with pytest.raises(RuntimeError):
        ps.load_state_dict(uni.state_dict())
    print("a width mismatch is a hard load failure")


def test_the_mixer_and_decoder_are_sized_on_the_width_sum(tmp_path):
    """``sum(hidden_widths)``, not ``n * h`` -- and identical to it when the widths are uniform.

    The mixer reads the CONCATENATED branch codes, so its input width is the sum. Using
    ``n * h`` with per-stream widths would build a mixer of the wrong width; using the sum for a
    uniform net must reproduce the old shape exactly, or every existing checkpoint stops
    loading.
    """
    uni = MultiStreamAutoencoder(DIMS, LATENT, 5, hidden_width=128)
    assert uni.mixer[0].in_features == 6 * 128 == sum(uni.hidden_widths)
    ps = MultiStreamAutoencoder(DIMS, LATENT, 5, hidden_width=[256, 128, 64, 64, 64, 64])
    assert ps.mixer[0].in_features == sum(ps.hidden_widths) == 640
    assert ps.decoder[0].out_features == 640
    # each branch's input projection matches ITS stream's channel count and ITS width
    for enc, d, h in zip(ps.encoders, DIMS, ps.hidden_widths):
        assert enc[0].in_features == d and enc[0].out_features == h
    print("mixer/decoder sized on sum(hidden_widths); uniform case unchanged")


def test_every_meta_width_reader_goes_through_the_one_helper():
    """A static check, because these readers only ever run on the cluster.

    Three call sites read ``hidden_width`` back out of ``desk_meta.npz``: the trainer, the cube
    builder, and the spacetime validator. ``int(dm["hidden_width"])`` works for a scalar and
    raises for a per-stream list, so a reader that reverts to it trains fine and then fails
    every downstream stage. There is no local fixture that exercises the cube or the validator,
    so the contract is checked in the source instead -- the same approach
    ``test_every_desk_z_ema_call_site_unpacks_two_values`` takes.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # int(dm["hidden_width"]) / int(meta["hidden_width"]) in any subscript form
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "int"
                    and len(node.args) == 1 and isinstance(node.args[0], ast.Subscript)):
                continue
            sl = node.args[0].slice
            if isinstance(sl, ast.Constant) and sl.value == "hidden_width":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "hidden_width may be a per-stream list; int() on a length-6 array raises. Use "
        "hidden_width_from_meta(). Offending sites: " + ", ".join(offenders))
    # and the two inference readers must actually import the helper
    for rel in ("community_encoder/build_final_z_cube.py",
                "community_encoder/train_DESK/validate_spacetime.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "hidden_width_from_meta" in src, rel
    print("all meta width readers go through hidden_width_from_meta")


def test_resolve_is_the_single_resolver_for_all_three_shapes():
    """Scalar, sequence and None arrive from a config, a constructor and a meta.

    Resolving the shape separately in each is how the distinction becomes a silent shape bug: a
    length-1 list broadcasts in one place and indexes out of range in another.
    """
    assert resolve_hidden_widths(None, 6, 64) == [256] * 6
    assert resolve_hidden_widths(0, 6, 64) == [256] * 6          # falsy scalar -> default
    assert resolve_hidden_widths(96, 3, 64) == [96, 96, 96]
    assert resolve_hidden_widths((8, 9), 2, 64) == [8, 9]        # tuples too
    with pytest.raises(ValueError, match="positional"):
        resolve_hidden_widths([1, 2, 3], 6, 64)
    with pytest.raises(ValueError, match="positional"):
        resolve_hidden_widths([], 6, 64)                          # empty is a mistake, not None
    print("one resolver handles scalar, sequence and None")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
