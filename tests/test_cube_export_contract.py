import json
import sys

import numpy as np
import pytest

from src.community_encoder.cube_contract import export_contract, validate_cube_contract
from src.community_encoder.cube_contract import select_block_prefixes, validate_model_feature_contract


def winter_head(width=2):
    return {"label": "winter_half", "private": width, "private_only": True, "identity": True,
            "coord_target": {"path": "winter_coords.npz", "basis_dir": "winter_basis"}}


def test_sum_layout_describes_the_exported_array_and_rejects_old_width():
    c = export_contract(3, "summer_basis", 2, [winter_head()])
    assert c["kernel"] == "ruzicka_sum"
    assert c["latent_dim"] == 5
    assert [(b["start"], b["stop"]) for b in c["feature_blocks"]] == [(0, 3), (3, 5)]
    validate_cube_contract(c, 5)
    with pytest.raises(ValueError, match="array width"):
        validate_cube_contract(c, 3)
    c["feature_blocks"][1]["start"] = 2
    with pytest.raises(ValueError, match="overlapping"):
        validate_cube_contract(c)


def test_private_pair_head_cannot_claim_a_coordinate_basis():
    h = winter_head()
    del h["coord_target"]["basis_dir"]
    with pytest.raises(ValueError, match="provenance"):
        export_contract(3, "summer_basis", 2, [h])
    with pytest.raises(ValueError, match="private_dim"):
        export_contract(3, "summer_basis", 4, [winter_head()])
    assert export_contract(3, "summer_basis")["kernel"] == "ruzicka"
    validate_cube_contract({"kernel": "ruzicka", "centered": False, "latent_dim": 3}, 3)


def test_seasonal_prefixes_preserve_the_sum_kernel_and_reject_silent_cuts():
    c = export_contract(3, "summer", 2, [winter_head()])
    c["feature_prior"] = "isotropic"
    z = np.array([[1., 2., 3., 4., 5.], [2., 1., 7., 3., 8.]])
    selected = select_block_prefixes(c, [0, 1, 3])
    validate_model_feature_contract(selected, 3)
    np.testing.assert_array_equal(z[:, [0, 1, 3]] @ z[:, [0, 1, 3]].T,
                                  z[:, :2] @ z[:, :2].T + z[:, 3:4] @ z[:, 3:4].T)
    assert selected["source_latent_dim"] == 5
    assert [(b["start"], b["stop"]) for b in selected["feature_blocks"]] == [(0, 2), (2, 3)]
    with pytest.raises(ValueError, match="nonempty leading prefix"):
        select_block_prefixes(c, [0, 1, 2])
    with pytest.raises(ValueError, match="nonempty leading prefix"):
        select_block_prefixes(c, [1, 3])
    selected["truncation"] = "top_eigenfeatures"
    with pytest.raises(ValueError, match="positional"):
        validate_model_feature_contract(selected, 3)


def make_seed(tmp_path, name, z, contract=None):
    d = tmp_path / name
    d.mkdir()
    c = contract or export_contract(z.shape[-1], "summer_basis")
    c.update(years=[2020], temporal_output="raw_instantaneous",
             desk_checkpoint={"content_sha256": name}, kernel_contract=c["kernel_contract"])
    (d / "cube_meta.json").write_text(json.dumps(c))
    np.save(d / "Z_latent_2020.npy", z)
    return d


def test_mean_cube_preserves_contract_and_all_member_identities(tmp_path, monkeypatch):
    from research.exp.final_mean_cube import main
    z = np.random.default_rng(7).normal(size=(4, 5, 3)).astype("float32")
    a, b = make_seed(tmp_path, "a", z), make_seed(tmp_path, "b", z * 1.1)
    out = tmp_path / "mean"
    monkeypatch.setattr(sys, "argv", ["mean", "--cubes", str(a), str(b), "--out", str(out)])
    main()
    meta = json.loads((out / "cube_meta.json").read_text())
    validate_cube_contract(meta, 3)
    assert [m["desk_checkpoint"]["content_sha256"] for m in meta["ensemble_members"]] == ["a", "b"]
    np.testing.assert_allclose(np.load(out / "Z_latent_2020.npy"), z * 1.05, atol=2e-7)


def test_mean_rejects_bad_half_even_when_whole_vector_median_passes(tmp_path, monkeypatch):
    from research.exp.final_mean_cube import main
    z = np.random.default_rng(8).normal(size=(4, 5, 5)).astype("float32")
    c = export_contract(3, "summer_basis", 2, [winter_head()])
    a = make_seed(tmp_path, "a", z, c)
    other = z.copy()
    other[..., 3:] *= -1
    b = make_seed(tmp_path, "b", other, export_contract(3, "summer_basis", 2, [winter_head()]))
    out = tmp_path / "mean"
    monkeypatch.setattr(sys, "argv", ["mean", "--cubes", str(a), str(b), "--out", str(out)])
    with pytest.raises(SystemExit, match="not aligned"):
        main()
    assert not out.exists()


def test_mean_rejects_different_basis_before_writing(tmp_path, monkeypatch):
    from research.exp.final_mean_cube import main
    z = np.random.default_rng(2).normal(size=(4, 5, 3)).astype("float32")
    a = make_seed(tmp_path, "a", z)
    b = make_seed(tmp_path, "b", z, export_contract(3, "another_basis"))
    out = tmp_path / "mean"
    monkeypatch.setattr(sys, "argv", ["mean", "--cubes", str(a), str(b), "--out", str(out)])
    with pytest.raises(ValueError, match="basis identities"):
        main()
    assert not out.exists()


def test_shared_target_mean_requires_provenance_and_keeps_disagreement(tmp_path, monkeypatch):
    from research.exp.final_mean_cube import main
    z = np.random.default_rng(13).normal(size=(4, 5, 3)).astype("float32")
    a = make_seed(tmp_path, "a", z)
    b = make_seed(tmp_path, "b", -z)
    out = tmp_path / "mean"
    argv = ["mean", "--cubes", str(a), str(b), "--out", str(out), "--shared-coordinate-targets"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="hashed basis"):
        main()
    for d in (a, b):
        m = json.loads((d / "cube_meta.json").read_text())
        m["feature_blocks"][0]["basis_identity"] = {k: {"content_sha256": "same-target"} for k in
            ["meta.json", "esk_projmat.npy", "esk_landmarks.npy"]}
        (d / "cube_meta.json").write_text(json.dumps(m))
    main()
    np.testing.assert_allclose(np.load(out / "Z_latent_2020.npy"), 0.)
    m = json.loads((out / "cube_meta.json").read_text())
    assert m["coordinate_alignment"] == "shared_hashed_targets"
    assert m["seed_correlation_diagnostics"]["0-1"]["median"] < 0


def test_metadata_repair_checks_artifact_identity_and_keeps_backup(tmp_path, monkeypatch):
    from scripts import repair_cube_contract as r
    basis = tmp_path / "basis"
    basis.mkdir()
    (basis / "meta.json").write_text("{}")
    for n in ("esk_projmat.npy", "esk_landmarks.npy"):
        np.save(basis / n, np.eye(2))
    cube = tmp_path / "cube"
    cube.mkdir()
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint identity")
    np.savez(tmp_path / "desk_meta.npz", latent_dim=2, private_dim=0, esk_basis_dir=str(basis))
    np.save(cube / "Z_latent_2020.npy", np.ones((3, 4, 2)))
    old = {"kernel": "ruzicka", "centered": False, "latent_dim": 2, "years": [2020],
           "desk_checkpoint": r.identity(checkpoint), "desk_meta": r.identity(tmp_path / "desk_meta.npz")}
    meta = cube / "cube_meta.json"
    meta.write_text(json.dumps(old))
    original_bytes = meta.read_bytes()
    cfg = {"paths": {"desk_output_dir": str(tmp_path)},
           "latent_cube": {"model_path": str(checkpoint), "output_dir": str(cube), "z_dir": str(basis)}}
    monkeypatch.setattr(r, "load_config", lambda _: cfg)
    r.repair("test config")
    assert meta.read_bytes() == original_bytes
    corrected = r.repair("test config", write=True)
    assert corrected["feature_blocks"][0]["basis_identity"]["esk_projmat.npy"]["content_sha256"]
    assert (cube / "cube_meta.json.pre-contract-v2").read_bytes() == original_bytes
    r.repair("test config", write=True)  # idempotent
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="identity differs"):
        r.repair("test config", write=True)
