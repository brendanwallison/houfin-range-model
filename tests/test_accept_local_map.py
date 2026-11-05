import hashlib
import json
import pickle
import sys
import types

import numpy as np
import pytest

from research.exp.accept_local_map import accept, identity


@pytest.fixture
def upload(tmp_path,monkeypatch,request):
    root=tmp_path/"houfin"; step=getattr(request,'param',"step31_nestsub")
    code=root/"research/grid54/runs"/step/"code"
    source=code/"src/model"; source.mkdir(parents=True)
    (source/"age_run_map.py").write_text("frozen source\n")
    (code.parent/"CODE_MANIFEST.sha256").write_text("manifest\n")
    synthetic=root/"research/grid54/synthetic/yb_j1_null_h0.npz"
    synthetic.parent.mkdir(); synthetic.write_bytes(b"synthetic")
    inputs=root/"processed/inputs"; inputs.mkdir(parents=True)
    (inputs/"metadata.pkl").write_bytes(b"inputs")
    nests=root/"processed/nests.npz"; nests.write_bytes(b"nests")
    (root/"processed/model_results").mkdir()
    cfg={"input_dir":str(inputs),"population_model":{"nest_likelihood":{"path":str(nests)}}}
    module=types.ModuleType("src.config_utils"); module.load_age_model_config=lambda:cfg
    monkeypatch.setitem(sys.modules,"src.config_utils",module)
    monkeypatch.syspath_prepend(str(tmp_path))  # restore sys.path after accept inserts code
    payload={"precision":"float32","map_profile":"quick90","age_model_config":cfg,
             "versions":{},"sources":[identity(source/"age_run_map.py")],"inputs":[identity(inputs/"metadata.pkl")]}
    checkpoint={"format_version":2,"step":400,"losses":np.ones(400),"params":{"alpha_a_auto_loc":np.array(.3)},"fingerprint_payload":payload,
                "fingerprint":hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=list).encode()).hexdigest()}
    stage=root/"upload"; stage.mkdir()
    with (stage/"map_checkpoint.pkl").open("wb") as stream: pickle.dump(checkpoint,stream)
    transfer={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in
              (("synthetic",synthetic),("nests",nests),("code_manifest",code.parent/"CODE_MANIFEST.sha256"))}
    (stage/"local_map_transfer.json").write_text(json.dumps(transfer))
    return root,step,stage,nests,source


def test_accepts_complete_matching_upload_before_marking_done(upload):
    root,step,stage,_,_=upload
    accept(stage,step,root)
    assert not stage.exists()
    assert (root/"research/grid54/runs"/step/"out/MAP_DONE").exists()


@pytest.mark.parametrize("changed",["nests","source"])
def test_mismatch_never_publishes_checkpoint_or_marker(upload,changed):
    root,step,stage,nests,source=upload
    (nests if changed=="nests" else source/"age_run_map.py").write_bytes(b"changed")
    with pytest.raises(ValueError,match="differs"):
        accept(stage,step,root)
    assert stage.exists()
    assert not (root/"research/grid54/runs"/step/"out/MAP_DONE").exists()


def test_does_not_replace_existing_remote_map(upload):
    root,step,stage,_,_=upload
    destination=root/"processed/model_results"/f"age_map_float32_grid54_{step}_quick90"
    destination.mkdir(); (destination/"keep").write_text("original")
    with pytest.raises(FileExistsError): accept(stage,step,root)
    assert (destination/"keep").read_text()=="original"


@pytest.mark.parametrize('upload',['step33_nestcbc_s41'],indirect=True)
def test_real_candidate_rejects_synthetic_transfer_then_accepts_real_data(upload):
    root,step,stage,_,_=upload
    with pytest.raises(ValueError,match='synthetic observations'):
        accept(stage,step,root)
    path=stage/'local_map_transfer.json'; info=json.loads(path.read_text()); del info['synthetic']; path.write_text(json.dumps(info))
    accept(stage,step,root)
    assert (root/'research/grid54/runs'/step/'out/MAP_DONE').exists()
