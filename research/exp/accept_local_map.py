"""Validate an isolated local-MAP upload before exposing it to a TACC worker.

Runs on the login node without importing JAX. No fingerprint exception is used.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import numpy as np

try:
    from .range_watch import HostUnpickler
except ImportError:
    from range_watch import HostUnpickler


def identity(path):
    size=path.stat().st_size
    with path.open("rb") as stream:
        if size<=64*1024*1024:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        else:
            h=hashlib.sha256(); h.update(stream.read(1024*1024))
            stream.seek(size-1024*1024); h.update(stream.read(1024*1024)); digest=h.hexdigest()
    return {"name":path.name,"size":size,"content_sha256":digest}


def accept(upload,step,root):
    synthetic_steps=("step31_nestsub","step31_nestdrop")
    real_steps=("step33_nestcbc_s41","step33_nestcbc_natdep_s41")
    if step not in synthetic_steps+real_steps:
        raise ValueError("run is not in the registered local offload bundle")
    code=root/"research/grid54/runs"/step/"code"
    os.environ["AGE_MODEL_CONFIG"]=str(code/"config/overlays"/f"grid54_{step}.json")
    os.environ["HOUFIN_PROCESSED"]=str(root/"processed")
    sys.path.insert(0,str(code))
    from src.config_utils import load_age_model_config
    cfg=load_age_model_config()
    transfer=json.loads((upload/"local_map_transfer.json").read_text())
    with (upload/"map_checkpoint.pkl").open("rb") as stream:
        ck=HostUnpickler(stream).load()
    if ck.get("format_version")!=2 or ck.get("step")!=400 or len(ck.get("losses",[]))!=400:
        raise ValueError("MAP checkpoint is incomplete")
    if not np.isfinite(np.asarray(ck["losses"])).all():
        raise ValueError("nonfinite MAP losses")
    if not ck.get("params") or any(not name.endswith("_auto_loc") or not np.isfinite(np.asarray(value)).all()
                                   for name,value in ck["params"].items()):
        raise ValueError("missing or invalid AutoDelta MAP parameters")
    payload=ck["fingerprint_payload"]
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":"),default=list)
    if hashlib.sha256(canonical.encode()).hexdigest()!=ck["fingerprint"]:
        raise ValueError("checkpoint fingerprint payload is inconsistent")
    if payload["precision"]!="float32" or payload["map_profile"]!="quick90" or payload["age_model_config"]!=cfg:
        raise ValueError("precision/profile/configuration mismatch")
    for package,expected in payload["versions"].items():
        if version(package)!=expected:
            raise ValueError(f"library version differs: {package}")
    for item in payload["sources"]:
        if identity(code/"src/model"/item["name"])!=item:
            raise ValueError(f"source differs: {item['name']}")
    for item in payload["inputs"]:
        if identity(Path(cfg["input_dir"])/item["name"])!=item:
            raise ValueError(f"input differs: {item['name']}")
    checks=[("nests",Path(cfg["population_model"]["nest_likelihood"]["path"])),
            ("code_manifest",code.parent/"CODE_MANIFEST.sha256")]
    if step in synthetic_steps:
        checks.append(("synthetic",root/"research/grid54/synthetic/yb_j1_null_h0.npz"))
    elif "synthetic" in transfer:
        raise ValueError("real-data candidate must not carry synthetic observations")
    for key,path in checks:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()!=transfer[key]:
                raise ValueError(f"auxiliary input differs: {key}")
    destination=root/"processed/model_results"/f"age_map_float32_grid54_{step}_quick90"
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing MAP directory: {destination}")
    os.rename(upload,destination)
    out=root/"research/grid54/runs"/step/"out"
    out.mkdir(exist_ok=True)
    (out/"MAP_DONE").write_text("Completed float32 on local RTX 3090; cross-machine fingerprint and auxiliary hashes verified.\n")
    print(f"Accepted {step}; TACC worker retains diagnostics and float64 refinement")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("upload",type=Path)
    p.add_argument("step")
    p.add_argument("--root",type=Path,default=Path("/work/07980/bwa386/ls6/houfin"))
    args=p.parse_args(); accept(args.upload,args.step,args.root)
