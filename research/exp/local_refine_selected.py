"""Select a comparable completed start and run its registered warm refinement."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def scientific_config(value):
    if isinstance(value, dict):
        return {k: scientific_config(v) for k,v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [scientific_config(v) for v in value]
    return value


def select_candidate(directories):
    rows, reference = [], None
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        model_sources = {}
        for name, source in manifest["sources"].items():
            if not name.startswith("src."):
                continue
            path = Path(source["path"])
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != source["sha256"]:
                raise ValueError(f"Fitted source changed: {path}")
            model_sources[name] = hashlib.sha256(payload.replace(b"\r\n",b"\n")).hexdigest()
            if os.environ.get("TAKEOVER_CODE_ROOT"):
                relative = Path(*path.parts[path.parts.index("src"):])
                current = (Path(os.environ["TAKEOVER_CODE_ROOT"]) / relative).read_bytes()
                if current.replace(b"\r\n",b"\n") != payload.replace(b"\r\n",b"\n"):
                    raise ValueError(f"Refinement model differs from selected fit: {relative}")
        identity = (scientific_config(manifest["config"]), manifest["objective_measure"],
                    [(p["sha256"], p["bytes"]) for p in manifest["inputs"]], model_sources,
                    manifest.get("versions"))
        if reference is not None and identity != reference:
            raise ValueError("Candidates differ in model configuration, objective measure or input identity")
        reference = identity
        if result["state"] != "finished" or not (directory / "best_params.npz").is_file():
            raise ValueError(f"Incomplete candidate {directory}")
        if not math.isfinite(result["objective"]) or not math.isfinite(result["gradient_max"]):
            raise ValueError(f"Nonfinite candidate {directory}")
        # The same constrained-site posterior can rank starts of this exact
        # model. It is never used to rank different field/onset families.
        rows.append({"directory": str(directory), "objective": result["objective"],
                     "gradient_max": result["gradient_max"],
                     "params_sha256": hashlib.sha256((directory / "best_params.npz").read_bytes()).hexdigest()})
    if not rows:
        raise ValueError("No candidate starts")
    return min(rows, key=lambda row: row["objective"]), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--maxiter", type=int, default=1500)
    args = parser.parse_args()
    selected, candidates = select_candidate(args.candidate)
    args.selection.write_text(json.dumps({"selected": selected, "candidates": candidates}, indent=2) + "\n")
    root = Path(os.environ["TAKEOVER_CODE_ROOT"])
    command = ["bash", str(root / "research/bin/local_takeover_run.sh"), "research/exp/local_joint_fit.py",
               "--init", str(Path(selected["directory"]) / "best_params.npz"), "--out", str(args.out),
               "--code-commit", args.code_commit, "--maxiter", str(args.maxiter)]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
