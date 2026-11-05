"""Run the registered first-start structural screens with one GPU owner.

Source/config changes abort the remaining queue instead of silently importing
different code. Completed/failed children keep their own finite checkpoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def fingerprints(root):
    paths = list((root / "src").rglob("*.py")) + list((root / "config").rglob("*.json"))
    paths += list((root / "research/exp/invasion_sandbox").glob("*.py"))
    paths += list((root / "research/exp").glob("local_*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--maxiter", type=int, default=500)
    args = parser.parse_args()
    root = Path(os.environ["TAKEOVER_CODE_ROOT"]).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    code = fingerprints(root)
    cases = [("f0_d0", "grid54_step43_factor_p2_veit_s41.json"),
             ("f2_d0", "takeover_f2_independent.json"),
             ("f0_d1", "takeover_f0_compact3.json"),
             ("f2_d1", "takeover_f2_compact3.json")]
    manifest = {"code_commit": args.code_commit, "source_hashes": code, "cases": cases,
                "maxiter_per_case": args.maxiter, "seed": 20261105, "starts": "mapped reference only; screening",
                "objective": "constrained-site MAP; raw data and priors reported separately",
                "pid": os.getpid(), "started_unix": time.time(), "command": sys.argv}
    (args.out / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.out / "events.jsonl").open("x") as events:
        def event(**data):
            row = {"time_unix": time.time(), **data}
            events.write(json.dumps(row) + "\n")
            events.flush()
            print(json.dumps(row), flush=True)
        for name, overlay in cases:
            if fingerprints(root) != code:
                event(state="aborted_source_changed", next_case=name)
                raise SystemExit(3)
            env = dict(os.environ, SBX_DEVICE="gpu", SBX_CORES="0-7", SBX_THREADS="8",
                       AGE_MODEL_CONFIG=str(root / "config/overlays" / overlay))
            cmd = ["bash", str(root / "research/bin/local_takeover_run.sh"),
                   "research/exp/local_joint_fit.py", "--out", str(args.out / name),
                   "--code-commit", args.code_commit, "--maxiter", str(args.maxiter)]
            with (args.out / f"{name}.log").open("x") as log:
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                event(state="running", case=name, pid=child.pid, command=cmd)
                code_result = child.wait()
            result_path = args.out / name / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            event(state="child_finished", case=name, exit_code=code_result, result=result)
        event(state="campaign_finished")


if __name__ == "__main__":
    main()
