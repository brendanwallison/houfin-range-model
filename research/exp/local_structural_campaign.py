"""Run registered structural comparisons with one GPU owner.

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
    parser.add_argument("--design", type=Path, help="Registered JSON with explicit name/overlay/initialization per case")
    args = parser.parse_args()
    root = Path(os.environ["TAKEOVER_CODE_ROOT"]).resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    code = fingerprints(root)
    settings = (json.loads(args.design.read_text())["cases"] if args.design else
                [{"name": "f0_d0", "overlay": "grid54_step43_factor_p2_veit_s41.json"},
                 {"name": "f2_d0", "overlay": "takeover_f2_independent.json"},
                 {"name": "f0_d1", "overlay": "takeover_f0_compact3.json"},
                 {"name": "f2_d1", "overlay": "takeover_f2_compact3.json"}])
    if len({case["name"] for case in settings}) != len(settings):
        raise ValueError("Duplicate campaign case names")
    cases = [(case["name"], case["overlay"]) for case in settings]
    manifest = {"code_commit": args.code_commit, "source_hashes": code, "cases": cases,
                "default_maxiter": args.maxiter, "default_seed": 20261105,
                "starts": "per case_settings; default is mapped reference; capped runs remain exploratory",
                "objective": "constrained-site MAP; raw data and priors reported separately",
                "pid": os.getpid(), "started_unix": time.time(), "command": sys.argv}
    manifest["case_settings"] = settings
    manifest["design_sha256"] = hashlib.sha256(args.design.read_bytes()).hexdigest() if args.design else None
    (args.out / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.out / "events.jsonl").open("x") as events:
        def event(**data):
            row = {"time_unix": time.time(), **data}
            events.write(json.dumps(row) + "\n")
            events.flush()
            print(json.dumps(row), flush=True)
        for setting in settings:
            name, overlay = setting["name"], setting["overlay"]
            if fingerprints(root) != code:
                event(state="aborted_source_changed", next_case=name)
                raise SystemExit(3)
            env = dict(os.environ, SBX_DEVICE="gpu", SBX_CORES="0-7", SBX_THREADS="8",
                       AGE_MODEL_CONFIG=str(root / "config/overlays" / overlay))
            cmd = ["bash", str(root / "research/bin/local_takeover_run.sh"),
                   "research/exp/local_joint_fit.py", "--out", str(args.out / name),
                   "--code-commit", args.code_commit, "--maxiter", str(setting.get("maxiter", args.maxiter)),
                   "--seed", str(setting.get("seed", 20261105)), "--jitter", str(setting.get("jitter", 0.))]
            if setting.get("init"):
                initial = Path(os.path.expandvars(setting["init"]))
                if not initial.is_file():
                    raise ValueError(f"Missing initial checkpoint: {initial}")
                cmd += ["--init", str(initial)]
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
