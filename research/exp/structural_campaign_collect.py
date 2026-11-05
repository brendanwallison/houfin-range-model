"""Collect terminal campaign children and compute readouts in an isolated tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def verify_model_sources(manifest, root):
    for name, identity in manifest["sources"].items():
        if not name.startswith("src."):
            continue
        original = Path(identity["path"])
        payload = original.read_bytes()
        if hashlib.sha256(payload).hexdigest() != identity["sha256"]:
            raise ValueError(f"Fitted source was modified: {original}")
        relative = Path(*original.parts[original.parts.index("src"):])
        current = (root / relative).read_bytes()
        if payload.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n"):
            raise ValueError(f"Readout tree model differs from fit: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=7200.)
    parser.add_argument("--cores", default="8-23")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    root = Path(os.environ["TAKEOVER_CODE_ROOT"]).resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    campaign = json.loads((args.campaign / "campaign_manifest.json").read_text())
    overlays = dict(campaign["cases"])
    started, attempted = time.monotonic(), set()
    while True:
        events = [json.loads(line) for line in (args.campaign / "events.jsonl").read_text().splitlines() if line.endswith("}")]
        for event in events:
            if event["state"] != "child_finished" or event["case"] in attempted:
                continue
            name = event["case"]
            attempted.add(name)
            destination = args.out / name
            if (destination / "readout.json").is_file():
                continue
            if event["exit_code"] != 0:
                print(json.dumps({"state": "fit_failed_no_readout", "case": name}), flush=True)
                continue
            manifest = json.loads((args.campaign / name / "manifest.json").read_text())
            verify_model_sources(manifest, root)
            env = dict(os.environ, SBX_DEVICE="cpu", SBX_CORES=args.cores, SBX_THREADS=str(args.threads),
                       AGE_MODEL_CONFIG=str(root / "config/overlays" / overlays[name]))
            command = ["bash", str(root / "research/bin/local_takeover_run.sh"),
                       "research/exp/structural_fit_readout.py", "--params", str(args.campaign / name / "best_params.npz"),
                       "--out", str(destination)]
            with (args.out / f"{name}.log").open("w") as log:
                print(json.dumps({"state": "collecting", "case": name, "command": command}), flush=True)
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(json.dumps({"state": "collected" if result.returncode == 0 else "readout_failed",
                              "case": name, "exit_code": result.returncode}), flush=True)
        terminal = any(e["state"] in ("campaign_finished", "aborted_source_changed") for e in events)
        if terminal or not args.watch or time.monotonic() - started >= args.max_seconds:
            print(json.dumps({"collector": "finished" if terminal else "observation_window_ended",
                              "campaign_terminal": terminal}), flush=True)
            return
        time.sleep(10.)


if __name__ == "__main__":
    main()
