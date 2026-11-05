"""Detach a registered supervisor; keep no dependency on the chat tool session."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from local_loop_supervisor import live, process_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    state = args.out / "state.json"
    if state.exists():
        previous = json.loads(state.read_text())
        if live(previous.get("supervisor")):
            print(json.dumps({"state":"already_live","identity":previous["supervisor"]}))
            return
    command = [sys.executable,str(Path(__file__).with_name("local_loop_supervisor.py")),"--plan",str(args.plan),"--out",str(args.out)]
    with (args.out / "supervisor.log").open("a") as log:
        child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(1.)
    if child.poll() is not None and child.returncode:
        raise RuntimeError(f"Supervisor launch failed; inspect {args.out / 'supervisor.log'}")
    print(json.dumps({"state":"launched_detached","identity":process_identity(child.pid),"command":command}))


if __name__ == "__main__":
    main()
