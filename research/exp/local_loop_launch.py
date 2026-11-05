"""Detach a registered supervisor; keep no dependency on the chat tool session."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

try:
    from local_loop_supervisor import live, process_identity, matching_leases, atomic_json
except ModuleNotFoundError:
    from research.exp.local_loop_supervisor import live, process_identity, matching_leases, atomic_json


def predecessor_ready(path):
    if not path.is_file():
        return False
    state = json.loads(path.read_text())
    if live(state.get("supervisor")):
        return False
    if state.get("phase") != "queue_drained":
        raise RuntimeError("Predecessor stopped without draining; inspect/adopt its work before chaining")
    for record in path.parent.glob("*.worker.json"):
        identities = json.loads(record.read_text())
        if live(identities.get("worker")) or live(identities.get("child")):
            return False
    if matching_leases({"tokens":["python"],"option":"--out","prefix":str(path.parent.parent)+"/"}):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--after-state",type=Path)
    parser.add_argument("--wait-detached",action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    state = args.out / "state.json"
    if state.exists():
        previous = json.loads(state.read_text())
        if live(previous.get("supervisor")):
            print(json.dumps({"state":"already_live","identity":previous["supervisor"]}))
            return
    if args.wait_detached:
        record = args.out / "activation_waiter.json"
        if record.exists() and live(json.loads(record.read_text()).get("waiter")):
            print(json.dumps({"state":"waiter_already_live"}))
            return
        command = [sys.executable,str(Path(__file__)),"--plan",str(args.plan),"--out",str(args.out)]
        if args.after_state:
            command += ["--after-state",str(args.after_state)]
        with (args.out / "activation_waiter.log").open("a") as log:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity = process_identity(child.pid)
        atomic_json(record,{"waiter":identity,"command":command})
        print(json.dumps({"state":"chained_activation_armed","waiter":identity}))
        return
    if args.after_state:
        while not predecessor_ready(args.after_state):
            plan = json.loads(args.plan.read_text())
            if Path(plan["stop_path"]).exists():
                print(json.dumps({"state":"stop_requested_before_activation"}),flush=True)
                return
            time.sleep(10.)
    command = [sys.executable,str(Path(__file__).with_name("local_loop_supervisor.py")),"--plan",str(args.plan),"--out",str(args.out)]
    with (args.out / "supervisor.log").open("a") as log:
        child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(1.)
    if child.poll() is not None and child.returncode:
        raise RuntimeError(f"Supervisor launch failed; inspect {args.out / 'supervisor.log'}")
    print(json.dumps({"state":"launched_detached","identity":process_identity(child.pid),"command":command}))


if __name__ == "__main__":
    main()
