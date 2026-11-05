"""Durable WSL task graph: adopt live work, reserve lanes, never duplicate a fit.

All commands/designs are explicit in a reviewed plan. This dispatcher does not
invent scientific conclusions or restart failed scientific work automatically.
Workers and the supervisor can detach from the tool connection independently.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        return {"pid": pid, "start_ticks": int(fields[19]), "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(identity):
    current = process_identity(identity["pid"]) if identity else None
    return bool(current and current["start_ticks"] == identity["start_ticks"] and current["state"] != "Z")


def argv_matches(argv, spec):
    tokens = spec["tokens"] if isinstance(spec,dict) else spec
    if not all(any(token in arg for arg in argv) for token in tokens):
        return False
    if isinstance(spec,dict) and spec.get("option"):
        return any(arg == spec["option"] and i+1 < len(argv) and argv[i+1].startswith(spec["prefix"])
                   for i,arg in enumerate(argv))
    return True


def matching_leases(spec):
    identities = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path / "cmdline").read_bytes().decode().split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if argv_matches(argv,spec):
            identity = process_identity(int(path.name))
            if identity and identity["state"] != "Z":
                identities.append(identity)
    return identities


def conditions_pass(conditions):
    for condition in conditions:
        path = Path(condition["path"])
        if not path.is_file():
            return False
        if condition.get("jsonl_state"):
            try:
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.endswith("}")]
            except ValueError:
                return False
            if not any(row.get("state") == condition["jsonl_state"] for row in rows):
                return False
        if "equals" in condition or condition.get("finite"):
            try:
                data = json.loads(path.read_text())
                for key, expected in condition.get("equals", {}).items():
                    value = data
                    for part in key.split("."):
                        value = value[part]
                    if value != expected:
                        return False
                for key in condition.get("finite", []):
                    value = data
                    for part in key.split("."):
                        value = value[part]
                    if not math.isfinite(float(value)):
                        return False
            except (ValueError, KeyError, TypeError):
                return False
    return True


def guard_sources(guards):
    for filename, expected in guards.items():
        path = Path(filename)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source/config drift: {filename}")


def acquire_lock(path):
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another autonomous supervisor owns this workspace")
    return handle


def worker(spec_path, record_path, receipt_path):
    task = json.loads(spec_path.read_text())
    started = time.time()
    atomic_json(record_path, {"worker": process_identity(os.getpid()), "started_unix": started})
    code, reason = 1, None
    try:
        guard_sources(task.get("guards", {}))
        child = subprocess.Popen(task["argv"], cwd=task.get("cwd"), env={**os.environ, **task.get("env", {})})
        atomic_json(record_path, {"worker": process_identity(os.getpid()), "child": process_identity(child.pid),
                                  "started_unix": started, "command": task["argv"]})
        code = child.wait()
        if code == 0 and not conditions_pass(task.get("complete", [])):
            reason, code = "Command exited zero but completion checks failed", 1
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
    atomic_json(receipt_path, {"state": "finished" if code == 0 else "failed", "returncode": code,
                               "reason": reason, "started_unix": started, "finished_unix": time.time()})


class Supervisor:
    def __init__(self, plan_path, directory):
        self.plan_path, self.directory = plan_path, directory
        self.plan = json.loads(plan_path.read_text())
        self.tasks = self.plan["tasks"]
        names = [task["id"] for task in self.tasks]
        if len(names) != len(set(names)) or any(not re.fullmatch(r"[a-z0-9_]+", name) for name in names):
            raise ValueError("Task IDs must be unique safe names")
        if any(not set(t.get("needs", [])).issubset(names) for t in self.tasks):
            raise ValueError("Unknown task dependency")
        graph = {t["id"]: t.get("needs", []) for t in self.tasks}
        def visit(name, active, done):
            if name in active:
                raise ValueError("Cyclic task dependency")
            if name not in done:
                for parent in graph[name]:
                    visit(parent, active | {name}, done)
                done.add(name)
        done = set()
        for name in names:
            visit(name, set(), done)
        if any(t.get("adopt") and not t.get("complete") for t in self.tasks):
            raise ValueError("Adoption requires explicit completion evidence")
        directory.mkdir(parents=True, exist_ok=True)
        self.identity = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        state_path = directory / "state.json"
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {
            "plan_sha256": self.identity, "started_unix": time.time(), "tasks": {}}
        if self.state["plan_sha256"] != self.identity:
            raise ValueError("Plan changed; use a new run directory and explicit adoption")
        self.lock = acquire_lock(Path(self.plan["lock_path"]))
        self.events = (directory / "events.jsonl").open("a")

    def emit(self, **row):
        self.events.write(json.dumps({"time_unix": time.time(), **row}) + "\n")
        self.events.flush()

    def inspect(self, task):
        name = task["id"]
        if task.get("adopt"):
            leases = matching_leases(task["leases"]) if task.get("leases") else []
            if leases:
                self.state["tasks"].setdefault(name,{})["observed_leases"] = leases
            if live(task["adopt"]) or leases:
                return "running"
            return "finished" if conditions_pass(task["complete"]) else "failed_external_incomplete"
        receipt = self.directory / f"{name}.receipt.json"
        record = self.directory / f"{name}.worker.json"
        if record.exists():
            identities = json.loads(record.read_text())
            if live(identities["worker"]) or live(identities.get("child")):
                return "running"
            if not receipt.exists() and conditions_pass(task.get("complete", [])) and task.get("complete"):
                atomic_json(receipt, {"state": "finished", "returncode": None,
                                      "reason": "Recovered explicit completion artifacts after worker/child termination",
                                      "finished_unix": time.time()})
        if receipt.exists():
            return json.loads(receipt.read_text())["state"]
        previous = self.state["tasks"].get(name, {})
        if live(previous.get("worker")):
            return "running"
        if previous.get("state") in {"dependency_failed", "budget_exhausted"}:
            return previous["state"]
        if previous.get("state", "").startswith("failed"):
            return previous["state"]
        if previous.get("state") in {"launch_intent", "running", "needs_investigation"}:
            # A worker might have started between durable intent and controller
            # bookkeeping. Never blindly relaunch this ambiguous crash window.
            return "needs_investigation"
        return "pending"

    def launch(self, task):
        guard_sources(task.get("guards", {}))
        if shutil.disk_usage(self.directory).free < self.plan.get("min_free_bytes", 5*1024**3):
            raise RuntimeError("Insufficient free output disk space")
        if self.plan.get("storage_root"):
            used = sum(p.stat().st_size for p in Path(self.plan["storage_root"]).rglob("*") if p.is_file() and not p.is_symlink())
            if used >= self.plan["max_storage_bytes"]:
                raise RuntimeError("Registered output-storage budget exhausted")
        name = task["id"]
        spec = self.directory / f"{name}.task.json"
        atomic_json(spec, task)
        self.state["tasks"][name] = {"state": "launch_intent", "started_unix": time.time()}
        atomic_json(self.directory / "state.json", self.state)
        with (self.directory / f"{name}.log").open("a") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(spec),
                                      "--record", str(self.directory / f"{name}.worker.json"),
                                      "--receipt", str(self.directory / f"{name}.receipt.json")],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.state["tasks"][name]["worker"] = process_identity(child.pid)
        self.state["tasks"][name]["state"] = "running"
        self.emit(state="launched", task=name, worker=self.state["tasks"][name]["worker"])

    def tick(self):
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        states = {task["id"]: self.inspect(task) for task in self.tasks}
        busy = {task["lane"] for task in self.tasks if states[task["id"]] == "running"}
        gpu_seconds = 0.
        for task in self.tasks:
            name = task["id"]
            old = self.state["tasks"].get(name, {})
            if task["lane"] == "gpu" and old.get("started_unix"):
                end = old.get("finished_unix", time.time())
                gpu_seconds += end - old["started_unix"]
            if old.get("state") != states[name]:
                self.emit(state=states[name], task=name)
            self.state["tasks"][name] = {**old, "state": states[name]}
            if states[name] == "running":
                self.state["tasks"][name].setdefault("started_unix", time.time())
            elif states[name] not in ("pending",):
                self.state["tasks"][name].setdefault("finished_unix", time.time())
        for task in self.tasks:
            name = task["id"]
            if states[name] != "pending" or task["lane"] in busy:
                continue
            dependencies = [states[d] for d in task.get("needs", [])]
            if any(s.startswith("failed") or s in {"needs_investigation", "dependency_failed", "budget_exhausted"} for s in dependencies):
                self.state["tasks"][name]["state"] = "dependency_failed"
                continue
            if any(s != "finished" for s in dependencies):
                continue
            if task["lane"] == "gpu" and gpu_seconds >= self.plan["gpu_budget_seconds"]:
                self.state["tasks"][name]["state"] = "budget_exhausted"
                continue
            # inspect() distinguishes pending from launch intent on restart.
            self.state["tasks"].pop(name, None)
            try:
                self.launch(task)
                busy.add(task["lane"])
            except Exception as error:
                self.state["tasks"][name] = {"state": "failed_to_launch", "reason": str(error)}
                self.emit(state="failed_to_launch", task=name, reason=str(error))
        self.state.update(phase="running", checked_unix=time.time(), supervisor=process_identity(os.getpid()), gpu_seconds=gpu_seconds)
        atomic_json(self.directory / "state.json", self.state)
        return self.state

    def close(self):
        self.events.close()
        self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.record, args.receipt)
        return
    supervisor = Supervisor(args.plan, args.out)
    try:
        while True:
            if Path(supervisor.plan["stop_path"]).exists():
                supervisor.emit(state="stop_requested_no_new_launches")
                supervisor.state["phase"] = "stop_requested_no_new_launches"
                atomic_json(supervisor.directory / "state.json", supervisor.state)
                return
            state = supervisor.tick()
            statuses = [t["state"] for t in state["tasks"].values()]
            if not any(s in ("pending", "running", "launch_intent") for s in statuses):
                supervisor.emit(state="queue_drained", task_states={k:v["state"] for k,v in state["tasks"].items()})
                supervisor.state["phase"] = "queue_drained"
                atomic_json(supervisor.directory / "state.json", supervisor.state)
                return
            time.sleep(10.)
    finally:
        supervisor.close()


if __name__ == "__main__":
    main()
