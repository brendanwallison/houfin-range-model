"""Real subprocess/restart tests for the durable experiment dispatcher."""
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from research.exp import local_loop_supervisor as loop


def setup_plan(tmp_path, tasks, budget=60):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"tasks": tasks, "lock_path": str(tmp_path / "owner.lock"),
                                "stop_path": str(tmp_path / "STOP"), "gpu_budget_seconds": budget,
                                "min_free_bytes": 0}))
    return plan, tmp_path / "state"


def task(tmp_path, name="job", delay=.05):
    output = tmp_path / f"{name}.done.json"
    code = ("import json,time; from pathlib import Path; "
            f"time.sleep({delay}); Path({str(output)!r}).write_text(json.dumps({{'state':'finished'}}))")
    return {"id": name, "lane": "gpu", "argv": [sys.executable, "-c", code],
            "complete": [{"path": str(output), "equals": {"state": "finished"}}]}


def wait_finished(supervisor, name):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = supervisor.tick()["tasks"][name]["state"]
        if state == "finished":
            return
        time.sleep(.02)
    raise AssertionError(f"Task did not finish: {supervisor.state}")


def test_restart_adopts_live_worker_without_duplicate_launch(tmp_path):
    plan, out = setup_plan(tmp_path, [task(tmp_path, delay=.3)])
    first = loop.Supervisor(plan, out)
    first.tick()
    first.close()
    second = loop.Supervisor(plan, out)
    try:
        wait_finished(second, "job")
        events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
        assert sum(e["state"] == "launched" for e in events) == 1
    finally:
        second.close()


def test_pending_dependency_survives_ticks_then_launches(tmp_path):
    marker = tmp_path / "external.done"
    adopted = {"id": "external", "lane": "gpu", "adopt": loop.process_identity(__import__('os').getpid()),
               "complete": [{"path": str(marker)}]}
    child = {**task(tmp_path), "needs": ["external"]}
    plan, out = setup_plan(tmp_path, [adopted, child])
    supervisor = loop.Supervisor(plan, out)
    try:
        assert supervisor.tick()["tasks"]["job"]["state"] == "pending"
        assert supervisor.tick()["tasks"]["job"]["state"] == "pending"
        supervisor.tasks[0]["adopt"] = {"pid": 999999999, "start_ticks": 0}
        marker.touch()
        wait_finished(supervisor, "job")
    finally:
        supervisor.close()


def test_failed_dependency_is_sticky_and_not_launched(tmp_path):
    adopted = {"id": "external", "lane": "gpu", "adopt": {"pid":999999999,"start_ticks":0},
               "complete": [{"path": str(tmp_path / "missing")}]}
    plan, out = setup_plan(tmp_path, [adopted, {**task(tmp_path), "needs": ["external"]}])
    supervisor = loop.Supervisor(plan, out)
    try:
        assert supervisor.tick()["tasks"]["job"]["state"] == "dependency_failed"
        assert supervisor.tick()["tasks"]["job"]["state"] == "dependency_failed"
        assert not (out / "job.worker.json").exists()
    finally:
        supervisor.close()


def test_ambiguous_launch_intent_never_relaunches(tmp_path):
    plan, out = setup_plan(tmp_path, [task(tmp_path)])
    supervisor = loop.Supervisor(plan, out)
    try:
        supervisor.state["tasks"]["job"] = {"state": "launch_intent"}
        assert supervisor.tick()["tasks"]["job"]["state"] == "needs_investigation"
        assert not (out / "job.worker.json").exists()
    finally:
        supervisor.close()


def test_source_drift_fails_task_without_running_it(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    command = task(tmp_path)
    command["guards"] = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    plan, out = setup_plan(tmp_path, [command])
    source.write_text("changed")
    supervisor = loop.Supervisor(plan, out)
    try:
        assert supervisor.tick()["tasks"]["job"]["state"] == "failed_to_launch"
        assert supervisor.tick()["tasks"]["job"]["state"] == "failed_to_launch"
        assert not (out / "job.worker.json").exists()
    finally:
        supervisor.close()


def test_single_writer_lock_and_gpu_budget(tmp_path):
    plan, out = setup_plan(tmp_path, [task(tmp_path)], budget=0)
    supervisor = loop.Supervisor(plan, out)
    try:
        with pytest.raises(RuntimeError, match="owns"):
            loop.Supervisor(plan, out)
        assert supervisor.tick()["tasks"]["job"]["state"] == "budget_exhausted"
        assert not (out / "job.worker.json").exists()
    finally:
        supervisor.close()


def test_pid_reuse_is_not_liveness(monkeypatch):
    monkeypatch.setattr(loop, "process_identity", lambda pid: {"pid":pid,"start_ticks":200,"state":"S"})
    assert not loop.live({"pid":123,"start_ticks":100})


def test_cycle_is_rejected(tmp_path):
    a, b = task(tmp_path, "a"), task(tmp_path, "b")
    a["needs"], b["needs"] = ["b"], ["a"]
    plan, out = setup_plan(tmp_path, [a,b])
    with pytest.raises(ValueError, match="Cyclic"):
        loop.Supervisor(plan, out)


def test_lost_worker_retains_live_child_and_recovers_explicit_completion(tmp_path):
    command = task(tmp_path)
    plan, out = setup_plan(tmp_path, [command])
    supervisor = loop.Supervisor(plan, out)
    try:
        dead = {"pid":999999999,"start_ticks":0}
        loop.atomic_json(out / "job.worker.json", {"worker":dead, "child":loop.process_identity(__import__('os').getpid())})
        Path(command["complete"][0]["path"]).write_text(json.dumps({"state":"finished"}))
        assert supervisor.inspect(command) == "running"
        loop.atomic_json(out / "job.worker.json", {"worker":dead,"child":dead})
        assert supervisor.inspect(command) == "finished"
        receipt = json.loads((out / "job.receipt.json").read_text())
        assert receipt["returncode"] is None and receipt["reason"].startswith("Recovered")
    finally:
        supervisor.close()


def test_adopted_parent_loss_retains_external_child_lane(tmp_path, monkeypatch):
    adopted = {"id":"external","lane":"gpu","adopt":{"pid":999999999,"start_ticks":0},
               "leases":["registered_job"],"complete":[{"path":str(tmp_path / "missing")}]}
    plan, out = setup_plan(tmp_path,[adopted,task(tmp_path)])
    monkeypatch.setattr(loop,"matching_leases",lambda tokens:[{"pid":12,"start_ticks":100,"state":"S"}])
    supervisor = loop.Supervisor(plan,out)
    try:
        state = supervisor.tick()["tasks"]
        assert state["external"]["state"] == "running" and state["job"]["state"] == "pending"
        assert not (out / "job.worker.json").exists()
    finally:
        supervisor.close()


def test_lease_matches_output_owner_not_checkpoint_input():
    spec = {"tokens":["local_joint_fit.py"],"option":"--out","prefix":"/old_campaign/"}
    assert loop.argv_matches(["python","local_joint_fit.py","--out","/old_campaign/case"],spec)
    assert not loop.argv_matches(["python","local_joint_fit.py","--init","/old_campaign/case/best.npz",
                                  "--out","/new_campaign/case"],spec)
