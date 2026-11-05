import json

from research.exp import local_loop_launch as launch


def test_chain_waits_for_supervisor_and_live_children(tmp_path,monkeypatch):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"phase":"queue_drained","supervisor":{"pid":1}}))
    monkeypatch.setattr(launch,"live",lambda identity:bool(identity and identity.get("pid") == 1))
    monkeypatch.setattr(launch,"matching_leases",lambda spec:[])
    assert not launch.predecessor_ready(state)
    state.write_text(json.dumps({"phase":"queue_drained","supervisor":{"pid":2}}))
    assert launch.predecessor_ready(state)
    (tmp_path / "fit.worker.json").write_text(json.dumps({"worker":{"pid":1}}))
    assert not launch.predecessor_ready(state)
