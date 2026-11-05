from research.exp.range_watch import changes, slurm_jobs, node_snapshot, atomic_json, read_json, log_tail


def test_allocation_changes_ignore_countdown_but_detect_new_node():
    a={"jobs":slurm_jobs("1|grid54_worker|RUNNING|v330-003|1:00\n")}
    b={"jobs":slurm_jobs("1|grid54_worker|RUNNING|v330-003|0:59\n")}
    assert changes(a,b)==[]
    b["jobs"]+=slurm_jobs("2|grid54_wsm3|RUNNING|v330-022|24:00\n")
    assert changes(a,b)[0]["kind"]=="allocations"


def test_completion_invalid_refinement_and_stall_events():
    before={"experiments":{"x":{"fit_done":False,"stale_logs":False}}}
    after={"experiments":{"x":{"fit_done":True,"stale_logs":True,"report_mtime":42,"report":{"valid":False}}}}
    events=changes(before,after)
    assert {e["kind"] for e in events}=={"fit_done","stale_logs","report_mtime"}
    assert all(e["valid"] is False for e in events)
    assert changes(after,after)==[]


def test_local_mirror_detects_remote_events():
    assert changes({"remote":{"queue":{"done":[]}}},
                   {"remote":{"queue":{"done":["0400.sh"]}}})==[{"kind":"queue_done","tasks":["0400.sh"]}]


def test_partial_json_and_bounded_log_tail(tmp_path):
    p=tmp_path/"latest.json"
    p.write_text('{"partial":')
    assert read_json(p) is None
    atomic_json(p,{"ok":True})
    assert read_json(p)=={"ok":True}
    log=tmp_path/"map.log"
    log.write_text("x"*10000+"\rstep1\rstep2\rstep3\rstep4")
    tail=log_tail(log)
    assert tail["tail"]==["step2","step3","step4"]
    assert tail["bytes"]>10000


def test_node_query_rejects_shell_patterns():
    assert "error" in node_snapshot("v330-[001-003]")


def test_stale_remote_watcher_is_an_event():
    assert changes({"remote_watch_stale":False},{"remote_watch_stale":True})==[
        {"kind":"monitor_health","remote_watch_stale":True}]


def test_completed_audit_is_collected_once():
    before={"audits":{}}
    after={"audits":{"null.json":{"status":"complete","points":{}}}}
    assert changes(before,after)==[{"kind":"audit_collected","name":"null.json","status":"complete"}]
    assert changes(after,after)==[]
