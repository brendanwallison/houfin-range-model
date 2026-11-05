"""Monitor one adopted Linux process by PID plus start time; never restart it.

File age is diagnostic only. Process disappearance, zombie state or PID reuse is
required before classifying work as stopped. Writes a bounded heartbeat record.
"""
import argparse
import json
from pathlib import Path
import time


def identity(pid, proc_root=Path("/proc")):
    try:
        text = (proc_root / str(pid) / "stat").read_text()
        # comm may contain spaces or parentheses; fields after the last ')' start
        # with field 3 (state). starttime is field 22, user/system CPU 14/15.
        fields = text[text.rfind(")") + 2:].split()
        return {"start_ticks": int(fields[19]), "state": fields[0],
                "cpu_ticks": int(fields[11]) + int(fields[12])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def classify(first, current):
    if current is None:
        return "process_missing"
    if first["start_ticks"] != current["start_ticks"]:
        return "pid_reused"
    if current["state"] == "Z":
        return "zombie"
    return "live"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=20.)
    parser.add_argument("--max-seconds", type=float, default=3600.)
    args = parser.parse_args()
    initial = identity(args.pid)
    if initial is None or initial["state"] == "Z":
        raise SystemExit("Cannot adopt a process that is not live")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous = None
    while True:
        current = identity(args.pid)
        state = classify(initial, current)
        record = {"pid": args.pid, "state": state, "initial_identity": initial,
                  "current_identity": current, "checked_unix": time.time(),
                  "log_bytes": args.log.stat().st_size if args.log.exists() else None,
                  "summary_present": args.summary.is_file()}
        temp = args.out / "heartbeat.tmp"
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(args.out / "heartbeat.json")
        change = (state, record["log_bytes"], record["summary_present"])
        if change != previous:
            print(json.dumps(record), flush=True)
            previous = change
        if state != "live":
            return
        if time.monotonic() - started >= args.max_seconds:
            print(json.dumps({"monitor": "time_limit", "task_state": "live", "pid": args.pid}), flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
