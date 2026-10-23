"""Tick bookkeeping for the research loop: LOCK/STOP, the per-tick journal, and the generated STATE.md.

    python research/tick.py begin        # STOP? stale or foreign LOCK? half-done tick? then open a journal
    python research/tick.py note "..."   # append a line to the open journal
    python research/tick.py end          # render STATE.md, close the journal, release LOCK
    python research/tick.py render       # STATE.md only
    python research/tick.py brief        # one status line + NOOP/WORK verdict (the cheap no-op path)

STATE.md is GENERATED from registry.jsonl, hypotheses.yaml, FINDINGS.md, PHASE, EVAL_VERSION and the
queue, plus intent.md verbatim (the only hand-written part, <=20 lines). If the files and a compacted
conversation disagree, the files win.
"""
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import paths  # noqa: E402
import jobs  # noqa: E402

R = paths.RESEARCH
LOCK, STOP, JOURNAL = R / "LOCK", R / "STOP", R / "journal"
STALE_LOCK_H = 3.0
MAX_LINES = 80


def _read(p, default=""):
    try:
        return Path(p).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default


def open_journal():
    if not JOURNAL.exists():
        return None
    for p in sorted(JOURNAL.glob("*.md"), reverse=True):
        if _read(p).splitlines()[:1] == ["status: open"]:
            return p
    return None


def verdict():
    """NOOP when nothing finished, nothing failed, and nothing is due: the tick should only reschedule."""
    rows = jobs.status(brief=True)
    uncollected = sum(r["state"] in ("finished", "FAILED") for r in rows)
    due = [r for r in rows if r["eta"] and r["eta"] <= jobs.now()[:16]]
    intent_due = "DUE:" in _read(R / "intent.md")
    v = "WORK" if (uncollected or due or intent_due or not rows) else "NOOP"
    print(f"verdict {v}  (uncollected {uncollected}, overdue {len(due)}, intent-due {intent_due}, "
          f"jobs tracked {len(rows)})")
    return v


def begin():
    if STOP.exists():
        print(f"STOP present ({_read(STOP)[:200]}); not starting a tick.")
        sys.exit(3)
    if LOCK.exists():
        lk = json.loads(_read(LOCK) or "{}")
        age_h = (_dt.datetime.now().astimezone()
                 - _dt.datetime.fromisoformat(lk.get("at", jobs.now()))).total_seconds() / 3600
        if age_h < STALE_LOCK_H:
            print(f"LOCKED by tick {lk.get('tick')} since {lk.get('at')} ({age_h:.1f} h); "
                  "another session is ticking. Exit.")
            sys.exit(4)
        print(f"stale LOCK ({age_h:.1f} h) from tick {lk.get('tick')}: taking it over.")
    half = open_journal()
    if half:
        print(f"HALF-DONE TICK: {half.name} was never closed. Finish or roll back its actions first:")
        print("\n".join(_read(half).splitlines()[-25:]))
    JOURNAL.mkdir(exist_ok=True)
    tick = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (JOURNAL / f"{tick}.md").write_text(f"status: open\n# tick {tick}\n", encoding="utf-8", newline="\n")
    LOCK.write_text(json.dumps({"tick": tick, "at": jobs.now()}), encoding="utf-8")
    print(f"tick {tick} begun")
    verdict()


def note(text):
    j = open_journal()
    if not j:
        sys.exit("no open journal; run `tick.py begin` first")
    with open(j, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"- {jobs.now()[11:19]} {text}\n")


def hypotheses_block():
    try:
        import yaml
        hs = yaml.safe_load(_read(R / "hypotheses.yaml")) or []
    except Exception as exc:  # noqa: BLE001 -- the state page must render even if the yaml is broken
        return [f"(hypotheses.yaml unreadable: {exc})"]
    counts = {}
    for h in hs:
        counts[h.get("status", "?")] = counts.get(h.get("status", "?"), 0) + 1
    out = [" | ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    testing = [h for h in hs if h.get("status") == "testing"]
    for h in testing[:8]:
        out.append(f"- testing {h['id']}: {str(h.get('statement', ''))[:110]}")
    top = [h["id"] for h in hs if h.get("status") == "open" and h.get("priority") == 1]
    if top:
        out.append("- open, priority 1: " + ", ".join(top))
    return out


def recent_events(n=10):
    ev = [e for e in jobs.read_registry() if e.get("event") != "queued"][-n:]
    return [f"- {e['ts'][5:16]} {e['event']:8s} {e.get('job', e.get('exp', ''))} "
            f"{(e.get('headline') or '')[:60]}" for e in ev]


def findings_block(n=3):
    heads = [l[3:].strip() for l in _read(R / "FINDINGS.md").splitlines() if l.startswith("## ")]
    return [f"{len(heads)} verified findings"] + [f"- {h[:110]}" for h in heads[-n:]]


def render():
    branch = jobs.git("rev-parse", "--abbrev-ref", "HEAD", check=False)
    head = jobs.git("rev-parse", "--short", "HEAD", check=False)
    lines = [f"# STATE -- generated {jobs.now()} by research/tick.py; do not edit (edit intent.md)",
             f"phase {_read(R / 'PHASE', '?')} | eval {jobs.eval_version()} | {branch} @ {head}",
             "", "## Jobs"]
    rows = jobs.status(brief=True)
    lines += [f"- {r['state']:9s} {r['group'] or '-':7s} {r['eta'] or '':17s} {r['job']}" for r in rows[:10]]
    lines += ["", "## Recent events"] + recent_events()
    lines += ["", "## Hypotheses"] + hypotheses_block()
    lines += ["", "## Findings"] + findings_block()
    lines += ["", "## Intent (research/intent.md)"] + _read(R / "intent.md").splitlines()[:20]
    lines = lines[:MAX_LINES]
    (R / "STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"STATE.md rendered ({len(lines)} lines)")


def end():
    render()
    j = open_journal()
    if j:
        txt = _read(j).splitlines()
        txt[0] = "status: closed"
        j.write_text("\n".join(txt) + f"\n- {jobs.now()[11:19]} tick closed\n", encoding="utf-8",
                     newline="\n")
    LOCK.unlink(missing_ok=True)
    print("tick ended; LOCK released")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "brief"
    if cmd == "begin":
        begin()
    elif cmd == "note":
        note(" ".join(sys.argv[2:]))
    elif cmd == "end":
        end()
    elif cmd == "render":
        render()
    elif cmd == "brief":
        verdict()
    else:
        sys.exit(__doc__)
