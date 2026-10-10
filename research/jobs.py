"""The research job runner: immutable code snapshots, a pueue queue, an append-only registry. Run in WSL.

    python research/jobs.py enqueue research/specs/E001_atlas.json   # snapshot, queue (skips if done)
    python research/jobs.py status [--brief] [--json]
    python research/jobs.py collect                                   # finished jobs -> registry + summary
    python research/jobs.py run <out_dir>                             # internal: what pueue executes

WHY SNAPSHOTS. A job runs from ``git archive <sha>`` of the commit it was pre-registered at, so an edit
made after enqueue cannot leak into it and every result names the exact code that produced it. The
RUNNER is a separate snapshot of HEAD at enqueue time, so a job can run code older than this file
(the reproduction gate runs 86acde3, which has no research/ directory).

WHY WSL GIT FOR ARCHIVES. The Windows checkout is core.autocrlf=true; Windows git archives CRLF shell
scripts. WSL git emits the stored LF blobs. Cleanliness checks pass ``-c core.autocrlf=true`` so WSL git
compares like the checkout does instead of reporting every text file modified.

WHY AN APPEND-ONLY REGISTRY. ``research/registry.jsonl`` holds one JSON event per line (queued,
cached, done, failed, verified, stale, quarantined, ...). A job's state is its last event; no line is
ever rewritten. Only enqueue/collect (driven by the agent, sequentially) write it; ``run`` never does,
because concurrent appends over /mnt/c are not guaranteed atomic.

Spec fields: exp, name, group (cpu|gpu|default), cmd (list; "python" -> venv; "{out}", "{code}",
"{cache}" substituted), optional: sha (default HEAD, tree must be clean), overlay (ESK_DESK_CONFIG,
relative to the code snapshot), env (dict), expected_minutes, summarize (cmd list, same substitutions),
note.
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import paths  # noqa: E402

REGISTRY = paths.RESEARCH / "registry.jsonl"
EVAL_VERSION_FILE = paths.RESEARCH / "EVAL_VERSION"


# ----------------------------- small helpers -----------------------------

def now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def git(*args, check=True, strip=True):
    out = subprocess.run(["git", "-c", "core.autocrlf=true", "-C", paths.REPO, *args],
                         capture_output=True, text=True)
    if check and out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    # porcelain status lines START with a significant space (" M path"): never strip those
    return out.stdout.strip() if strip else out.stdout


def eval_version():
    try:
        return EVAL_VERSION_FILE.read_text().strip()
    except FileNotFoundError:
        return "unversioned"


def atomic_json(path, obj):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def read_registry():
    if not REGISTRY.exists():
        return []
    return [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]


def append_registry(event):
    event = {"ts": now(), **event}
    with open(REGISTRY, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(event, default=str) + "\n")
    return event


def latest_by_job(events=None):
    state = {}
    for e in events if events is not None else read_registry():
        if "job" in e:
            state.setdefault(e["job"], {}).update(e)
    return state


def snapshot(sha):
    """The code at ``sha``, extracted once into CODE_SNAPSHOTS/<sha>. Returns its path."""
    dest = paths.CODE_SNAPSHOTS / sha
    if dest.exists():
        return dest
    paths.CODE_SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=paths.CODE_SNAPSHOTS, prefix=f".{sha[:8]}."))
    arch = subprocess.run(["git", "-C", paths.REPO, "archive", "--format=tar", sha],
                          capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=arch, check=True)
    os.replace(tmp, dest)
    return dest


#: Loop STATE files: written continuously by the runner and the agent, never read by a job's code,
#: so their edits must not count as an uncommitted pre-registration.
STATE_FILES = ("research/registry.jsonl", "research/ledger.csv", "research/STATE.md",
               "research/DIGEST.md", "research/intent.md", "research/FINDINGS.md",
               "research/hypotheses.yaml", "research/PHASE", "research/journal/")


def resolve_sha(spec_sha):
    if spec_sha:
        return git("rev-parse", "--verify", f"{spec_sha}^{{commit}}")
    dirty = "\n".join(line for line in git("status", "--porcelain", "--untracked-files=no",
                                            strip=False).splitlines()
                      if line.strip() and not line[3:].startswith(STATE_FILES))
    if dirty:
        raise SystemExit("tracked files are modified; commit (the pre-registration) before enqueue:\n"
                         + dirty)
    return git("rev-parse", "HEAD")


def cfg_hash(spec, sha):
    key = {k: spec.get(k) for k in ("exp", "name", "cmd", "overlay", "env", "group")}
    key["sha"] = sha
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def substitute(cmd, out_dir, code_dir, runner_dir=None, env=None):
    """Placeholders: {out} job dir, {code} the job's code snapshot, {runner} the research-tools
    snapshot (a job may run code older than research/), {cache} the cache root; then ${VAR} from
    the job environment (e.g. ${HOUFIN_PROCESSED}); a leading "python" is the venv's."""
    sub = {"{out}": str(out_dir), "{code}": str(code_dir), "{cache}": str(paths.CACHE_ROOT),
           "{runner}": str(runner_dir or code_dir), "{python}": paths.VENV_PY,
           "{outroot}": str(paths.OUT_ROOT)}
    res = []
    for i, a in enumerate(cmd):
        a = str(a)
        for k, v in sub.items():
            a = a.replace(k, v)
        if env is not None and "$" in a:
            import string
            a = string.Template(a).safe_substitute(env)
        res.append(paths.VENV_PY if (i == 0 and a == "python") else a)
    return res


def parse_ts(s):
    """pueue's RFC 3339 timestamps carry nanoseconds; fromisoformat takes at most microseconds."""
    import re
    s = re.sub(r"(\.\d{6})\d+", r"\1", s.replace("Z", "+00:00"))
    return _dt.datetime.fromisoformat(s)


def pueue_status():
    out = subprocess.run([paths.PUEUE, "status", "--json"], capture_output=True, text=True)
    if out.returncode != 0:
        return {}
    try:
        return json.loads(out.stdout).get("tasks", {})
    except json.JSONDecodeError:
        return {}


# ----------------------------- commands -----------------------------

def enqueue(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    for k in ("exp", "name", "group", "cmd"):
        if k not in spec:
            raise SystemExit(f"spec {spec_path} lacks {k!r}")
    sha = resolve_sha(spec.get("sha"))
    runner_sha = git("rev-parse", "HEAD")
    code = snapshot(sha)
    runner = snapshot(runner_sha)
    h = cfg_hash(spec, sha)
    job = f"{spec['exp']}/{spec['name']}-{h}"
    out = paths.OUT_ROOT / spec["exp"] / f"{spec['name']}-{h}"
    if (out / "DONE.json").exists():
        append_registry({"event": "cached", "job": job, "out": str(out)})
        print(f"[jobs] {job}: already done -> {out}")
        return
    st = latest_by_job().get(job)
    if st and st.get("event") == "queued" and not (out / "FAILED.json").exists():
        print(f"[jobs] {job}: already queued (pueue id {st.get('pueue_id')})")
        return
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("FAILED.json", "RUNNING.json"):
        if (out / stale).exists():
            (out / stale).rename(out / f"{stale}.prev")
    full = {**spec, "sha": sha, "runner_sha": runner_sha, "cfg_hash": h, "job": job,
            "eval_version": eval_version(), "enqueued_at": now()}
    atomic_json(out / "spec.json", full)
    group = spec["group"]
    # "after_jobs": ["E002/cache-x", ...] -> pueue --after on their latest queued task ids. A
    # dependency that finished long ago (no live task) is simply satisfied; one never queued is an
    # error, because running without its inputs would fail later and less legibly.
    after = []
    st_all = latest_by_job()
    for dep in spec.get("after_jobs", []):
        hits = [s for j, s in st_all.items() if j.startswith(dep + "-") or j == dep]
        if not hits:
            raise SystemExit(f"after_jobs: {dep!r} was never queued")
        live = [s for s in hits if s.get("event") == "queued" and s.get("pueue_id") is not None]
        after += [str(s["pueue_id"]) for s in live]
    res = subprocess.run([paths.PUEUE, "add", "--group", group, "--label", job, "--print-task-id",
                          "--working-directory", str(code)]
                         + (["--after", *after] if after else []) + ["--",
                          paths.VENV_PY, str(runner / "research" / "jobs.py"), "run", str(out)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"pueue add failed: {res.stderr.strip()}")
    pid = int(res.stdout.strip())
    append_registry({"event": "queued", "job": job, "exp": spec["exp"], "name": spec["name"],
                     "sha": sha, "runner_sha": runner_sha, "cfg_hash": h, "group": group,
                     "pueue_id": pid, "out": str(out), "eval_version": full["eval_version"],
                     "expected_minutes": spec.get("expected_minutes"), "note": spec.get("note")})
    print(f"[jobs] queued {job} as pueue task {pid} (code {sha[:10]}, group {group})")


def run(out_dir):
    """Executed by pueue. Never touches the registry (see module docstring)."""
    out = Path(out_dir)
    spec = json.loads((out / "spec.json").read_text())
    code = snapshot(spec["sha"])
    env = paths.job_env(spec["group"], str(code), spec.get("overlay"), spec.get("env"))
    cmd = substitute(spec["cmd"], out, code, snapshot(spec["runner_sha"]), env)
    atomic_json(out / "RUNNING.json", {"started_at": now(), "pid": os.getpid(), "cmd": cmd})
    t0 = _dt.datetime.now()
    with open(out / "log.txt", "w") as log:
        log.write(f"# {now()}  code {spec['sha']}  cmd {' '.join(cmd)}\n")
        log.flush()
        rc = subprocess.run(["/usr/bin/time", "-v", "-o", str(out / "time.txt"), *cmd],
                            cwd=str(code), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = (_dt.datetime.now() - t0).total_seconds()
    rss_kb = None
    try:
        for line in (out / "time.txt").read_text().splitlines():
            if "Maximum resident set size" in line:
                rss_kb = int(line.split(":")[-1])
    except (FileNotFoundError, ValueError):
        pass
    rec = {"exit": rc, "wall_s": round(wall, 1), "max_rss_gb": rss_kb and round(rss_kb / 2**20, 2),
           "finished_at": now()}
    atomic_json(out / ("DONE.json" if rc == 0 else "FAILED.json"), rec)
    (out / "RUNNING.json").unlink(missing_ok=True)
    sys.exit(rc)


def _tail(path, n):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return lines[-n:]


def _pueue_outcome(task_id, tasks):
    """'success' / 'failed:<why>' / None (not finished) for a pueue task id, from ``status --json``."""
    t = tasks.get(str(task_id))
    if not t:
        return None
    raw = t.get("status", {})
    if isinstance(raw, dict) and "Done" in raw:
        res = raw["Done"].get("result")
        return "success" if res == "Success" else f"failed:{res}"
    return None


def collect(max_lines=40):
    st = latest_by_job()
    pending = [j for j, s in st.items() if s.get("event") == "queued"]
    tasks = pueue_status()
    n_done = 0
    for job in sorted(pending):
        s = st[job]
        out = Path(s["out"])
        outcome = _pueue_outcome(s.get("pueue_id"), tasks)
        if (outcome and outcome.startswith("failed") and not (out / "DONE.json").exists()
                and not (out / "FAILED.json").exists()):
            # killed, or died before the runner could write FAILED.json (e.g. a missing snapshot)
            append_registry({"event": "failed", "job": job, "exit": None, "reason": outcome})
            print(f"=== FAILED {job}  (pueue: {outcome}; no runner record)")
            for line in _tail(out / "log.txt", 10):
                print("  " + line)
            n_done += 1
            continue
        if (out / "DONE.json").exists():
            done = json.loads((out / "DONE.json").read_text())
            spec = json.loads((out / "spec.json").read_text())
            summary = []
            if spec.get("summarize"):
                code = snapshot(spec["sha"])
                env = paths.job_env("cpu", str(code), spec.get("overlay"), spec.get("env"))
                r = subprocess.run(substitute(spec["summarize"], out, code, snapshot(spec["runner_sha"]), env), cwd=str(code), env=env,
                                   capture_output=True, text=True)
                summary = (r.stdout + (r.stderr if r.returncode else "")).splitlines()
                (out / "summary.txt").write_text("\n".join(summary) + "\n")
            append_registry({"event": "done", "job": job, "wall_s": done.get("wall_s"),
                             "max_rss_gb": done.get("max_rss_gb"),
                             "summary": str(out / "summary.txt") if summary else None})
            print(f"=== DONE {job}  ({done.get('wall_s')} s, {done.get('max_rss_gb')} GB)")
            for line in summary[:max_lines]:
                print("  " + line)
            if len(summary) > max_lines:
                print(f"  ... {len(summary) - max_lines} more lines in {out / 'summary.txt'}")
            n_done += 1
        elif (out / "FAILED.json").exists():
            f = json.loads((out / "FAILED.json").read_text())
            append_registry({"event": "failed", "job": job, "exit": f.get("exit"),
                             "wall_s": f.get("wall_s")})
            print(f"=== FAILED {job}  exit {f.get('exit')} after {f.get('wall_s')} s; log tail:")
            for line in _tail(out / "log.txt", 25):
                print("  " + line)
            n_done += 1
    if not n_done:
        print("[jobs] nothing new to collect")


def status(brief=False, as_json=False):
    st = latest_by_job()
    tasks = pueue_status()
    by_label = {}
    for t in tasks.values():
        lab = t.get("label")
        if lab:
            by_label[lab] = t
    rows = []
    for job, s in st.items():
        if s.get("event") not in ("queued",):
            continue
        out = Path(s["out"])
        if (out / "DONE.json").exists() or (out / "FAILED.json").exists():
            state, eta = ("finished" if (out / "DONE.json").exists() else "FAILED"), None
        else:
            t = by_label.get(job, {})
            raw = t.get("status", {})
            key = next(iter(raw), "?") if isinstance(raw, dict) else str(raw)
            state = key.lower()
            eta = None
            if state == "running" and s.get("expected_minutes"):
                start = raw[key].get("start") if isinstance(raw.get(key), dict) else None
                if start:
                    try:
                        t0 = parse_ts(start)
                        eta = (t0 + _dt.timedelta(minutes=float(s["expected_minutes"]))).astimezone()
                    except ValueError:
                        eta = None
        rows.append({"job": job, "state": state, "group": s.get("group"),
                     "eta": eta.isoformat(timespec="minutes") if eta else None})
    if as_json:
        print(json.dumps(rows, indent=1))
        return rows
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    etas = sorted(r["eta"] for r in rows if r["eta"])
    stop = (paths.RESEARCH / "STOP").exists()
    lock = (paths.RESEARCH / "LOCK").exists()
    line = (f"jobs {counts or '{}'} | next_eta {etas[0][11:16] if etas else '-'} | "
            f"uncollected {counts.get('finished', 0) + counts.get('FAILED', 0)} | "
            f"STOP {'yes' if stop else 'no'} | LOCK {'yes' if lock else 'no'}")
    print(line)
    if not brief:
        for r in rows:
            print(f"  {r['state']:9s} {r['group'] or '-':7s} {r['eta'] or '':17s} {r['job']}")
    return rows


def cancel(job_prefix, reason):
    """Kill (running) or remove (queued) a registry job's pueue task and record a terminal 'cancelled' event, so it
    neither runs nor stays 'queued' in the registry forever. Never touches its output directory."""
    st = latest_by_job()
    hits = [j for j, s in st.items() if j.startswith(job_prefix) and s.get("event") == "queued"]
    if len(hits) != 1:
        raise SystemExit(f"cancel: {len(hits)} queued jobs match {job_prefix!r}: {hits}")
    job = hits[0]
    tid = str(st[job].get("pueue_id"))
    tasks = pueue_status()
    raw = tasks.get(tid, {}).get("status", {})
    state = raw if isinstance(raw, str) else next(iter(raw), "")
    # pueue refuses to remove a task another task depends on, and a queued task can start between a kill of its
    # predecessor and its own removal -- so act, then VERIFY, and fail loudly rather than record a cancel that did
    # not happen (2026-10-09: two "cancelled" runs went on to use the GPU).
    deps = [i for i, t in tasks.items() if int(tid) in (t.get("dependencies") or [])
            and not (isinstance(t.get("status"), dict) and "Done" in t["status"])]
    if deps:
        raise SystemExit(f"cancel: pueue task {tid} has live dependants {deps}; cancel those jobs first")
    if state == "Running":
        subprocess.run([paths.PUEUE, "kill", tid], check=False)
    elif state in ("Queued", "Stashed", "Paused"):
        subprocess.run([paths.PUEUE, "remove", tid], check=False)
        after = pueue_status().get(tid)
        if after and not (isinstance(after.get("status"), dict) and "Done" in after["status"]):
            a_raw = after.get("status")
            a_state = a_raw if isinstance(a_raw, str) else next(iter(a_raw), "")
            if a_state == "Running":                    # it started meanwhile
                subprocess.run([paths.PUEUE, "kill", tid], check=False)
            else:
                raise SystemExit(f"cancel: pueue task {tid} is still {a_state}; not recording a cancel")
    append_registry({"event": "cancelled", "job": job, "pueue_id": tid, "reason": reason})
    print(f"[jobs] cancelled {job} (pueue task {tid}, was {state or 'unknown'}): {reason}")


def retry(job_prefix, reason):
    """Re-run a FAILED job in its OWN output directory (same spec, same code snapshot), so a job that checkpoints
    (DESK: resume_checkpoint.pt every 10 epochs) resumes instead of restarting. FAILED.json is renamed, never
    deleted; a fresh 'queued' event is appended for the same job id."""
    st = latest_by_job()
    hits = [j for j, s in st.items() if j.startswith(job_prefix) and s.get("event") == "failed"]
    if len(hits) != 1:
        raise SystemExit(f"retry: {len(hits)} failed jobs match {job_prefix!r}: {hits}")
    job = hits[0]
    out = Path(st[job]["out"])
    failed = out / "FAILED.json"
    if failed.exists():
        n = len(list(out.glob("FAILED.prev*.json")))
        failed.rename(out / f"FAILED.prev{n}.json")
    spec = json.loads((out / "spec.json").read_text())
    runner = snapshot(spec["runner_sha"])
    cmd = [paths.VENV_PY, str(runner / "research" / "jobs.py"), "run", str(out)]
    r = subprocess.run([paths.PUEUE, "add", "--group", spec["group"], "--label", job, "--print-task-id",
                        "--working-directory", str(snapshot(spec["sha"])), "--"] + cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"retry: pueue add failed: {r.stderr.strip()}")
    tid = int(r.stdout.strip())
    append_registry({"event": "queued", "job": job, "exp": spec.get("exp"), "name": spec.get("name"),
                     "sha": spec.get("sha"), "runner_sha": spec.get("runner_sha"), "cfg_hash": spec.get("cfg_hash"),
                     "group": spec["group"], "pueue_id": tid, "out": str(out), "retry_of_failure": True,
                     "reason": reason})
    print(f"[jobs] retrying {job} as pueue task {tid} in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enqueue")
    e.add_argument("spec", nargs="+")
    rt = sub.add_parser("retry")
    rt.add_argument("job")
    rt.add_argument("--reason", required=True)
    cn = sub.add_parser("cancel")
    cn.add_argument("job")
    cn.add_argument("--reason", required=True)
    s = sub.add_parser("status")
    s.add_argument("--brief", action="store_true")
    s.add_argument("--json", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--max-lines", type=int, default=40)
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    sn = sub.add_parser("snapshot")
    sn.add_argument("sha")
    a = ap.parse_args()
    if a.cmd == "enqueue":
        for sp in a.spec:
            enqueue(sp)
    elif a.cmd == "retry":
        retry(a.job, a.reason)
    elif a.cmd == "cancel":
        cancel(a.job, a.reason)
    elif a.cmd == "status":
        status(a.brief, a.json)
    elif a.cmd == "collect":
        collect(a.max_lines)
    elif a.cmd == "run":
        run(a.out_dir)
    elif a.cmd == "snapshot":
        print(snapshot(git("rev-parse", "--verify", f"{a.sha}^{{commit}}")))


if __name__ == "__main__":
    main()
