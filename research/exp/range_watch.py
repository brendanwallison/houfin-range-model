"""Persistent, read-only experiment/node monitoring plus local snapshot mirroring.

Run remote and local roles independently under nohup. STOP in each state directory
stops that watcher. Existing Slurm workers retain ownership of queue dispatch.
No jobs are submitted, cancelled, moved or retried by this monitor.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import pickle
from pathlib import Path
import re
import subprocess
import time


def command(args, timeout=20):
    result=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}: {result.stderr[-600:]}")
    return result.stdout


def atomic_json(path, data):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n")
    os.replace(temporary,path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return None


def log_tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0,2)
            size=stream.tell(); stream.seek(max(0,size-2048))
            lines=stream.read().decode(errors="replace").replace("\r","\n").splitlines()
        return {"mtime":path.stat().st_mtime,"bytes":size,"tail":lines[-3:]}
    except OSError:
        return None


def slurm_jobs(text):
    jobs=[]
    for line in text.splitlines():
        parts=line.strip().split("|",4)
        if len(parts)==5:
            jobs.append(dict(zip(("id","name","state","nodes","remaining"),parts)))
    return jobs


def node_snapshot(node):
    if not re.fullmatch(r"[a-zA-Z0-9.-]+",node):
        return {"error":"unexpanded or invalid node name"}
    try:
        gpu=command(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=6",node,
                     "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits; "
                     "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits"],15)
        return {"gpu_and_processes":gpu.strip().splitlines()}
    except (RuntimeError,subprocess.TimeoutExpired) as exc:
        return {"error":str(exc)}


class HostUnpickler(pickle.Unpickler):
    """Read trusted internal JAX checkpoints into NumPy without backend startup."""
    def find_class(self,module,name):
        if module=="jax._src.array" and name=="_reconstruct_array":
            def reconstruct(fun,args,state,aval):
                value=fun(*args); value.__setstate__(state); return value
            return reconstruct
        if module.startswith("jax"):
            raise ValueError(f"unsupported checkpoint type {module}.{name}")
        return super().find_class(module,name)


def checkpoint_scalars(path):
    import numpy as np
    with path.open("rb") as stream:
        checkpoint=HostUnpickler(stream).load()
    return {k:np.asarray(v).tolist() for k,v in checkpoint["params"].items() if np.asarray(v).size<=8}


def remote_snapshot(root, previous):
    jobs=slurm_jobs(command(["squeue","-h","-u",os.environ["USER"],"-o","%i|%j|%T|%N|%L"]))
    workers=[j for j in jobs if j["name"].startswith("grid54_")]
    nodes=sorted({j["nodes"] for j in workers if j["state"]=="RUNNING" and j["nodes"]})
    with ThreadPoolExecutor(max_workers=4) as pool:
        telemetry=dict(zip(nodes,pool.map(node_snapshot,nodes)))
    queue={}
    active=set()
    for status in ("pending","running","done","failed","local_running"):
        queue[status]=sorted(p.name for p in (root/"queue"/status).glob("*.sh"))
        if status in ("running","local_running"):
            for name in queue[status]:
                path=root/"queue"/status/name
                try:
                    active.update(re.findall(r"/runs/([^/]+)/grid54_fit\.sh",path.read_text()))
                except OSError:
                    pass
    experiments={}
    cached=(previous or {}).get("experiments",{})
    for run in sorted((root/"runs").glob("step*")):
        if not re.match(r"step(?:3_native|25_|27_|29_|30_|31_|32_|33_|34_)",run.name):
            continue
        out=run/"out"
        # grid54 root is .../houfin/research/grid54; processed is under houfin.
        report=root.parents[1]/"processed/model_results"/f"refined_float64__age_map_float32_grid54_{run.name}_quick90/refine_report.json"
        entry={"active_bundle":run.name in active,"fit_done":(out/"FIT_DONE").exists()}
        entry["logs"]={name:value for name in ("map.log","refine.log","viz_ref.log")
                       if (value:=log_tail(out/name)) is not None}
        if report.exists():
            stamp=report.stat().st_mtime
            old=cached.get(run.name,{})
            entry["report_mtime"]=stamp
            entry["report"]=old.get("report") if old.get("report_mtime")==stamp else read_json(report)
            if (entry["report"] or {}).get("valid"):
                try:
                    entry["scalars"]=old["scalars"] if old.get("report_mtime")==stamp and "scalars" in old else checkpoint_scalars(report.parent/"map_checkpoint.pkl")
                except Exception as exc:
                    entry["collection_error"]=str(exc)
        metrics=read_json(out/"viz_ref/metrics.json")
        if metrics:
            entry["metrics"]={k:metrics.get(k) for k in ("early_mean_lambda","modern_mean_lambda","early_suitable_fraction","modern_suitable_fraction","holdout")}
        for key,filename in (("pathology","pathology.json"),("front_speed","front_speed.json")):
            value=read_json(out/filename)
            if value is not None: entry[key]=value
        entry["stale_logs"]=bool(entry["active_bundle"] and entry["logs"] and not entry["fit_done"]
            and time.time()-max(v["mtime"] for v in entry["logs"].values())>1800)
        experiments[run.name]=entry
    audits={p.name:value for p in (root/"overnight").glob("*.json")
            if (value:=read_json(p)) is not None}
    return {"jobs":workers,"nodes":telemetry,"queue":queue,"experiments":experiments,"audits":audits,
            "second_wave":read_json(root/'monitor/second_wave_ledger.json')}


def local_snapshot(args):
    ssh=["ssh","-S",str(Path(args.socket).expanduser()),"-o","BatchMode=yes","-o","ConnectTimeout=10",args.host]
    # Remote state path is a fixed deployment argument, never interpreted as executable code.
    if not re.fullmatch(r"/[a-zA-Z0-9_./-]+",args.remote_state):
        raise ValueError("invalid remote state path")
    remote=json.loads(command(ssh+["cat",args.remote_state+"/latest.json"],25))
    result={"remote":remote}
    try:
        result["local_gpu"]=command(["nvidia-smi","--query-gpu=name,memory.total,memory.used,utilization.gpu","--format=csv,noheader,nounits"],10).strip()
    except (RuntimeError,subprocess.TimeoutExpired) as exc:
        result["local_gpu_error"]=str(exc)
    try:
        data=json.loads(command([str(Path.home()/".local/bin/pueue"),"status","--json"],10))
        result["local_tasks"]={k:{a:v.get(a) for a in ("label","status","command")}
                               for k,v in data["tasks"].items() if "Done" not in str(v["status"])}
    except (RuntimeError,ValueError,subprocess.TimeoutExpired) as exc:
        result["local_tasks_error"]=str(exc)
    result["remote_heartbeat_age_seconds"]=time.time()-remote["epoch"]
    result["remote_watch_stale"]=result["remote_heartbeat_age_seconds"]>180
    return result


def changes(previous, current):
    old=previous.get("remote",previous) if previous else {}
    new=current.get("remote",current)
    events=[]
    if current.get("remote_watch_stale")!=(previous or {}).get("remote_watch_stale"):
        if "remote_watch_stale" in current:
            events.append({"kind":"monitor_health","remote_watch_stale":current["remote_watch_stale"]})
    if old.get("jobs")!=new.get("jobs"):
        # Remaining time changes every poll; compare allocation identity/state only.
        key=lambda jobs:[{k:j[k] for k in ("id","name","state","nodes")} for j in jobs]
        if key(old.get("jobs",[]))!=key(new.get("jobs",[])):
            events.append({"kind":"allocations","jobs":new.get("jobs",[])})
    for name,run in new.get("experiments",{}).items():
        before=old.get("experiments",{}).get(name,{})
        for key in ("report_mtime","fit_done","stale_logs"):
            if run.get(key)!=before.get(key):
                events.append({"kind":key,"run":name,"value":run.get(key),
                               "valid":(run.get("report") or {}).get("valid")})
        if run.get("pathology")!=before.get("pathology"):
            events.append({"kind":"pathology_readout","run":name,"status":(run.get("pathology") or {}).get("status")})
    for state in ("done","failed","local_running"):
        added=sorted(set(new.get("queue",{}).get(state,[]))-set(old.get("queue",{}).get(state,[])))
        if added:
            events.append({"kind":"queue_"+state,"tasks":added})
    for node,info in new.get("nodes",{}).items():
        before=old.get("nodes",{}).get(node)
        if before is None or before.get("error")!=info.get("error"):
            events.append({"kind":"node_health","node":node,"error":info.get("error")})
    for name,audit in new.get("audits",{}).items():
        if old.get("audits",{}).get(name)!=audit:
            events.append({"kind":"audit_collected","name":name,"status":audit.get("status")})
    return events


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role",choices=["remote","local"])
    parser.add_argument("--root",type=Path)
    parser.add_argument("--state",type=Path,required=True)
    parser.add_argument("--interval",type=int,default=60)
    parser.add_argument("--host",default="bwa386@ls6.tacc.utexas.edu")
    parser.add_argument("--socket",default="~/.ssh/cm-tacc")
    parser.add_argument("--remote-state",default="/work/07980/bwa386/ls6/houfin/research/grid54/monitor")
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    args.state.mkdir(parents=True,exist_ok=True)
    with (args.state/"watch.lock").open("w") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("watcher already running for this state directory")
        (args.state/"watch.pid").write_text(str(os.getpid())+"\n")
        previous=read_json(args.state/"latest.json")
        while not (args.state/"STOP").exists():
            try:
                current=remote_snapshot(args.root,previous) if args.role=="remote" else local_snapshot(args)
                current.update(epoch=time.time(),checked_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid())
                events=changes(previous,current)
                atomic_json(args.state/"latest.json",current)
                if events:
                    with (args.state/"events.jsonl").open("a") as stream:
                        for event in events:
                            stream.write(json.dumps({"epoch":current["epoch"],**event})+"\n")
                    atomic_json(args.state/"action_required.json",{"checked_utc":current["checked_utc"],"events":events})
                previous=current
                (args.state/"heartbeat").write_text(current["checked_utc"]+"\n")
            except Exception as exc:
                with (args.state/"errors.jsonl").open("a") as stream:
                    stream.write(json.dumps({"epoch":time.time(),"error":str(exc)})+"\n")
                print(str(exc),flush=True)
            if args.once:
                break
            time.sleep(max(10,args.interval))


if __name__=="__main__":
    main()
