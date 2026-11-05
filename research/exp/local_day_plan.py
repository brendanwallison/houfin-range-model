"""Instantiate the registered day graph against live local process identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from local_loop_supervisor import atomic_json, process_identity


def find_process(required):
    matches = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path / "cmdline").read_bytes().decode().split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if all(any(part in arg for arg in argv) for part in required):
            identity = process_identity(int(path.name))
            if identity and identity["state"] != "Z":
                matches.append(identity)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous live ownership for {required}")
    return matches[0] if matches else {"pid":0, "start_ticks":0}


def source_guards(root):
    paths = list((root / "src").rglob("*.py")) + list((root / "config").rglob("*.json"))
    paths += list((root / "research/exp").glob("local_*.py"))
    paths += list((root / "research/exp").glob("structural_*.py"))
    paths += list((root / "research/exp/invasion_sandbox").glob("*.py"))
    paths += [root / "research/bin/local_takeover_run.sh"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def fit_checks(directory):
    return [{"path":str(directory / "result.json"), "equals":{"state":"finished"},
             "finite":["objective", "data_nll.obs", "data_nll.nest_loglik", "gradient_l2"]},
            {"path":str(directory / "best_params.npz")}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    root, artifacts = args.root.resolve(), args.artifacts.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    guards = source_guards(root)
    gpu_campaign = artifacts / "campaign_03"
    gpu_cases = [row[0] for row in json.loads((gpu_campaign / "campaign_manifest.json").read_text())["cases"]]
    cpu = artifacts / "cpu_compact_start2"
    tasks = [
        {"id":"adopt_gpu03", "lane":"gpu", "adopt":find_process(["local_structural_campaign.py", str(gpu_campaign)]),
         "leases":{"tokens":["local_joint_fit.py"],"option":"--out","prefix":str(gpu_campaign)+"/"},
         "complete":[{"path":str(gpu_campaign / "events.jsonl"),"jsonl_state":"campaign_finished"}]
                     + [check for case in gpu_cases for check in fit_checks(gpu_campaign / case)]},
        {"id":"adopt_cpu2", "lane":"cpu_fit", "adopt":find_process(["local_joint_fit.py", str(cpu)]),
         "complete":fit_checks(cpu)},
        {"id":"adopt_collector03", "lane":"readout", "adopt":find_process(["structural_campaign_collect.py", str(gpu_campaign)]),
         "leases":{"tokens":["structural_fit_readout.py"],"option":"--out","prefix":str(artifacts / "campaign_03_readouts")+"/"},
         "complete":[{"path":str(artifacts / "campaign_03_readouts" / case / "readout.json"),
                       "finite":["diagnostic_population_max_error", "mortality_numpy_reconstruction_max_error"]} for case in gpu_cases]},
    ]
    base_env = {"TAKEOVER_CODE_ROOT":str(root), "TAKEOVER_ARTIFACT_ROOT":str(artifacts)}
    runner = str(root / "research/bin/local_takeover_run.sh")
    def add_readout(name, params, overlay, dependency):
        destination = args.out / "readouts" / name
        tasks.append({"id":"readout_"+name,"lane":"readout","needs":[dependency],"cwd":str(root),"guards":guards,
                      "env":{**base_env,"SBX_DEVICE":"cpu","SBX_CORES":"8-15","SBX_THREADS":"8",
                             "AGE_MODEL_CONFIG":str(root / "config/overlays" / overlay)},
                      "argv":["bash",runner,"research/exp/structural_fit_readout.py","--params",str(params),"--out",str(destination)],
                      "complete":[{"path":str(destination / "readout.json"),
                                   "finite":["diagnostic_population_max_error","mortality_numpy_reconstruction_max_error"]}]})
    warm = artifacts / "campaign_02/f0_d1_refine/best_params.npz"
    settings = [("duration2","takeover_f0_compact2.json",warm,1500),
                ("duration6","takeover_f0_compact6.json",warm,1500),
                ("midpoint_refine","takeover_f0_compact3_midpoint.json",artifacts / "campaign_02/f0_compact_midpoint/best_params.npz",1000),
                ("sigmoid_refine","takeover_f0_sigmoid_matched.json",artifacts / "campaign_02/f0_sigmoid_matched/best_params.npz",1000)]
    for name, overlay, initial, iterations in settings:
        destination = args.out / "fits" / name
        tasks.append({"id":"fit_"+name,"lane":"gpu","needs":[],"cwd":str(root),"guards":guards,
                      "env":{**base_env,"SBX_DEVICE":"gpu","SBX_CORES":"0-7","SBX_THREADS":"8",
                             "AGE_MODEL_CONFIG":str(root / "config/overlays" / overlay)},
                      "argv":["bash",runner,"research/exp/local_joint_fit.py","--init",str(initial),"--out",str(destination),
                              "--code-commit",args.code_commit,"--maxiter",str(iterations)], "complete":fit_checks(destination)})
        add_readout(name, destination / "best_params.npz", overlay, "fit_"+name)
    best_out = args.out / "fits/compact_best_refine"
    candidates = [artifacts / "campaign_02/f0_d1_refine", artifacts / "campaign_03/f0_d1_start3", cpu]
    tasks.append({"id":"fit_compact_best_refine","lane":"gpu","needs":["adopt_gpu03","adopt_cpu2"],
                  "cwd":str(root),"guards":guards,
                  "env":{**base_env,"SBX_DEVICE":"gpu","SBX_CORES":"0-7","SBX_THREADS":"8",
                         "AGE_MODEL_CONFIG":str(root / "config/overlays/takeover_f0_compact3.json")},
                  "argv":[sys.executable,str(root / "research/exp/local_refine_selected.py"),
                          *[arg for candidate in candidates for arg in ("--candidate",str(candidate))],
                          "--out",str(best_out),"--selection",str(args.out / "selected_start.json"),
                          "--code-commit",args.code_commit,"--maxiter","1500"],"complete":fit_checks(best_out)})
    add_readout("compact_best_refine",best_out / "best_params.npz","takeover_f0_compact3.json","fit_compact_best_refine")
    add_readout("cpu_start2",cpu / "best_params.npz","takeover_f0_compact3.json","adopt_cpu2")
    plan = {"registration":"research/experiments/TAKEOVER_DAY_LOOP_20261105.md", "code_commit":args.code_commit,
            "lock_path":str(artifacts / "autonomous_supervisor.lock"),
            "stop_path":"/mnt/c/Dev/houfin-range-model/research/STOP", "gpu_budget_seconds":6*3600,
            "storage_root":str(artifacts),"max_storage_bytes":150*1024**3,"min_free_bytes":5*1024**3,"tasks":tasks}
    atomic_json(args.out / "plan.json",plan)
    print(json.dumps({"plan":str(args.out / "plan.json"),"tasks":len(tasks),
                      "adopted":[{"id":t["id"],"identity":t["adopt"]} for t in tasks if t.get("adopt")}))


if __name__ == "__main__":
    main()
