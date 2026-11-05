"""Instantiate the preregistered conditional profile grid with explicit lanes."""
import argparse
import json
from pathlib import Path
import sys

from local_day_plan import source_guards
from local_loop_supervisor import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--code-commit",required=True)
    args = parser.parse_args()
    root, artifacts = args.root.resolve(),args.artifacts.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    guards = source_guards(root)
    common = artifacts / "campaign_02/f0_d0_refine/best_params.npz"
    compact = artifacts / "autonomous_day_20261105/fits/compact_best_refine/best_params.npz"
    original = artifacts / "campaign_02/f0_d0_refine/best_params.npz"
    env = {"TAKEOVER_CODE_ROOT":str(root),"SBX_DEVICE":"gpu","SBX_CORES":"0-7","SBX_THREADS":"8"}
    tasks = []
    def readout(name,params,overlay,parent):
        destination = args.out / "readouts" / name
        tasks.append({"id":"readout_"+name,"lane":"readout","needs":[parent],"cwd":str(root),"guards":guards,
                      "env":{**env,"SBX_DEVICE":"cpu","SBX_CORES":"8-15","AGE_MODEL_CONFIG":str(root / "config/overlays" / overlay)},
                      "argv":["bash",str(root / "research/bin/local_takeover_run.sh"),"research/exp/structural_fit_readout.py",
                              "--params",str(params),"--out",str(destination)],
                      "complete":[{"path":str(destination / "readout.json"),"finite":["diagnostic_population_max_error","mortality_numpy_reconstruction_max_error"]}]})
    owner = json.loads((artifacts / "conditional_profiles/free_compact_owner/worker.json").read_text())
    free_compact = artifacts / "conditional_profiles/free_compact"
    tasks.append({"id":"free_compact","lane":"cpu_fit","adopt":owner["worker"],
                  "leases":{"tokens":["local_joint_fit.py"],"option":"--out","prefix":str(free_compact)+"/"},
                  "complete":[{"path":str(free_compact / "profile.json"),"equals":{"state":"finished"}},
                              {"path":str(free_compact / "final_params.npz")}]})
    readout("free_compact",free_compact / "final_params.npz","takeover_f0_compact3.json","free_compact")
    def profile(name,overlay,initial,target,observer=common):
        destination = args.out / "points" / name
        command = [sys.executable,str(root / "research/exp/local_profile_point.py"),"--out",str(destination),
                   "--init",str(initial),"--common-nuisance",str(observer),"--code-commit",args.code_commit]
        if target is not None:
            command += ["--target",str(target)]
        tasks.append({"id":name,"lane":"gpu","needs":[],"cwd":str(root),"guards":guards,
                      "env":{**env,"AGE_MODEL_CONFIG":str(root / "config/overlays" / overlay)},"argv":command,
                      "complete":[{"path":str(destination / "profile.json"),"equals":{"state":"finished"}},
                                  {"path":str(destination / "final_params.npz")}]})
        readout(name,destination / "final_params.npz",overlay,name)
    profile("free_original","grid54_step43_factor_p2_veit_s41.json",original,None)
    for family,overlay,initial in (("original","grid54_step43_factor_p2_veit_s41.json",original),
                                    ("compact","takeover_f0_compact3.json",compact)):
        for target in (.30,.15,.05):
            profile(f"{family}_rms{int(target*100):02d}",overlay,initial,target)
        profile(f"{family}_rms15_nuisance_sensitivity",overlay,initial,.15,compact)
    atomic_json(args.out / "plan.json",{"registration":"research/experiments/TAKEOVER_LIKELIHOOD_PROFILE_20261105.md",
                "code_commit":args.code_commit,"lock_path":str(artifacts / "autonomous_supervisor.lock"),
                "stop_path":"/mnt/c/Dev/houfin-range-model/research/STOP","gpu_budget_seconds":6*3600,
                "storage_root":str(artifacts),"max_storage_bytes":150*1024**3,"min_free_bytes":5*1024**3,"tasks":tasks})
    print(json.dumps({"plan":str(args.out / "plan.json"),"tasks":len(tasks),"objective":"conditional likelihood, no parameter priors"}))


if __name__ == "__main__":
    main()
