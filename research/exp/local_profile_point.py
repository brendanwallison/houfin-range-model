"""Solve one diagnostic conditional-likelihood constraint by augmented Lagrangian."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def multiplier_step(multiplier,rho,residual,previous_error,tolerance):
    new_multiplier = multiplier+rho*residual
    new_rho = rho*10. if abs(residual) > max(tolerance,previous_error*.5) else rho
    return new_multiplier,new_rho


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--init",type=Path,required=True)
    parser.add_argument("--common-nuisance",type=Path,required=True)
    parser.add_argument("--code-commit",required=True)
    parser.add_argument("--target",type=float)
    parser.add_argument("--max-stages",type=int,default=4)
    parser.add_argument("--maxiter",type=int,default=400)
    parser.add_argument("--tolerance",type=float,default=.005)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root = Path(os.environ["TAKEOVER_CODE_ROOT"])
    rho,multiplier,previous_error = 10000.,0.,float("inf")
    initial, stages, failure = args.init,[],None
    for index in range(args.max_stages if args.target is not None else 1):
        destination = args.out / f"stage_{index}"
        command = ["bash",str(root / "research/bin/local_takeover_run.sh"),"research/exp/local_joint_fit.py",
                   "--objective","conditional_data","--method","full_bfgs","--common-nuisance",str(args.common_nuisance),
                   "--init",str(initial),"--out",str(destination),"--code-commit",args.code_commit,
                   "--maxiter",str(args.maxiter if args.target is not None else 1500)]
        if args.target is not None:
            command += ["--profile-target",str(args.target),"--profile-rho",str(rho),"--profile-multiplier",str(multiplier)]
        print(json.dumps({"state":"profile_stage_running","stage":index,"command":command}),flush=True)
        with (args.out / f"stage_{index}.log").open("x") as log:
            code = subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
        if code or not (destination / "result.json").is_file():
            failure = f"stage {index} failed with exit {code}"
            break
        result = json.loads((destination / "result.json").read_text())
        row = {"directory":str(destination),"rho":rho,"multiplier":multiplier,"fit":result}
        stages.append(row)
        print(json.dumps({"state":"profile_stage_finished",**row}),flush=True)
        if args.target is None:
            break
        residual = result["profile_residual"]
        if abs(residual) <= args.tolerance and result["gradient_max"] < .1:
            break
        multiplier,rho = multiplier_step(multiplier,rho,residual,previous_error,args.tolerance)
        previous_error = abs(residual)
        initial = destination / "best_params.npz"
    feasible = [row for row in stages if args.target is None or abs(row["fit"]["profile_residual"]) <= args.tolerance]
    selected = min(feasible,key=lambda row:sum(row["fit"]["data_nll"].values())) if feasible else None
    if selected:
        temporary = args.out / "final_params.tmp"
        shutil.copyfile(Path(selected["directory"]) / "best_params.npz",temporary)
        temporary.replace(args.out / "final_params.npz")
    result = {"state":"failed_stage" if failure else "finished" if selected else "constraint_not_reached",
              "failure":failure,"target":args.target,"tolerance":args.tolerance,"stages":stages,"selected":selected,
              "scope":"Conditional data likelihood with common observer effects fixed; local optimization upper bounds, not fully marginalized likelihood or production priors",
              "params_sha256":hashlib.sha256((args.out / "final_params.npz").read_bytes()).hexdigest() if selected else None}
    (args.out / "profile.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"state":result["state"],"target":args.target,"selected":selected}),flush=True)
    if result["state"] != "finished":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
