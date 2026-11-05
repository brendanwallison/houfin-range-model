"""Evaluate generating and fitted points under identical synthetic likelihoods.

Run from each arm's frozen code with its unchanged config and synthetic BBS file.
This is a fixed-point audit, not optimization or posterior inference.
"""
import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density

from src.config_utils import load_age_model_config
from src.model import hmc_common as hc
from src.model.age_priors import build_model_2d
from src.model.checkpoints import load_map_params, auto_delta_params_to_latents
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import require_gpu


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--truth",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    jax.config.update("jax_enable_x64",True)
    device=require_gpu("fixed-point NestWatch truth audit")
    config=load_age_model_config()
    data=load_data(config["input_dir"],target_device=device,precision="float64",verbose=False)
    arrays,static=hc.split_data(data)
    model=hc.hide_deterministics(hc.array_model(build_model_2d,static))
    kwargs={"arrays":arrays}
    truth_info=json.loads(args.truth.read_text())
    base,_=load_map_params(truth_info["run_dir"])
    fitted,checkpoint=load_map_params(str(args.run))
    base=auto_delta_params_to_latents(base)
    base.update(truth_info["set"])
    base.update(truth_info["nuisance_truth"])
    fitted=auto_delta_params_to_latents(fitted)
    # Only fitted sample sites belong in the substituted point. Missing truth
    # sites are errors, not randomly filled-in nuisance parameters.
    if set(fitted)-set(base): raise ValueError(f"missing generating sites: {set(fitted)-set(base)}")
    truth={k:jnp.asarray(base[k]) for k in fitted}
    fitted={k:jnp.asarray(v) for k,v in fitted.items()}

    @jax.jit
    def evaluate(point):
        joint,trace=log_density(model,(),kwargs,point)
        pieces={}
        for name,site in trace.items():
            if site["type"]=="sample":
                scale=site.get("scale")
                pieces[name]=jnp.sum(site["fn"].log_prob(site["value"])*(1 if scale is None else scale))
        return joint,pieces

    result={"status":"complete","run":str(args.run),"truth_record":str(args.truth),
            "synthetic_bbs":os.environ.get("HOUFIN_SYNTHETIC_OBS"),"points":{},
            "interpretation":"Same-data fixed-point comparison. Generating parameters need not maximize a finite-sample posterior. A better generating-point objective would flag optimization; a better fitted objective alone does not prove identification."}
    for name,point in (("generating",truth),("fitted",fitted)):
        joint,pieces=evaluate(point)
        pieces={k:float(v) for k,v in pieces.items()}
        entry={"constrained_log_joint":float(joint),"site_log_density":pieces,
               "sum_check_error":float(sum(pieces.values())-joint)}
        if abs(entry["sum_check_error"])>1e-4:
            raise ValueError("site decomposition does not reproduce joint density")
        # Reproduce the refinement's own transformed objective as well; a zero
        # HalfNormal boundary cannot be represented in unconstrained coordinates.
        interior=dict(point)
        if float(interior.get("journey_distance_hazard",1))==0:
            interior["journey_distance_hazard"]=jnp.asarray(1e-8)
            entry["transformed_objective_boundary_h_approximation"]=1e-8
        z,potential,_=hc.potential_with_args(model,kwargs,interior)
        entry["refinement_potential"]=float(jax.jit(lambda v:potential(v,kwargs))(z))
        result["points"][name]=entry
        print(name,entry["constrained_log_joint"],entry["refinement_potential"],flush=True)
    report_path=args.run/"refine_report.json"
    if report_path.exists():
        report=json.loads(report_path.read_text())
        result["stored_refinement_U_end"]=report["U_end"]
        result["fitted_potential_minus_stored"]=result["points"]["fitted"]["refinement_potential"]-report["U_end"]
        result["stored_objective_agreement"]=abs(result["fitted_potential_minus_stored"])<1.
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    temporary.replace(args.out)


if __name__=="__main__": main()
