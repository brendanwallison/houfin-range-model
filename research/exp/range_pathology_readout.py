"""Whole-configuration biological scorecard under each run's frozen model.

No fitting, new priors or population-model modifications. Use float64 on TACC.
"""
import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive
import rasterio
from rasterio.warp import transform as crs_transform

from src.config_utils import load_age_model_config,load_data_config
from src.model.age_priors import build_model_2d
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents
from src.model.data_loading import load_data
from src.model.runtime_diagnostics import require_gpu
from src.vis.age_model_math import local_growth_lambda


def describe(values,weights=None):
    x=np.asarray(values,dtype=float).ravel()
    if weights is None:
        x=x[np.isfinite(x)]
        return {"mean":float(x.mean()),"q10_50_90":np.quantile(x,[.1,.5,.9]).tolist()} if len(x) else None
    w=np.broadcast_to(weights,np.shape(values)).ravel().astype(float)
    keep=np.isfinite(x)&np.isfinite(w)&(w>0)
    x,w=x[keep],w[keep]
    if not len(x): return None
    order=np.argsort(x); x,w=x[order],w[order]
    q=x[np.searchsorted(np.cumsum(w)/w.sum(),[.1,.5,.9],side="left")]
    return {"mean":float(np.dot(x,w)/w.sum()),"q10_50_90":q.tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--precision",choices=['float32','float64'],default='float64')
    parser.add_argument("--save-state",action='store_true')
    parser.add_argument("--device",choices=['gpu','cpu'],default='cpu' if os.environ.get('JAX_PLATFORMS')=='cpu' else 'gpu')
    args=parser.parse_args()
    jax.config.update("jax_enable_x64",args.precision=='float64')
    report=json.loads((args.run/"refine_report.json").read_text())
    if not report.get("valid"): raise ValueError("refinement not accepted")
    cfg,dcfg=load_age_model_config(),load_data_config()
    device=jax.devices('cpu')[0] if args.device=='cpu' else require_gpu("configuration pathology readout")
    data=load_data(cfg["input_dir"],target_device=device,precision=args.precision,verbose=False)
    params,_=load_map_params(str(args.run))
    lat=auto_delta_params_to_latents(params)
    names=["simulated_density","Sa_flat","Sj_flat","Fmax_flat","K_flat",
           "nest_F_real","nest_mean_fledged","nest_cycle_success","nest_broods"]
    if args.save_state:names.append('expected_obs')
    prediction=Predictive(build_model_2d,posterior_samples={k:jnp.asarray(v)[None] for k,v in lat.items()},return_sites=names)
    result=prediction(jax.random.PRNGKey(104),data=data,prior_scale=1.)
    sim={k:np.asarray(v[0]) for k,v in jax.block_until_ready(result).items()}
    rows,cols=np.asarray(data["land_rows"]),np.asarray(data["land_cols"])
    years=np.asarray(data["years"])
    density=sim["simulated_density"]
    if density.ndim==3:
        density=density[:,rows,cols]
    elif density.shape!=(len(years),len(rows)):
        raise ValueError(f"unrecognized density layout: {density.shape}")
    with rasterio.open(dcfg["grid"]["ref_raster"]) as source:
        x,y=rasterio.transform.xy(source.transform,rows,cols,offset="center")
        lon,latitudes=map(np.asarray,crs_transform(source.crs,"EPSG:4326",list(x),list(y)))
    regions={"all_land":np.ones(len(rows),bool),"west":lon < -100,
             "plains":(lon>=-100)&(lon < -95),
             "central_east":(lon>=-95)&(lon < -80)&(latitudes>=25)&(latitudes<49),
             "east":(lon>=-80)&(latitudes>=25)&(latitudes<49)}
    lam=np.asarray(local_growth_lambda(sim["Sa_flat"],sim["Sj_flat"],sim["Fmax_flat"]))
    fields={"Sa":sim["Sa_flat"],"Sj_stayer":sim["Sj_flat"],
            "Sj_over_Sa":sim["Sj_flat"]/np.maximum(sim["Sa_flat"],1e-15),
            "Fmax":sim["Fmax_flat"],"K_relative":sim["K_flat"],"lambda0":lam}
    eras={"early":[1902,1915],"introduction":[1940,1955],"bbs_early":[1966,1985],"modern":[2010,2025]}
    output={"status":"complete","run":str(args.run),"precision":args.precision,"input_dir":cfg["input_dir"],
            "life_cycle":cfg["population_model"].get("reproduction",{}),
            "allee":cfg["population_model"].get("allee_prior",{}),
            "native_departure":cfg["population_model"].get("native_departure",{}),
            "regions":{},"nests":{},"notes":["Population weighting uses model abundance, not independent BBL sampling.",
             "Sj is resident/stayer survival; Q replaces it for movers.","Low-K thresholds are descriptive flags, not adoption gates."]}
    for name,mask in regions.items():
        output["regions"][name]={}
        for era,(first,last) in eras.items():
            t=(years>=first)&(years<=last)
            weights=density[t][:,mask]
            block={key:{"land":describe(value[t][:,mask]),"population":describe(value[t][:,mask],weights)} for key,value in fields.items()}
            block["sink_area_share"]=float((lam[t][:,mask]<=1).mean())
            total=float(weights.sum())
            block["sink_population_share"]=float(np.sum(weights*(lam[t][:,mask]<=1))/total) if total>1e-30 else None
            block["juvenile_ratio_above_095_population_share"]=float(np.sum(weights*(fields["Sj_over_Sa"][t][:,mask]>.95))/total) if total>1e-30 else None
            output["regions"][name][era]=block
    for key in names:
        if key.startswith("nest_") and key in sim:
            output["nests"][key]=describe(sim[key])
    pinned=cfg["population_model"].get("pinned_sites",{})
    log_founders=pinned.get("log_inv_total",lat.get("log_inv_total"))
    if log_founders is not None:
        output["founders_birds_conversion_102_8"]=float(102.8*np.exp(np.asarray(log_founders)))
    for name in ("metrics.json","west_sink_audit.json"):
        p=args.out.parent/"viz_ref"/name
        if p.exists(): output[name]=json.loads(p.read_text())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.save_state:
        np.savez_compressed(args.out.with_suffix('.state.npz'),density=density,years=years,east=(lon>=-95)&(latitudes>=25)&(latitudes<49),expected_obs=sim['expected_obs'])
    tmp=args.out.with_suffix(".tmp.json"); tmp.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n"); tmp.replace(args.out)
    print("Saved",args.out,flush=True)


if __name__=="__main__": main()
