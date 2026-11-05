"""Post-fit range pressure and original/derived counterfactual comparison."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from src.config_utils import load_age_model_config
from src.model.checkpoints import load_map_params,auto_delta_params_to_latents

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--budget',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=load_age_model_config();spec=cfg['population_model']['historical_range']
    with np.load(spec['file']) as z:d={k:z[k] for k in z.files}
    with np.load(a.budget) as z:n=z['next_adult']+z['next_juvenile'];young=z['next_juvenile'];years=z['years'];rr=z['land_rows'];cc=z['land_cols']
    m=pickle.load(open(Path(cfg['input_dir'])/'metadata.pkl','rb'));scale=float(m['pop_scalar']);n*=scale;young*=scale
    params,_=load_map_params(str(a.run));lat=auto_delta_params_to_latents(params)
    zero=lambda x,k:k*np.log1p(np.maximum(x,0)/k)
    k=float(spec['fixed_concentration']);zt,zc=d['hr_zero_t'],d['hr_zero_cell'];zw=d['hr_zero_weight']
    zl=zero(n[zt,zc],k)*zw
    available=np.sum(n[d['hr_positive_t'][:,None],d['hr_positive_idx']]*d['hr_positive_w'],axis=1)
    pl=-np.log(-np.expm1(-zero(available+1e-12,k)))*d['hr_positive_weight']
    maps=Path(spec['file']).parent/'comparison_maps.npz'
    with np.load(maps) as z:lon=z['lon'];latgrid=z['lat']
    regions=dict(east=lon>=-80,great_plains=(lon>=-100)&(lon<-95),west=lon<-100,other=(lon>=-95)&(lon<-80))
    out=dict(status='complete',scope='Conditional accepted-MAP range-constraint pressure; not CBC count validation',
      fixed_range_concentration=k,fitted_BBS_concentration=float(lat['concentration']),absence_loss=float(zl.sum()),presence_loss=float(pl.sum()),regions={})
    for name,mask in regions.items():
        out['regions'][name]={}
        for era,(lo,hi) in dict(historical=(1902,1939),invasion=(1951,1965)).items():
            keep=mask[zc]&(years[zt]>=lo)&(years[zt]<=hi)
            out['regions'][name][era]=dict(absence_loss=float(zl[keep].sum()),effective_weight=float(zw[keep].sum()))
    lut={(int(r),int(c)):i for i,(r,c) in enumerate(zip(rr,cc))}
    oi=np.array([lut[(int(r),int(c))] for r,c in zip(m['obs_rows'],m['obs_cols'])]);t=np.asarray(m['obs_time_indices'])
    previous=np.where(t>0,n[np.maximum(t-1,0),oi],np.asarray(m['initpop_latent'])[np.asarray(m['obs_rows']),np.asarray(m['obs_cols'])]*scale)
    # t=0 native seed depends on fitted K, not this metadata shape. Avoid guessing
    # it: report the exact t>0 part, with the initial year explicitly excluded.
    mu=np.maximum(previous+float(lat['obs_fledgling_weight'])*young[t,oi],1e-6)
    old_concentration=float(lat['concentration'])
    if cfg['population_model']['observation'].get('conc_density_power'):
        old_concentration=old_concentration*np.maximum(mu,1e-3)**float(lat['obs_conc_kappa'])
    w=d['hr_old_zero_weight'];old_keep=(t>0)&(w>0)
    out['original_pseudozero_counterfactual_1903_1939']=dict(
        loss=float((zero(mu,old_concentration)*w*old_keep).sum()),
        scope='Old active pseudozero weights and original breeding readout at the new trajectory; excludes1902 because fitted native initial K is not saved in the budget')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
