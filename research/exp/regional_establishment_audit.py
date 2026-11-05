"""Regional cached establishment diagnostics; no fitting or JAX execution.

Uses exactly the saved modern K and Allee-dead mask, not a new definition of
establishment. Positive-record weights count repeated survey records, not cells.
"""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--audit',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
R=args.root
d=json.load(open(args.audit))
cache={};out=[]
for run in d['runs']:
 if not run['valid'] or run['synthetic_observations'] or not run['observation_log_available']:continue
 s=run['step'];p=R/'runs'/s
 files=list((p/'out').glob('**/07_source_sink_fields.npz'))
 if not files:continue
 inp=run['input_dir']
 if inp not in cache:
  m=pickle.load(open(Path(inp)/'metadata.pkl','rb'))
  cfg=json.load(open(p/'code/config/overlays/grid54_data.json'))
  raster=cfg['grid']['ref_raster'].replace('${HOUFIN_PROCESSED}',str(R.parents[1]/'processed')).replace('${HOUFIN_DATA}',str(R.parents[1]/'data'))
  with rasterio.open(raster) as src:
   rr,cc=np.indices((src.height,src.width));x,y=rasterio.transform.xy(src.transform,rr.ravel(),cc.ravel())
   lon,lat=transform(src.crs,'EPSG:4326',x,y);lon=np.array(lon).reshape(rr.shape);lat=np.array(lat).reshape(rr.shape)
  oy=np.asarray(m['years'])[np.asarray(m['obs_time_indices'],int)]
  obs=(oy>=2010)&(oy<=2025)&(np.asarray(m['observed_results'])>0)
  cache[inp]=(lon,lat,np.asarray(m['obs_rows'],int)[obs],np.asarray(m['obs_cols'],int)[obs],float(m['pop_scalar']),m.get('invasion_year'))
 lon,lat,orr,occ,scale,iy=cache[inp]
 with np.load(files[0]) as z:
  k=z['K_modern'];land=np.isfinite(k);dead=z['allee_dead_mask'].astype(bool)
  row={'step':s,'regions':{},'input_invasion_year':iy,'release_allee_exemption_years':run['config']['population_model'].get('release_allee_exemption_years')}
  for name,mask in {'all_land':land,'plains_100W_95W':land&(lon>=-100)&(lon<-95),'broad_plains_105W_95W_25N_49N':land&(lon>=-105)&(lon<-95)&(lat>=25)&(lat<49)}.items():
   pos=mask[orr,occ];n=int(pos.sum())
   row['regions'][name]={'land_cells':int(mask.sum()),'dead_cells':int(dead[mask].sum()),'dead_land_fraction':float(dead[mask].mean()),'positive_records':n,'dead_positive_records':int(dead[orr[pos],occ[pos]].sum()),'dead_positive_fraction':float(dead[orr[pos],occ[pos]].mean()) if n else None,'K_low_positive_fraction':float((k[orr[pos],occ[pos]]*scale<.05).mean()) if n else None}
  out.append(row)
path=args.out
path.write_text(json.dumps(out,indent=2,default=lambda a:a.item() if hasattr(a,'item') else str(a)))
for r in out:
 if r['step'] in ['step33_nestcbc_s41','step33_nestcbc_abs_s41','step34_abs_natdep_s41','step34_abs_natdep_s42','step34_abs_v2_natdep_s41','step34_abs_v2_natdep_s42']:
  print(json.dumps(r))
