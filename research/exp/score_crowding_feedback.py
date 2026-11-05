"""Flag persistent local oscillation from the existing finite-propagule replays."""
import argparse,json,os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform
p=argparse.ArgumentParser();p.add_argument('step');a=p.parse_args()
R=Path('/mnt/c/Dev/houfin-range-model');root=Path('/mnt/e/Datasets/houfin/movement_followup_20261101')/a.step
# The monitor's runner is the source of the output path; supplied override avoids
# assuming that a future collector retains the same root.
root=Path(os.environ.get('MOVEMENT_OUT',str(root)))
with np.load(root/'local_propagules.npz') as z:
 K=z['K'];seeds={k:z[k] for k in z.files if k.startswith('seed_')}
with np.load(root/'movement_moments.npz') as z:rr,cc=z['land_rows'],z['land_cols']
with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
 xx,yy=rasterio.transform.xy(src.transform,rr,cc);lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
regions={'east':(lon>=-80)&(lat>=25)&(lat<49),'plains':(lon>=-100)&(lon<-95),'west':lon<-100}
result=dict(step=a.step,scope='Last20-year amplitude in existing80-year fixed-environment isolated replays; flags are not proof of a stable cycle',regions={})
for region,mask in regions.items():
 result['regions'][region]={}
 for seed,trajectory in seeds.items():
  tail=trajectory[-20:];amplitude=np.ptp(tail,axis=0)/np.maximum(K,1e-12)
  result['regions'][region][seed]=dict(cells=int(mask.sum()),median_tail_range_over_K=float(np.median(amplitude[mask])),
      share_tail_range_above_point1K=float(np.mean(amplitude[mask]>.1)),
      share_nonfinite=float(np.mean(~np.isfinite(tail[:,mask]).all(0))))
dest=R/'.research_runtime/crowding_feedback'/a.step;dest.mkdir(parents=True,exist_ok=True)
(dest/'score.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
