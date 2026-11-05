"""Build matched CBC observation operators for the later-state pilot."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from pyproj import Geod,Transformer
from scipy.sparse import csr_matrix,eye
from scipy.spatial import cKDTree

repo=Path('/mnt/c/Dev/houfin-range-model');out=repo/'.research_runtime/cbc_joint_inputs_v2';out.mkdir(exist_ok=True)
cfg=json.loads(Path('/mnt/e/Datasets/houfin/grid54/runs/step35_est1951_ctl_mid_s41/code/config/overlays/grid54_step35_est1951_ctl_mid_s41.json').read_text())
m=pickle.load(open(Path(cfg['input_dir'])/'metadata.pkl','rb'))
rr,cc=np.asarray(m['land_rows']),np.asarray(m['land_cols']);years=np.asarray(m['years'])
with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
    trans,crs,shape=src.transform,src.crs,src.shape
    xx,yy=rasterio.transform.xy(trans,rr,cc)
lon,lat=map(np.asarray,transform(crs,'EPSG:4326',xx,yy));xy=np.column_stack([xx,yy]);tree=cKDTree(xy)
forward=Transformer.from_crs('EPSG:4326',crs,always_xy=True);geod=Geod(ellps='WGS84')
lut=np.full(shape,-1,int);lut[rr,cc]=np.arange(len(rr))
f=pd.read_parquet('/mnt/e/Datasets/houfin/joint_observations_20261101/cbc/circle_years.parquet')
nraw=len(f);d=pd.to_datetime(f.date)
window=((d.dt.month==12)&(d.dt.day>=14))|((d.dt.month==1)&(d.dt.day<=5))
f=f.loc[window&f.date_year_mismatch.eq(0)&~f.unknown_count.astype(bool)&f['count'].notna()].copy()
d=pd.to_datetime(f.date);f['summer_year']=d.dt.year-(d.dt.month==1).astype(int)
f=f[f.summer_year.between(1951,2025)].copy();before_effort=len(f)
f=f[f.party_hours.notna()&(f.party_hours>0)].copy()
after_effort=len(f)
if not np.all(f['count']==np.floor(f['count'])):raise ValueError('CBC count not integer')
if f.duplicated(['abbrev','count_yr']).any():raise ValueError('duplicate circle/winter')
circles=f.groupby('abbrev',sort=True)[['latitude','longitude']].median().reset_index()
# Keep distinct relocated coordinates from being silently merged into one footprint.
spread=f.groupby('abbrev').agg(latspan=('latitude',lambda a:a.max()-a.min()),lonspan=('longitude',lambda a:a.max()-a.min()))
if ((spread.latspan>.1)|(spread.lonspan>.1)).any():
    bad=spread.index[(spread.latspan>.1)|(spread.lonspan>.1)]
    f=f[~f.abbrev.isin(bad)].copy();circles=circles[~circles.abbrev.isin(bad)].reset_index(drop=True)
else:bad=[]
qrows=[];qcols=[];qvalues=[]
# CBC radius 7.5 statute miles. Equal-area polar quadrature, 4 radii x 16 angles.
for ci,r in circles.iterrows():
    angle=np.tile(np.arange(16)*22.5,4);radius=np.repeat(np.sqrt((np.arange(4)+.5)/4)*12070.08,16)
    gx,gy,_=geod.fwd(np.full(64,r.longitude),np.full(64,r.latitude),angle,radius)
    x,y=forward.transform(gx,gy);rows,cols=rasterio.transform.rowcol(trans,x,y)
    rows,cols=np.asarray(rows),np.asarray(cols);inside=(rows>=0)&(rows<shape[0])&(cols>=0)&(cols<shape[1])
    idx=lut[rows[inside],cols[inside]];idx=idx[idx>=0]
    qrows.extend([ci]*len(idx));qcols.extend(idx);qvalues.extend([1/64]*len(idx))
Q=csr_matrix((qvalues,(qrows,qcols)),shape=(len(circles),len(rr)))
valid=np.asarray(Q.sum(1)).ravel()>0
circles=circles[valid].reset_index(drop=True);Q=Q[valid]
f=f[f.abbrev.isin(circles.abbrev)].copy();ci=pd.Index(circles.abbrev).get_indexer(f.abbrev)
cx,cy=forward.transform(circles.longitude.to_numpy(),circles.latitude.to_numpy())
blockx=np.floor(np.asarray(cx)/216000).astype(int);blocky=np.floor(np.asarray(cy)/216000).astype(int)
blocks=[f'{x},{y}' for x,y in zip(blockx,blocky)]
held=np.array([int(hashlib.sha256(('cbc_joint_v1:'+b).encode()).hexdigest()[:8],16)%5==0 for b in blocks])
# The experiment explicitly supervises the establishment area. Keep its spatial blocks
# in training by geography, never by observed finch counts. Validate elsewhere/through time.
release=np.asarray(m['inv_locations'])[0]
release_i=np.flatnonzero((rr==release[0])&(cc==release[1]))
if len(release_i)!=1:raise ValueError('release not a unique land cell')
touch=np.asarray(Q[:,release_i].sum(1)).ravel()>0
founder_blocks=set(np.asarray(blocks)[touch]);held[np.isin(blocks,list(founder_blocks))]=False
spatial=held[ci];temporal=f.summer_year.to_numpy()>=2021
train=~spatial&~temporal
feeder=np.log1p(np.maximum(f.feeder_hrs.fillna(0).to_numpy(),0))
# Explicit feeder missingness is retained as a covariate rather than assigned known zero.
feeder_missing=f.feeder_hrs.isna().to_numpy().astype(float)
receipt={'raw_rows':nraw,'eligible_before_effort':before_effort,'measured_effort_rows':len(f),
 'excluded_missing_effort':before_effort-after_effort,
 'relocated_circles_excluded':list(map(str,bad)),'circles':len(circles),'train':int(train.sum()),
 'spatial_holdout':int(spatial.sum()),'temporal_holdout_nonspatial':int((temporal&~spatial).sum()),
 'historical_1951_1965_train':int((train&(f.summer_year.to_numpy()<=1965)).sum()),
 'historical_positive_train':int((train&(f.summer_year.to_numpy()<=1965)&(f['count'].to_numpy()>0)).sum()),
 'historical_founder_cell_train':int((train&(f.summer_year.to_numpy()<=1965)&touch[ci]).sum()),
 'historical_east_train':int((train&(f.summer_year.to_numpy()<=1965)&(f.longitude.to_numpy()>-80)).sum()),
 'founder_training_blocks':sorted(founder_blocks),
 'count_alignment':'December y and January y+1 observe postbreeding summer y; no pre-initialization CBC likelihood',
 'effort_policy':'Measured party-hours only. Missing retained in raw export, excluded here, never treated as zero.',
 'scope':'CBC relative abundance supervision, not a founder census or absolute density calibration',
 'seasonal_map':'Conservative static southwest redistribution at observation time; no return or feedback to annual demographic process',
 'circle_radius_m':12070.08,'circle_quadrature_points':64,'spatial_split':'216km projected blocks, SHA256 cbc_joint_v1 modulo5 ==0, except geographically defined founder blocks retained for supervision',
 'spatial_transfer_caveat':'CBC held out in these blocks; existing BBS fits are not jointly refitted with corresponding BBS holdouts. No independent BBS transfer claim.'}
f.assign(circle_index=ci,train=train,spatial_holdout=spatial,temporal_holdout=temporal).to_parquet(out/'rows.parquet',index=False)
circles.to_parquet(out/'circles.parquet',index=False)
gate=(1/(1+np.exp(-(lat-37)/2)))*(1/(1+np.exp(-(lon+100)/2)))
for distance in [0,300,600]:
    if distance==0:M=eye(len(rr),format='csr')
    else:
        dest=[];origin=[];weights=[]
        for bearing in [202.5,225.,247.5]:
            gx,gy,_=geod.fwd(lon,lat,np.full(len(rr),bearing),np.full(len(rr),distance*1000.))
            x,y=forward.transform(gx,gy);_,ii=tree.query(np.column_stack([x,y]))
            dest.extend(ii);origin.extend(np.arange(len(rr)));weights.extend(gate/3)
        dest.extend(np.arange(len(rr)));origin.extend(np.arange(len(rr)));weights.extend(1-gate)
        M=csr_matrix((weights,(dest,origin)),shape=(len(rr),len(rr)))
    if not np.allclose(np.asarray(M.sum(0)),1.,atol=1e-12):raise ValueError('seasonal mass not conserved')
    S=(Q@M).tocsr();support=(Q+S).tocsr();width=max(np.diff(support.indptr));idx=np.zeros((len(circles),width),np.int32)
    w0=np.zeros_like(idx,dtype=float);w1=w0.copy()
    for i in range(len(circles)):
        jj=support.indices[support.indptr[i]:support.indptr[i+1]];idx[i,:len(jj)]=jj
        w0[i,:len(jj)]=Q[i,jj].toarray().ravel();w1[i,:len(jj)]=S[i,jj].toarray().ravel()
    arrays={'cbc_source_idx':idx,'cbc_w_local':w0,'cbc_w_seasonal':w1,'cbc_circle':ci.astype(np.int32),
      'cbc_t':pd.Index(years).get_indexer(f.summer_year).astype(np.int32),'cbc_count':f['count'].to_numpy(),
      'cbc_log_effort':np.log(f.party_hours.to_numpy()/20.),'cbc_feeder':feeder,'cbc_feeder_missing':feeder_missing,
      'cbc_train':train.astype(float),'cbc_spatial_holdout':spatial.astype(float),'cbc_temporal_holdout':(temporal&~spatial).astype(float)}
    if (arrays['cbc_t']<0).any():raise ValueError('CBC year outside model')
    path=out/f'cbc_joint_{distance}km.npz';np.savez_compressed(path,**arrays)
    receipt[f'operator_{distance}km']={'width':int(width),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'source_gate':'logistic((lat-37)/2) * logistic((lon+100)/2)',
      'distance_status':'300/600km sensitivity scenarios, not estimated migration distances or permanent dispersal kernels',
      'destination_assignment':'nearest modeled land cell for each of three SW bearings; columns conserve abundance'}
(out/'RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if receipt['historical_founder_cell_train']<1:raise ValueError('founder CBC supervision absent')
