"""Plot a checked edge-connected potential-growth bridge and realized source maps."""
from pathlib import Path
from collections import deque
import numpy as np
import rasterio
from rasterio.warp import transform
from scipy.ndimage import binary_dilation,binary_propagation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

repo=Path('/mnt/c/Dev/houfin-range-model');step='step35_est1951_nat_mid_s41'
with np.load(Path('/mnt/e/Datasets/houfin/contribution_20261101_v3')/step/'annual_budget.npz') as z:
    rr,cc=z['land_rows'],z['land_cols'];tt=(z['years']>=2010)&(z['years']<=2025)
    a=z['input_adult'][tt];n=a+z['input_juvenile'][tt]
    ratio=lambda x,y:np.divide(x,y,out=np.full_like(x,np.nan),where=y>1e-20)
    fields=[z['potential_lambda'][tt].mean(0),ratio(z['breeder_contribution'][tt].sum(0),a.sum(0)),
      ratio((np.nan_to_num(z['local_total_replacement'][tt])*n).sum(0),n.sum(0))]
with rasterio.open('/work/07980/bwa386/ls6/houfin/processed/grid54/ref_grid_54km.tif') as src:
    shape=src.shape;xx,yy=rasterio.transform.xy(src.transform,rr,cc)
    lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
land=np.zeros(shape,bool);land[rr,cc]=True;water=~land;seed=np.zeros(shape,bool)
seed[[0,-1],:]=water[[0,-1],:];seed[:,[0,-1]]=water[:,[0,-1]]
coast=(land&binary_dilation(binary_propagation(seed,mask=water)))[rr,cc]
keep=(lat>=25)&(lat<49);source=keep&(fields[0]>1)
starts=np.where(source&coast&(lon<-120))[0];goals=set(np.where(source&coast&(lon>-81))[0])
index={(int(r),int(c)):i for i,(r,c) in enumerate(zip(rr,cc))}
parent={int(i):None for i in starts};queue=deque(parent);end=None
while queue:
    i=queue.popleft()
    if i in goals:end=i;break
    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        j=index.get((int(rr[i])+dr,int(cc[i])+dc))
        if j is not None and source[j] and j not in parent:parent[j]=i;queue.append(j)
if end is None:raise RuntimeError('Expected source bridge absent')
path=[]
while end is not None:path.append(end);end=parent[end]
path=np.array(path[::-1]);np.savez(repo/'research/experiments/SOURCE_BRIDGE_PATH_20261101.npz',land_indices=path,lon=lon[path],lat=lat[path])
fig,axes=plt.subplots(3,1,figsize=(11,12),sharex=True,sharey=True,layout='constrained')
for ax,v,title in zip(axes,fields,['Potential growth: lambda > 1','Runge breeding contribution: C > 1','Actual-state no-movement replacement > 1']):
    ax.scatter(lon[keep],lat[keep],c=(v[keep]>1).astype(int),cmap=ListedColormap(['#d7c2a2','#24815d']),vmin=0,vmax=1,marker='s',s=8,linewidths=0,rasterized=True)
    ax.axvspan(-100,-95,color='#4466bb',alpha=.12);ax.axvline(-100,color='#4466bb',lw=.7);ax.axvline(-95,color='#4466bb',lw=.7)
    ax.set_title(title,loc='left');ax.set_ylabel('Latitude');ax.set_xlim(-126,-66);ax.set_ylim(25,49);ax.grid(alpha=.15)
axes[0].plot(lon[path],lat[path],color='#c62828',lw=1.5,label='One edge-connected source path')
axes[0].legend(loc='lower left',fontsize=9)
axes[2].set_xlabel('Longitude')
fig.suptitle('Later-state middle abundance + native departure | 2010-2025\nSource continuity differs between potential growth and realized contribution',fontsize=14)
fig.legend(handles=[Line2D([],[],marker='s',linestyle='',color='#24815d',label='Above replacement (>1)'),
 Line2D([],[],marker='s',linestyle='',color='#d7c2a2',label='At/below replacement (<=1)'),
 Line2D([],[],color='#4466bb',lw=5,alpha=.3,label='Primary Plains band')],loc='outside lower center',ncol=3)
out=repo/'research/experiments/SOURCE_BRIDGE_MAP_20261101.png';fig.savefig(out,dpi=180);plt.close(fig)
print('BRIDGE PATH CELLS',len(path),'MAP',out)
