"""Join actual contribution arrays to saved K/Allee maps and plot diagnostics."""
import argparse,json,os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    import rasterio
    from rasterio.warp import transform
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ready in a.root.glob('*/READY.json'):
        out=ready.parent;field=out/'source_sink_fields.npz'
        if not field.exists() or (out/'CONTEXT.json').exists():continue
        step=out.name
        cfgpath=Path('/mnt/e/Datasets/houfin/grid54/runs')/step/'code/config/overlays/grid54_data.json'
        if not cfgpath.exists():continue
        cfg=json.load(open(cfgpath));ref=cfg['grid']['ref_raster'].replace('${HOUFIN_PROCESSED}','/work/07980/bwa386/ls6/houfin/processed').replace('${HOUFIN_DATA}','/home/breallis/houfin/scratch/houfin/data')
        with np.load(out/'annual_budget.npz') as z:
            lr,lc=z['land_rows'],z['land_cols'];t=(z['years']>=2010)&(z['years']<=2025)
            adult=np.where(z['pulse'][t]==0,z['input_adult'][t],0.)
            den=adult.sum(0);num=np.where(z['pulse'][t]==0,z['breeder_contribution'][t],0.).sum(0)
            cr=np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-20)
            stden=(z['input_adult'][t]+z['input_juvenile'][t]).sum(0)
            cs=np.divide(z['total_descendants_origin'][t].sum(0),stden,out=np.full_like(num,np.nan),where=stden>1e-20)
            dest=z['next_breeders'][t].sum(0)
            im=np.divide(z['immigrant_breeders'][t].sum(0),dest,out=np.full_like(num,np.nan),where=dest>1e-20)
        with rasterio.open(ref) as src:
            xx,yy=rasterio.transform.xy(src.transform,lr,lc)
            lon,lat=map(np.asarray,transform(src.crs,'EPSG:4326',xx,yy))
        with np.load(field) as z:
            shape=z['K_modern'].shape;dead=z['allee_dead_mask'][lr,lc].astype(bool);lam=z['lam_fundamental_modern'][lr,lc];k=z['K_modern'][lr,lc]
        regions={'all':np.ones(len(lr),bool),'west':lon<-100,'plains':(lon>=-100)&(lon<-95),
            'broad_plains':(lon>=-105)&(lon<-95)&(lat>=25)&(lat<49),'east':(lon>=-80)&(lat>=25)&(lat<49)}
        context={}
        for reg,mask in regions.items():
            dm=mask&dead
            context[reg]={'potential_sink_land_share':float((lam[mask]<1).mean()),
              'K_Allee_dead_land_share':float(dead[mask].mean()),
              'K_Allee_dead_C_breeding':float(num[dm].sum()/den[dm].sum()) if den[dm].sum()>0 else None,
              'K_Allee_dead_breeder_weight':float(den[dm].sum()/den[mask].sum()) if den[mask].sum()>0 else None,
              'K_relative_dead_q10_50_90':np.quantile(k[dm],[.1,.5,.9]).tolist() if dm.any() else None}
        (out/'CONTEXT.json').write_text(json.dumps(context,indent=2,allow_nan=False)+'\n')
        fig,axes=plt.subplots(1,4,figsize=(16,6),constrained_layout=True)
        for ax,v,title,cmap,lo,hi in zip(axes,[cr,cs,im,dead.astype(float)],
             ['Breeding contribution C','Stage-weighted contribution C','Cell immigrant fraction','Positive-growth K/Allee nonviability'],
             ['RdBu','RdBu','viridis','Reds'],[.5,.5,0,0],[1.5,1.5,1,1]):
            image=np.full(shape,np.nan);image[lr,lc]=v
            pic=ax.imshow(image,cmap=cmap,vmin=lo,vmax=hi);ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(pic,ax=ax,shrink=.6)
        fig.suptitle(step+' | 2010–2025 | model grid; contribution colors clipped to 0.5–1.5')
        fig.savefig(out/'contribution_maps.png',dpi=160);plt.close(fig)
        print('CONTEXT',step,flush=True)


if __name__=='__main__':main()
