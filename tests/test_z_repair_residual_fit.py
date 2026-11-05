"""Full small-data training smoke, independent of the real ecological outcomes."""
import json
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from research.exp.z_repair_residual_fit import main


def test_complete_residual_training_and_early_gradient_report(tmp_path,monkeypatch):
    rng=np.random.default_rng(4);root=tmp_path
    data=root/'data/backcast40';aux=root/'aux/backcast40/full'
    cube=root/'cubes/backcast40/control_s1';model=root/'models/backcast40/control_s1'
    for p in [data/'summer_train_points',data/'winter_full/count_points',aux,cube,model]:
        p.mkdir(parents=True)
    (model/'CONTROL_CUBE_DONE').write_text('synthetic test fixture')
    keys=np.array([[r,c,y] for y in [1996,2006] for r in range(2) for c in range(2)])
    early=np.array([[r,c,1902] for r in range(2) for c in range(2)])
    np.save(data/'summer_train_points/point_index.npy',keys)
    np.save(data/'winter_full/count_points/point_index.npy',keys)
    for name in ['summer_train_points','winter_full/count_points']:
        np.save(data/name/'X_points.npy',rng.uniform(.1,1,(8,5)).astype('float32'))
    np.save(data/'summer_train_points/point_weights.npy',np.ones(8,'float32'))
    for name in ['summer','winter']:
        np.save(aux/(name+'_training_coords.npy'),rng.normal(0,.1,(8,64)).astype('float32'))
    for year in [1902,1996,2006]:
        np.save(cube/f'Z_latent_{year}.npy',rng.normal(0,.1,(2,2,128)).astype('float32'))
    np.savez(aux/'cbc_ruzicka.npz',a=np.float32(.1),b=np.float32(.5),weight=np.float32(1))
    np.savez(aux/'cbc_presence_early_w.npz',row=early[:,0],col=early[:,1],year=early[:,2],
             P=np.ones((4,5),'float32'),W=rng.normal(0,.1,(64,5)).astype('float32'),
             b=np.zeros(5,'float32'),comp=np.zeros(4,'float32'))
    grid=root/'grid.tif'
    with rasterio.open(grid,'w',driver='GTiff',width=2,height=2,count=1,dtype='uint8',
                       crs='ESRI:102003',transform=from_origin(0,54000,27000,27000)) as ds:
        ds.write(np.ones((1,2,2),'uint8'))
    monkeypatch.setattr('src.config_utils.load_data_config',lambda:{'grid':{'ref_raster':str(grid)}})
    monkeypatch.setattr(sys,'argv',['residual','--root',str(root),'--steps','2'])
    main()
    out=root/'residuals/backcast40/rank2'
    info=json.loads((out/'FIT_DONE.json').read_text())
    assert info['initial_weighted_gradient_norms']['early_presence']>0
    state=torch.load(out/'model.pt',weights_only=True)
    assert all(torch.isfinite(v).all() for v in state['model'].values())
