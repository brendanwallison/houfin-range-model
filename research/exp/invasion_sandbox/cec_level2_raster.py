"""Rasterise CEC Level II ecoregions onto the 54-km model lattice, matching cec_level1_54km.tif: 9 x 9 sub-points per cell,
majority of non-water sub-points. Writes <regions>/cec_level2_54km.tif (int codes) + .json (code -> 'L2 code  NAME', L1 parent)."""
import json, sys
from pathlib import Path
import numpy as np, rasterio, geopandas as gpd
from rasterio import features
from affine import Affine
SHP = '/mnt/e/Datasets/reference/epa_na_cec_eco_l2/NA_CEC_Eco_Level2.shp'
TPL = Path('/mnt/e/Datasets/houfin/grid54/regions/cec_level1_54km.tif')
g = gpd.read_file(SHP); print('columns', list(g.columns))
code_col = next(c for c in g.columns if c.upper() in ('NA_L2CODE', 'NA_L2KEY', 'L2_KEY'))
name_col = next((c for c in g.columns if c.upper() in ('NA_L2NAME',)), code_col)
l1_col = next((c for c in g.columns if c.upper() in ('NA_L1CODE',)), None)
with rasterio.open(TPL) as s:
    prof = s.profile.copy(); T = s.transform; H, Wd = s.height, s.width; crs = s.crs
g = g.to_crs(crs)
codes = sorted({str(c) for c in g[code_col]})
idx = {c: i + 1 for i, c in enumerate(c for c in codes if c not in ('0.0', '0'))}
water = [c for c in codes if c in ('0.0', '0')]
shapes = [(geom, idx[str(c)]) for geom, c in zip(g.geometry, g[code_col]) if str(c) in idx]
k = 9
Tsub = T * Affine.scale(1. / k)
sub = features.rasterize(shapes, out_shape=(H * k, Wd * k), transform=Tsub, fill=0, dtype='int32')
sub = sub.reshape(H, k, Wd, k).transpose(0, 2, 1, 3).reshape(H, Wd, k * k)
out = np.zeros((H, Wd), 'int32')
for i in range(H):
    for j in range(Wd):
        v = sub[i, j][sub[i, j] > 0]
        if v.size:
            out[i, j] = np.bincount(v).argmax()
prof.update(dtype='int32', count=1, nodata=0)
with rasterio.open(TPL.parent / 'cec_level2_54km.tif', 'w', **prof) as d:
    d.write(out, 1)
names = {}
for c, n, l1 in zip(g[code_col], g[name_col], g[l1_col] if l1_col else [None] * len(g)):
    if str(c) in idx:
        names[idx[str(c)]] = dict(code=str(c), name=str(n), l1=(str(l1) if l1 is not None else str(c).split('.')[0]))
(TPL.parent / 'cec_level2_54km.json').write_text(json.dumps(dict(source=SHP, template=str(TPL), subgrid=k, rule='majority of non-water sub-points',
                                                                  classes={str(k_): v for k_, v in sorted(names.items())}), indent=1))
print('classes', len(names), '| cells assigned', int((out > 0).sum()))
