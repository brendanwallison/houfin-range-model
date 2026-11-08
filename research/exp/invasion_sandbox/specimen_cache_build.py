"""Precompute the museum-specimen likelihood cache (specimen_lik.build_cache) for laplace_fit --specimens, so the fit host needs
no GBIF files, rasters or pandas. Usage: specimen_cache_build.py OUT.npz [--min_neff 5]"""
import argparse, pickle
import numpy as np
import specimen_lik as SL

p = argparse.ArgumentParser(); p.add_argument('out'); p.add_argument('--min_neff', type=float, default=0.)
a = p.parse_args()
m = pickle.load(open('/mnt/e/Datasets/houfin/grid54/inputs_shape/established1951_vl/metadata.pkl', 'rb'))
c = SL.build_cache(np.asarray(m['years']), min_neff_absence=a.min_neff)
np.savez_compressed(a.out, **{k: np.asarray(c[k]) for k in ('w', 'cell', 'dec', 'E', 'y', 'pos', 'tmask', 'era', 'neff')},
                    raw=c['raw'], min_neff=a.min_neff, n_land=len(np.asarray(m['land_rows'])))
print(a.out, 'rows', c['n'], 'positive', int(np.asarray(c['pos']).sum()), 'weighted rows', float(np.asarray(c['w']).sum()))
