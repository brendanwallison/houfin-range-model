"""Compare float32/float64 forward replays of one fixed fitted point."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(a,b):
    """a=float64 reference, b=float32; use the reference's threshold for both."""
    for key in ['years','east']:
        if not np.array_equal(a[key],b[key]):raise ValueError('replay designs differ')
    if a['density'].shape!=b['density'].shape:raise ValueError('population shapes differ')
    years=a['years'];east=a['east'];ref=a['density'][:,east];other=b['density'][:,east]
    modern=ref[years>=2010];scale=float(np.median(modern[modern>0]))
    if not np.isfinite(scale) or scale<=0:raise ValueError('no positive reference population')
    result={'status':'complete','same_parameters':True,'reference_modern_density_scale':scale,
            'density_log1p_rmse':float(np.sqrt(np.mean((np.log1p(a['density'])-np.log1p(b['density']))**2))),
            'expected_count_log1p_rmse':float(np.sqrt(np.mean((np.log1p(a['expected_obs'])-np.log1p(b['expected_obs']))**2))),
            'arrival':{}}
    for fraction in [.01,.10]:
        x=ref>=fraction*scale;y=other>=fraction*scale
        common=x.any(0)&y.any(0)
        delta=years[y[:,common].argmax(0)]-years[x[:,common].argmax(0)]
        result['arrival'][str(fraction)]={'common_arrived_cells':int(common.sum()),
            'median_float32_minus_float64_years':float(np.median(delta)) if len(delta) else None,
            'share_absolute_shift_at_least_5_years':float((abs(delta)>=5).mean()) if len(delta) else None,
            'float64_only_arrived_cells':int((x.any(0)&~y.any(0)).sum()),
            'float32_only_arrived_cells':int((~x.any(0)&y.any(0)).sum())}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('other',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with np.load(args.reference,allow_pickle=False) as a,np.load(args.other,allow_pickle=False) as b: result=compare(a,b)
    tmp=args.out.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(args.out)
