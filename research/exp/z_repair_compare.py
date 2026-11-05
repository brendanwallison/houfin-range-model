"""Paired development comparisons; never promote an encoder automatically."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.exp.z_repair_evaluate import bootstrap_components,weights_for_cells
from research.exp.final_winter_score import ERAS,paired_circle_summary


def summary(values):
    good=np.isfinite(values[1:])
    return {'point':float(values[0]) if np.isfinite(values[0]) else None,
            'valid_bootstraps':int(good.sum()),
            'ci95':np.percentile(values[1:][good],[2.5,97.5]).tolist() if good.mean()>=.95 else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=['time_r2','residual_r2'])
    p.add_argument('--reference',default='control')
    p.add_argument('--evaluation-root',type=Path,default=None)
    p.add_argument('--output-stem',default='PAIRED_COMPARISON')
    a=p.parse_args();root=a.evaluation_root or a.root/'evaluation/backcast40'
    control=json.loads((root/a.reference/'SUMMER_DONE.json').read_text())
    cw=dict(np.load(root/a.reference/'winter_lists.npz'))
    result={'scope':'Initial development comparison on reused ecological records, new exclusion masks; no adoption.',
            'control':a.reference,
            'comparisons':{}}
    for arm in a.arms:
        treatment=json.loads((root/arm/'SUMMER_DONE.json').read_text())
        if control['species']!=treatment['species']:raise ValueError('Species differ')
        item={'summer':{},'winter':{}}
        for panel in control['panels']:
            c=dict(np.load(root/a.reference/(panel+'.npz')))
            d=dict(np.load(root/arm/(panel+'.npz')))
            for k in ['cells','truth_a','truth_b']:
                if not np.array_equal(c[k],d[k]):raise ValueError(f'Unpaired {arm} {panel} {k}')
            W,nblocks=weights_for_cells(c['cells'])
            C=bootstrap_components(c['predicted_change'],c['truth_a'],c['truth_b'],W)
            D=bootstrap_components(d['predicted_change'],d['truth_a'],d['truth_b'],W)
            scores={k:summary(D[k]-C[k]) for k in ['corr','mse_gain_over_no_change']}
            for endpoint in ['historical','modern']:
                v=d[endpoint+'_observed_mse']-c[endpoint+'_observed_mse']
                scores[endpoint+'_mse_difference']=summary((W@v)/W.sum(1))
            item['summer'][panel]={'cells':len(c['cells']),'blocks':nblocks,
                                   'treatment_minus_control':scores}
        tw=dict(np.load(root/arm/'winter_lists.npz'))
        for k in ['test_rows','species','year','circle']:
            if not np.array_equal(cw[k],tw[k]):raise ValueError('Winter support differs: '+k)
        metrics={'backcast_difference':tw['backcast']-cw['backcast'],
                 'modern_placebo_difference':tw['modern']-cw['modern'],
                 'temporal_gain_difference':(tw['backcast']-tw['modern'])-(cw['backcast']-cw['modern'])}
        circles,years=cw['circle'],cw['year']
        modern_counts=np.bincount(circles[years>=2006],minlength=int(circles.max())+1)
        for era,lo,hi in ERAS:
            m=(years>=lo)&(years<=hi)
            counts=np.bincount(circles[m],minlength=len(modern_counts))
            stable=m&(counts[circles]>=3)&(modern_counts[circles]>=3)
            low=m&(cw['field_counters']>0)&(cw['field_counters']<=6)
            item['winter'][era]={label:paired_circle_summary(circles[keep],{k:v[keep] for k,v in metrics.items()})
                                for label,keep in [('primary',m),('same_circles',stable),('counters_1_6',low)]}
        result['comparisons'][arm]=item
    (root/(a.output_stem+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Initial masked reconstruction comparison','',result['scope'],'',
           'Positive correlation/gain differences favor the candidate; negative endpoint-MSE differences favor it.','',
           '| Candidate | Panel | Change correlation difference | Historical MSE difference | Modern MSE difference |',
           '|---|---|---:|---:|---:|']
    def fmt(d):
        if d['point'] is None:return 'unsupported'
        ci=d['ci95'];return f"{d['point']:+.4f}"+(f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ' [unsupported interval]')
    for arm,item in result['comparisons'].items():
        for panel,p in item['summer'].items():
            s=p['treatment_minus_control']
            lines.append(f"|{arm}|{panel}|{fmt(s['corr'])}|{fmt(s['historical_mse_difference'])}|{fmt(s['modern_mse_difference'])}|")
    lines+=['',f'Winter paired effects and effort/circle sensitivities are in {a.output_stem}.json.',
            'These runs do not validate early summer or identify ecological versus observation effects.']
    (root/(a.output_stem+'.md')).write_text('\n'.join(lines)+'\n')
    print(root/(a.output_stem+'.md'))


if __name__=='__main__':main()
