"""CPU-only inventory/checklist from archived checkpoints and diagnostic arrays.

Does not rerun models, fit parameters or rank incomparable configurations.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


def finite_json(value):
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def inventory(runs,results):
    rows=[]
    for run in sorted(runs.iterdir()):
        if not run.is_dir():continue
        paths=[results/f'refined_float64__age_map_float32_grid54_{run.name}_quick90']
        log=run/'out/refine.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-24000))
                found=re.findall(r'use HOUFIN_HMC_MAP_DIR=(\S+)',stream.read().decode(errors='replace'))
            if found:paths.insert(0,Path(found[-1]))
        ref=next((p for p in paths if (p/'map_checkpoint.pkl').exists()),paths[0])
        report=read(ref/'refine_report.json') or {}
        cfg=read(run/'out/resolved_config.json') or {}
        head=''
        if (run/'out/map.log').exists():
            with (run/'out/map.log').open('rb') as stream:head=stream.read(65536).decode(errors='replace')
        synthetic=re.search(r'\[synthetic\] observed counts replaced from (\S+)',head)
        viz=run/'out/viz_ref'; metrics=read(viz/'metrics.json') or {}
        item={'step':run.name,'refined_dir':str(ref),'checkpoint':(ref/'map_checkpoint.pkl').exists(),
              'valid':report.get('valid'),'frozen_code':(run/'code/src/model/age_priors.py').exists(),
              'manifest':(run/'CODE_MANIFEST.sha256').exists(),'fields':(viz/'07_source_sink_fields.npz').exists(),
              'input_dir':cfg.get('input_dir'),'config_available':bool(cfg),'life_cycle':cfg.get('population_model',{}).get('reproduction',{}),
              'allee':cfg.get('population_model',{}).get('allee_prior',{}),'metrics':metrics,
              'synthetic_observations':synthetic.group(1) if synthetic else None,
              'observation_log_available':bool(head)}
        if item['valid'] and item['fields']:
            with np.load(viz/'07_source_sink_fields.npz',allow_pickle=False) as z:
                k=np.asarray(z['K_modern']);land=np.isfinite(k)
                item['cached']={'grid_shape':list(k.shape),'K_land_min':float(k[land].min()),
                    'K_land_q10_median_q90':np.quantile(k[land],[.1,.5,.9]).tolist(),
                    'fundamental_sink_land_share':float((z['lam_fundamental_modern'][land]<=1).mean()),
                    'allee_dead_land_share':float(z['allee_dead_mask'][land].mean()) if 'allee_dead_mask' in z else None}
            item['west']=read(viz/'west_sink_audit.json')
            item['front_speed']=read(run/'out/front_speed.json')
        rows.append(item)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=inventory(a.runs,a.results)
    out={'status':'complete','root':str(a.runs),'n_runs':len(rows),'n_checkpoints':sum(r['checkpoint'] for r in rows),
         'n_accepted':sum(r['valid'] is True for r in rows),'runs':rows,
         'notes':['Cached fields are area summaries, not population-weighted vital rates.',
                  'Different input directories/life cycles are separate comparison strata.',
                  'Missing remote artifacts do not establish deletion; verify local copies.']}
    a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(finite_json(out),indent=2,allow_nan=False)+'\n');tmp.replace(a.out)
    lines=['# Retrospective pathology checklist — cached diagnostics','',
           f"Root: {a.runs}. {out['n_runs']} run directories; {out['n_checkpoints']} checkpoints; {out['n_accepted']} accepted refinements.",'',
           'Historical fits use different habitat inputs, life cycles and observation models. This table is an inventory of tradeoffs, not a pooled ranking. Lambda change is not itself a pathology. Tiny K in unoccupied cells also requires map/observation context. Synthetic-observation runs identified in the logs are retained in JSON but excluded from this real-data table.','',
           '| Run | Yearling breeding | Lambda change | Suitable share early → modern | Modern K minimum | Allee-dead share of land | Temporal / spatial log score |','|---|---|---:|---|---:|---:|---|']
    for r in rows:
        if not r['valid'] or 'cached' not in r or r['synthetic_observations']:continue
        m=r['metrics'];c=r['cached'];h=m.get('holdout',{})
        change='—' if any(m.get(k) is None for k in ['modern_mean_lambda','early_mean_lambda']) else f"{m['modern_mean_lambda']-m['early_mean_lambda']:+.3f}"
        suitable='—' if any(m.get(k) is None for k in ['early_suitable_fraction','modern_suitable_fraction']) else f"{m['early_suitable_fraction']:.3f} → {m['modern_suitable_fraction']:.3f}"
        scores=' / '.join(f"{h[s]['mean_lpd']:.3f}" if s in h else '—' for s in ['temporal','spatial'])
        dead=c['allee_dead_land_share'];dt='—' if dead is None else f'{dead:.3f}'
        y=(bool(r['life_cycle'].get('yearling_breeding')) or r['life_cycle'].get('form')=='veit') if r['config_available'] else 'unknown'
        lines.append(f"| {r['step']} | {y} | {change} | {suitable} | {c['K_land_min']:.2g} | {dt} | {scores} |")
    a.out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(out['n_runs'],out['n_checkpoints'],out['n_accepted'])


if __name__=='__main__':main()
