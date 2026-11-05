"""Cache the unchanged E014 count loader, plus observer identities for sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.exp.route_nugget import KEYS, route_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'DONE.json').exists():
        print('Existing completed cache:', out, flush=True)
        return
    print('Loading route counts with the original E014 loader', flush=True)
    R, cov, species = route_counts()
    assert np.array_equal(cov['k'].to_numpy(), np.arange(len(cov)))
    from src.data.preprocess import bbs
    from src.config_utils import load_config, load_data_config
    w = pd.read_csv(bbs.WEATHER_FILE, usecols=KEYS + ['Year', 'RPID', 'RunType', 'ObsN'])
    w = w[(w.RPID == 101) & (w.RunType != 0)].copy()
    w['ObsN'] = pd.to_numeric(w.ObsN, errors='coerce').fillna(-1).astype(int)
    g = w.groupby(KEYS + ['Year']).ObsN
    observer = g.first().where(g.nunique() == 1, -1).rename('observer').reset_index()
    cov = cov.merge(observer, on=KEYS + ['Year'], how='left', validate='one_to_one')
    cov['observer'] = cov.observer.fillna(-1).astype(int)
    assert np.array_equal(cov['k'].to_numpy(), np.arange(len(cov)))
    np.save(out / 'counts.npy', R)
    cov.to_parquet(out / 'coverage.parquet', index=False)
    cfg, dcfg = load_config(), load_data_config()
    reference = Path((cfg.get('trend', {}) or {}).get('community_trend_list')
                     or dcfg['community_trend_list'])
    sources = [Path(__file__), Path(__file__).with_name('route_nugget.py'), reference]
    meta = {
        'shape': list(R.shape), 'species': list(species),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'weather': str(bbs.WEATHER_FILE),
        'input_files': [{'path': str(p), 'size': p.stat().st_size,
                         'mtime_ns': p.stat().st_mtime_ns}
                        for p in [Path(bbs.WEATHER_FILE), *sorted(Path(bbs.BBS_STATES_DIR).glob('*.csv'))]],
    }
    (out / 'DONE.json').write_text(json.dumps(meta, indent=2))
    print('Completed', R.shape, 'route-year counts:', out, flush=True)


if __name__ == '__main__':
    main()
