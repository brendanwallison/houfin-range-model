import json
import sys

import numpy as np

from research.exp.z_repair_compare import main


def test_comparison_pairs_endpoints_and_handles_historical_only_circles(tmp_path,monkeypatch):
    root=tmp_path/'evaluation/backcast40'
    rng=np.random.default_rng(8)
    cells=np.column_stack([np.arange(30),np.arange(30)%6])
    truth=rng.normal(size=(30,4))
    for arm,scale in [('control',.5),('time_r2',.8),('residual_r2',1.)]:
        out=root/arm;out.mkdir(parents=True)
        (out/'SUMMER_DONE.json').write_text(json.dumps({'species':['a','b','c','d'],'panels':{'k2_known':{}}}))
        np.savez(out/'k2_known.npz',cells=cells,truth_a=truth,truth_b=truth,
                 predicted_change=scale*truth,historical_observed_mse=np.full(30,1-scale),
                 modern_observed_mse=np.ones(30))
        # Circle99 disappears before modern sampling: indexing must remain valid.
        np.savez(out/'winter_lists.npz',test_rows=np.arange(4),species=np.array(['a','b']),
                 year=np.array([1902,1903,2010,2011]),circle=np.array([99,99,1,1]),
                 backcast=np.full(4,-1+scale*.1),modern=np.full(4,-1.),field_counters=np.ones(4))
    monkeypatch.setattr(sys,'argv',['compare','--root',str(tmp_path)])
    main()
    result=json.loads((root/'PAIRED_COMPARISON.json').read_text())
    for arm in ['time_r2','residual_r2']:
        effect=result['comparisons'][arm]['summer']['k2_known']['treatment_minus_control']
        assert effect['historical_mse_difference']['point']<0
        assert effect['modern_mse_difference']['point']==0
