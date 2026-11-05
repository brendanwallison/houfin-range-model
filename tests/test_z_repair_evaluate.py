import numpy as np

from research.exp.z_repair_evaluate import bootstrap_components,matched_groups
from research.exp.z_repair_data import summer_masks


def test_direction_score_does_not_hide_magnitude_failure():
    truth=np.array([[-1.,-2.],[0.,1.],[2.,3.]])
    w=np.ones((1,3))
    exact=bootstrap_components(truth,truth,truth,w)
    doubled=bootstrap_components(2*truth,truth,truth,w)
    np.testing.assert_allclose(exact['corr'],1)
    np.testing.assert_allclose(doubled['corr'],1)
    np.testing.assert_allclose(doubled['truth_on_prediction_slope'],.5)
    assert exact['mse_gain_over_no_change'][0]>0
    np.testing.assert_allclose(doubled['mse_gain_over_no_change'],0,atol=1e-15)


def test_matched_effort_uses_only_independent_reserved_records():
    years=np.r_[np.arange(1966,1986),np.arange(1996,2026)]
    keys=np.column_stack([np.zeros(len(years),int),np.zeros(len(years),int),years])
    zero=np.zeros((2,2),bool)
    masks=summer_masks(keys,zero,zero,~zero,{'target':[1966,1985],'summer_start':1996})
    cells,spatial,groups=matched_groups(keys,masks,2)
    assert len(cells)==1
    used=[]
    for epoch in groups[0]:
        for half in epoch:
            assert len(half)==2;assert not masks['train'][half].any();used.extend(half)
    assert len(set(used))==8
