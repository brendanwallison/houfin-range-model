import numpy as np

from research.exp.z_repair_teacher_bottleneck import score
from research.exp.z_repair_evaluate import bootstrap_components,weights_for_cells


def test_deterministic_score_matches_existing_estimator_and_identity_oracle():
    rng=np.random.default_rng(17)
    cells=np.column_stack([np.arange(30)*6,np.zeros(30,int)])
    a=rng.normal(size=(30,5));b=a+rng.normal(size=a.shape)*.2
    pred=.5*a
    W,_=weights_for_cells(cells)
    expected=bootstrap_components(pred,a,b,W)['corr'][0]
    result=score(cells,pred,pred,a,b,True)
    np.testing.assert_allclose(result['corr']['point'],expected)
    identity=score(cells,a,b,a,b)
    np.testing.assert_allclose(identity['corr']['point'],1)
