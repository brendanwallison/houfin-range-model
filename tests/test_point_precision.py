import numpy as np
import pytest
from research.exp.compare_point_precision import compare


def test_precision_arrival_shift_uses_one_reference_threshold():
    a={'years':np.array([2010,2011,2012]),'east':np.array([True]),
       'density':np.array([[0.],[1.],[1.]]),'expected_obs':np.array([1.])}
    b={**a,'density':np.array([[0.],[0.],[1.]])}
    out=compare(a,b)
    assert out['arrival']['0.1']['median_float32_minus_float64_years']==1
    assert out['expected_count_log1p_rmse']==0


def test_precision_comparison_refuses_different_designs():
    a={'years':np.array([2010]),'east':np.array([True]),'density':np.ones((1,1)),'expected_obs':np.array([1.])}
    with pytest.raises(ValueError,match='designs differ'):compare(a,{**a,'years':np.array([2011])})
