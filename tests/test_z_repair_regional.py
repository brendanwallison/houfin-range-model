import numpy as np

from research.exp.z_repair_regional import partition_change
from research.exp.z_repair_evaluate import bootstrap_components,weights_for_cells


def test_regional_signal_and_local_departures_are_distinguished():
    cells=np.array([[0,0],[0,1],[6,0],[6,1],[12,0]])
    regional=np.array([[-1.,-2.],[-1.,-2.],[1.,2.],[1.,2.],[9.,9.]])
    local=np.array([[.5,-.5],[-.5,.5],[.7,-.7],[-.7,.7],[0.,0.]])
    truth=regional+local
    keep,parts=partition_change(cells,regional,truth,truth)
    np.testing.assert_array_equal(keep,[True,True,True,True,False])
    W,_=weights_for_cells(cells[keep],boot=20)
    r=bootstrap_components(parts[0][0],parts[1][0],parts[2][0],W[:1])
    l=bootstrap_components(parts[0][1],parts[1][1],parts[2][1],W[:1])
    np.testing.assert_allclose(r['corr'],1)
    assert np.isnan(l['corr'][0])  # No local prediction; not a perfect-zero success.
    total=bootstrap_components(regional[keep],truth[keep],truth[keep],W)
    region=bootstrap_components(parts[0][0],parts[1][0],parts[2][0],W)
    departure=bootstrap_components(parts[0][1],parts[1][1],parts[2][1],W)
    for name in ['truth_centered_signal','mse_gain_over_no_change']:
        np.testing.assert_allclose(total[name],region[name]+departure[name],atol=1e-12)
    np.testing.assert_allclose(parts[1][0]+parts[1][1],truth[keep])
