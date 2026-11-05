import hashlib
import numpy as np
import torch

from research.exp.z_change_neural import ChangeVectorLoss


def test_vector_loss_matches_planted_temporal_change_and_preserves_spatial_offsets(tmp_path):
    path=tmp_path/'target.npz'
    np.savez(path,year=np.array([[2000,2001,2010,2011]]),row=np.array([0]),col=np.array([0]),
        target=np.array([[2.,-1.]],'float32'),weight=np.ones(1,'float32'))
    spec={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'calibration_path':str(tmp_path/'calibration.json')}
    loss=ChangeVectorLoss(spec,[2000,2001,2010,2011],'cpu')
    z=torch.tensor([[3.,5.],[3.,5.],[5.,4.],[5.,4.]],requires_grad=True).reshape(4,1,1,2)
    torch.testing.assert_close(loss(z),torch.tensor(0.))
    torch.testing.assert_close(loss(z+100),torch.tensor(0.))
    shifted=z.clone();shifted[2:]=shifted[2:]+1
    assert loss(shifted)>0
