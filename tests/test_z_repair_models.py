import numpy as np
import pytest
import torch

from research.exp.z_repair_models import (
    CubicTimeBasis,GaussianRegionalBasis,LowRankCommunityResidual,TimeConditionedDESK)
from src.community_encoder.train_DESK.model_arch import MultiStreamAutoencoder


def test_zero_modulation_reproduces_existing_desk_and_explicit_time_changes_outputs():
    torch.manual_seed(2)
    base=MultiStreamAutoencoder([3,2],4,spatial_kernel=3,hidden_width=8,dropout=0,private_dim=3).eval()
    model=TimeConditionedDESK([3,2],4,spatial_kernel=3,hidden_width=8,dropout=0,private_dim=3,
                              temporal_width=5,time_rank=2).eval()
    missing,extra=model.load_state_dict(base.state_dict(),strict=False)
    assert all(k.startswith('time_modulation.') for k in missing) and not extra
    x=torch.randn(2,3,4,5);mask=torch.ones(2,3,4);t=torch.randn(2,5)
    expected=base(x,mask,with_private=True)
    actual=model(x,mask,with_private=True,temporal_basis=t)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with torch.no_grad():model.time_modulation.time_weights.fill_(.3)
    changed=model(x,mask,temporal_basis=t)[0]
    assert not torch.equal(changed,expected[0])
    zero=model(x,mask,with_private=True,temporal_basis=torch.zeros_like(t))
    for a,b in zip(zero,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_residual_nested_null_and_planted_rank_two_recovery():
    torch.manual_seed(8)
    model=LowRankCommunityResidual(6,2,4,5).double()
    z=torch.randn(40,6,dtype=torch.float64)
    s=torch.randn(40,4,dtype=torch.float64);t=torch.randn(40,5,dtype=torch.float64)
    torch.testing.assert_close(model(z,s,t),z,rtol=0,atol=0)
    u=torch.linalg.qr(model.directions.detach()).Q
    planted=torch.randn(4,5,2,dtype=torch.float64)*.02
    target=torch.einsum('np,nq,pqr->nr',s,t,planted)@u.T
    # Independent linear solve recovers a planted correction using only training rows.
    design=torch.einsum('np,nq->npq',s,t).reshape(40,-1)
    estimate=torch.linalg.lstsq(design[:30],(target@u)[:30]).solution.reshape(4,5,2)
    with torch.no_grad():model.scores.copy_(estimate)
    torch.testing.assert_close(model(z,s,t)-z,target,rtol=1e-10,atol=1e-12)
    model(z,s,t).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_time_anchor_and_declared_domain():
    basis=CubicTimeBasis(reference=(2006,2025))
    np.testing.assert_allclose(basis(np.arange(2006,2026)).mean(0),0,atol=1e-15)
    assert np.isfinite(basis(np.arange(1902,2026))[:,0]).all()
    with pytest.raises(ValueError):basis([1901])


def test_spatial_centers_use_only_supplied_training_coordinates():
    xy=np.array([[0,0],[100,0],[0,100],[100,100]],float)
    basis=GaussianRegionalBasis(xy,n_centers=3,scale_km=150)
    assert all(tuple(c) in set(map(tuple,xy)) for c in basis.centers)
    values=basis(xy)
    np.testing.assert_array_equal(values[:,0],1)
    assert np.isfinite(values).all()
