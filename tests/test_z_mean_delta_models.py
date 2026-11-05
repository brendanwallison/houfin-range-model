import torch

from research.exp.z_mean_delta_models import MeanDeltaDESK,center_deviations


def test_reference_anchor_mean_map_and_shared_latent_output():
    torch.manual_seed(6)
    model=MeanDeltaDESK([3,2],4,private_dim=3,hidden_width=8,spatial_kernel=3,
                       dropout=0,change_rank=3,context_width=8).eval()
    ref=torch.randn(1,3,4,5);mask=torch.ones(1,3,4)
    state=model.mean_state(ref,mask)
    x=ref+torch.randn_like(ref)
    outputs=model(x,mask,True,reference_covariates=ref,reference_state=state)
    torch.testing.assert_close(outputs[0],state[0],rtol=0,atol=0)
    torch.testing.assert_close(outputs[2],state[1],rtol=0,atol=0)
    with torch.no_grad():model.change_output.weight.normal_()
    anchored=model(ref,mask,True,reference_covariates=ref,reference_state=state)
    torch.testing.assert_close(anchored[0],state[0],rtol=0,atol=0)
    delta=torch.randn_like(ref)
    plus=model(ref+delta,mask,reference_covariates=ref,reference_state=state,return_components=True)
    minus=model(ref-delta,mask,reference_covariates=ref,reference_state=state,return_components=True)
    torch.testing.assert_close(plus[1]+minus[1],torch.zeros_like(plus[1]),atol=2e-6,rtol=0)
    (plus[0]+plus[1]).square().mean().backward()
    assert torch.isfinite(model.change_output.weight.grad).all()


def test_post_ema_centering_preserves_declared_mean_and_gradients():
    mean=torch.randn(2,3,5,requires_grad=True)
    delta=torch.randn(8,2,3,5,requires_grad=True)
    # A finite-window causal smoother may shift the raw deviation's window mean.
    smoothed=torch.stack([delta[:i+1].mean(0) for i in range(len(delta))])
    reference=torch.tensor([False]*3+[True]*5)
    z=center_deviations(mean,smoothed,reference)
    torch.testing.assert_close(z[reference].mean(0),mean)
    z.square().sum().backward()
    assert torch.isfinite(delta.grad).all() and torch.isfinite(mean.grad).all()
