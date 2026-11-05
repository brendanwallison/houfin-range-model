"""Competing mean-map/deviation DESK architecture; no production trainer changes.

The mean map uses reference covariate levels. Change is linear in covariate
deltas with nonlinear, place-dependent context gates. The exported community
state is one sum; focal models do not receive separate spatial/time readouts.
"""
import torch
from torch import nn

from src.community_encoder.train_DESK.model_arch import MultiStreamAutoencoder


def center_deviations(mean,delta,reference):
    """Define a mean/departure split for the explicitly supplied output series.

    Raw latent deviations are centered before the observation EMA is applied.
    Centering an already smoothed series defines an observation-space mean;
    it must not silently replace the raw mean or the instantaneous export.
    """
    if delta.ndim!=mean.ndim+1 or tuple(delta.shape[1:])!=tuple(mean.shape):
        raise ValueError('Expected delta=(year,*mean.shape)')
    reference=torch.as_tensor(reference,dtype=torch.bool,device=delta.device)
    if reference.shape!=(len(delta),) or not bool(reference.any()):
        raise ValueError('Require a nonempty reference window on the explicit year axis')
    return mean[None]+delta-delta[reference].mean(0,keepdim=True)


class MeanDeltaDESK(MultiStreamAutoencoder):
    """Zmean(reference levels) + gated low-rank response to covariate deltas.

    No year or coordinate concatenation. Reference tensors and optional reusable
    mean states are explicit arguments, safe for checkpoint recomputation. A
    reference state should be computed once per window when using dropout.
    The current prototype needs isolated trainer/exporter integration before a
    full DESK run; it is not a trained or production-compatible checkpoint.
    """
    def __init__(self,*args,change_rank=8,context_width=64,**kwargs):
        super().__init__(*args,**kwargs)
        width=sum(self.dims);out=self.mixer[-1].out_features+self.private_dim
        if not 0<change_rank<=out:raise ValueError('Invalid change rank')
        self.delta_projection=nn.Linear(width,change_rank,bias=False)
        self.context_gate=nn.Sequential(nn.Linear(width,context_width),nn.GELU(),
                                        nn.Linear(context_width,change_rank),nn.Tanh())
        self.change_output=nn.Linear(change_rank,out,bias=False)
        nn.init.zeros_(self.change_output.weight)

    def mean_state(self,reference_covariates,mask):
        outputs=super().forward(reference_covariates,mask,with_private=self.private_dim>0)
        return (outputs[0],outputs[2] if self.private_dim>0 else None)

    def forward(self,x,mask,with_private=False,*,reference_covariates,reference_state=None,
                return_components=False):
        if x.shape!=reference_covariates.shape or x.ndim!=4:
            raise ValueError('Current and reference covariates must have identical grid shapes')
        if with_private and self.private_dim<=0:raise ValueError('Private coordinates are not configured')
        mean,private_mean=(self.mean_state(reference_covariates,mask) if reference_state is None else reference_state)
        delta=x-reference_covariates
        code=self.delta_projection(delta)*self.context_gate(reference_covariates)
        deviation=self.change_output(code)
        d=mean.shape[-1];z=mean+deviation[...,:d]
        recon=self.decoder(z.reshape(-1,d)).reshape(*x.shape)
        if return_components:
            combined_mean=mean if self.private_dim<=0 else torch.cat([mean,private_mean],dim=-1)
            return combined_mean,deviation
        if with_private:return z,recon,private_mean+deviation[...,d:]
        return z,recon
