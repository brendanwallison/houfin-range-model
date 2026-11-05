"""Nested, community-level candidate modules; no focal-species readout parameters.

Research modules only. Time is an explicit forward argument so checkpointed
recomputation cannot accidentally read a later iteration's mutable year context.
"""
import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import nn
from torch.nn import functional as F

from src.community_encoder.train_DESK.model_arch import MultiStreamAutoencoder


class CubicTimeBasis:
    def __init__(self, start=1902, end=2025, knot_spacing=20, reference=(2006,2025)):
        if not start <= reference[0] <= reference[1] <= end:
            raise ValueError('Reference must be inside the modeled domain')
        self.start, self.end = start, end
        self.knots = np.r_[[start]*4, np.arange(start+knot_spacing,end,knot_spacing), [end]*4]
        self.reference = tuple(reference)
        self.center = self._raw(np.arange(reference[0],reference[1]+1)).mean(0)

    def _raw(self, years):
        y=np.asarray(years, float)
        if np.any((y < self.start) | (y > self.end)):
            raise ValueError('Time extrapolation outside the declared domain is unsupported')
        return BSpline.design_matrix(y.reshape(-1),self.knots,3,extrapolate=False).toarray().reshape(*y.shape,-1)

    def __call__(self, years):
        return self._raw(years)-self.center

    @property
    def width(self):
        return len(self.center)


class GaussianRegionalBasis:
    """Deterministic coverage-only centers; projected coordinates are in kilometers."""
    def __init__(self, training_xy, n_centers=64, scale_km=300.):
        xy=np.unique(np.asarray(training_xy,float),axis=0)
        if not len(xy) or not np.isfinite(xy).all() or scale_km <= 0:
            raise ValueError('Invalid spatial training support')
        selected=[int(np.argmin(np.sum((xy-xy.mean(0))**2,axis=1)))]
        distances=np.sum((xy-xy[selected[0]])**2,axis=1)
        for _ in range(1,min(n_centers,len(xy))):
            j=int(np.argmax(distances));selected.append(j)
            distances=np.minimum(distances,np.sum((xy-xy[j])**2,axis=1))
        self.centers=xy[selected]
        self.scale_km=float(scale_km)

    def __call__(self, xy):
        xy=np.asarray(xy,float)
        d2=np.sum((xy[...,None,:]-self.centers)**2,axis=-1)
        return np.concatenate([np.ones((*xy.shape[:-1],1)),np.exp(-.5*d2/self.scale_km**2)],axis=-1)

    @property
    def width(self):
        return len(self.centers)+1


class LowRankCommunityResidual(nn.Module):
    def __init__(self, latent_dim, rank, spatial_width, temporal_width):
        super().__init__()
        if not 0 < rank <= latent_dim:
            raise ValueError('Invalid residual rank')
        self.directions=nn.Parameter(torch.randn(latent_dim,rank)/np.sqrt(latent_dim))
        self.scores=nn.Parameter(torch.zeros(spatial_width,temporal_width,rank))

    def forward(self, base_z, spatial_basis, temporal_basis):
        u=torch.linalg.qr(self.directions,mode='reduced').Q
        a=torch.einsum('...p,...q,pqr->...r',spatial_basis,temporal_basis,self.scores)
        return base_z+a@u.T

    def penalty(self, temporal_grid):
        # Unit-normalized U fixes the factorization scale. Penalize amplitude and
        # actual annual score curvature, not arbitrary coordinate differences.
        trajectories=torch.einsum('tq,pqr->ptr',temporal_grid,self.scores)
        return self.scores.square().mean(),torch.diff(trajectories,n=2,dim=1).square().mean()


class LowRankTimeModulation(nn.Module):
    def __init__(self, width, rank, temporal_width):
        super().__init__()
        self.output_directions=nn.Parameter(torch.randn(width,rank)/np.sqrt(width))
        self.input_directions=nn.Parameter(torch.randn(rank,width)/np.sqrt(width))
        self.time_weights=nn.Parameter(torch.zeros(temporal_width,rank))

    def forward(self, hidden, temporal_basis):
        u=torch.linalg.qr(self.output_directions,mode='reduced').Q
        v=F.normalize(self.input_directions,dim=1)
        gates=temporal_basis@self.time_weights
        return hidden+((hidden@v.T)*gates)@u.T

    def penalty(self, temporal_grid):
        gates=temporal_grid@self.time_weights
        return self.time_weights.square().mean(),torch.diff(gates,n=2,dim=0).square().mean()


class TimeConditionedDESK(MultiStreamAutoencoder):
    """Modulate the mixer's late hidden response; retain the established architecture."""
    def __init__(self, *args, time_rank=2, temporal_width, **kwargs):
        super().__init__(*args,**kwargs)
        self.time_modulation=LowRankTimeModulation(self.mixer[0].out_features,time_rank,temporal_width)

    def forward(self,x,mask,with_private=False,*,temporal_basis):
        if x.dim()!=4:
            raise ValueError('Expected (batch,height,width,channels)')
        B,H,W,C=x.shape
        t=temporal_basis
        if t.ndim==1:
            t=t[None].expand(B,-1)
        if t.shape[0]!=B:
            raise ValueError('One explicit time basis per grid in the batch is required')
        flat=x.reshape(B*H*W,C)
        parts=torch.split(flat,self.dims,dim=1)
        h=torch.cat([enc(p) for enc,p in zip(self.encoders,parts)],dim=1)
        h=self.mixer[1](self.mixer[0](h))
        h=self.time_modulation(h,t.repeat_interleave(H*W,dim=0))
        z_grid=self.mixer[2](h).reshape(B,H,W,-1).permute(0,3,1,2)
        if self.spatial_kernel>0:
            residual,_=self.spatial(z_grid,mask.reshape(B,1,H,W).to(z_grid.dtype))
            z_grid=z_grid+self.gamma*self.spatial_mix(F.gelu(residual))
        z=z_grid.permute(0,2,3,1).contiguous()
        recon=self.decoder(z.reshape(B*H*W,-1)).reshape(B,H,W,C)
        if with_private:
            if self.private_dim<=0:
                raise ValueError('Private coordinates are not configured')
            return z,recon,self.private_out(h).reshape(B,H,W,self.private_dim)
        return z,recon
