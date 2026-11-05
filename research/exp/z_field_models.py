"""DESK variants that carry the z_assim field's own regional space-time basis (comparable-conditions test).

The basis is exactly z_assim's R term: S(s) = [1, Gaussian RBF at 128 coverage centres, 200 km] and T(t) = cubic
B-spline, 10-yr knots on 1902-2025, centred on its 2006-2025 mean (research/exp/z_repair_models.py). Centres come
from the same observation cells as field_d_interp (prepared by z_field_run.py).

    field_output : Z_summer += (S(s) C T(t)) L, added AFTER the spatial residual and the decoder, so it does not
                   enter reconstruction (as in the field, which has no decoder). C (129 x 16 x rank) starts at zero.
    basis_input  : the same S(s) (x) T(t) features enter the pointwise hidden layer through one zero-initialised
                   linear map (W: hidden x 129 x 16), i.e. the field basis in the coordinate-input position.
    rff          : the original CoordDESK (random Fourier features), unchanged.

Parameters named field_* (and the rff coord_proj) are trained in their own optimizer group at the field's learning
rate with no weight decay; field_penalty() mirrors z_assim's ridge + second-difference (time) penalties.
"""
import numpy as np
import torch
from torch import nn

from src.community_encoder.train_DESK.model_arch import MultiStreamAutoencoder

YEAR0 = 1902


def load_basis(spec, grid_xy):
    import hashlib
    from pathlib import Path
    path = Path(spec['basis_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['basis_sha256']:
        raise ValueError('Field basis inputs changed')
    b = np.load(path)
    xy = np.asarray(grid_xy, float).reshape(-1, 2)
    d2 = ((xy[:, None, :] - b['centers'][None]) ** 2).sum(-1)
    S = np.concatenate([np.ones((len(xy), 1)), np.exp(-.5 * d2 / float(b['scale_km']) ** 2)], 1)
    return S.astype('float32'), b['T'].astype('float32')


class _YearConditioned(MultiStreamAutoencoder):
    def __init__(self, *args, field_spec, grid_xy, **kwargs):
        super().__init__(*args, **kwargs)
        S, T = load_basis(field_spec, grid_xy)
        self.grid_shape = tuple(np.asarray(grid_xy).shape[:2])
        self.register_buffer('field_S', torch.as_tensor(S))
        self.register_buffer('field_T', torch.as_tensor(T))
        self.spec = field_spec
        self._year = None

    def t_row(self, year):
        i = int(round(float(year))) - YEAR0
        if not 0 <= i < self.field_T.shape[0]:
            raise ValueError(f'Year {year} outside the field time basis')
        return self.field_T[i]

    def _check(self, x):
        if tuple(x.shape[1:3]) != self.grid_shape:
            raise ValueError('Field basis needs the full grid (no spatial tiles)')


class FieldOutputDESK(_YearConditioned):
    def __init__(self, *args, field_spec, grid_xy, **kwargs):
        super().__init__(*args, field_spec=field_spec, grid_xy=grid_xy, **kwargs)
        P, Q, r = self.field_S.shape[1], self.field_T.shape[1], int(field_spec['rank'])
        g = torch.Generator().manual_seed(int(field_spec['seed']))
        self.field_C = nn.Parameter(torch.zeros(P, Q, r))
        self.field_L = nn.Parameter(torch.randn(r, self.mixer[-1].out_features, generator=g) * .1)

    def field(self, year):
        M = torch.einsum('pqr,q->pr', self.field_C, self.t_row(year))
        return (self.field_S @ M) @ self.field_L                     # (H*W, latent)

    def field_penalty(self):
        s = self.spec
        p = s['lam_C'] * self.field_C.square().mean()
        p = p + s['lam_curv'] * torch.diff(self.field_C, n=2, dim=1).square().mean()
        return p + 1e-3 * (self.field_L.square().mean() - 1.).square()

    def forward(self, x, mask, with_private=False, *, year):
        self._check(x)
        out = super().forward(x, mask, with_private)
        B, H, W, _ = x.shape
        z = out[0] + self.field(year).reshape(1, H, W, -1).to(out[0].dtype)
        return (z,) + tuple(out[1:])


class BasisInputDESK(_YearConditioned):
    def __init__(self, *args, field_spec, grid_xy, **kwargs):
        super().__init__(*args, field_spec=field_spec, grid_xy=grid_xy, **kwargs)
        P, Q = self.field_S.shape[1], self.field_T.shape[1]
        self.field_W = nn.Parameter(torch.zeros(self.mixer[0].in_features, P, Q))

    def field_penalty(self):
        s = self.spec
        return (s['lam_C'] * self.field_W.square().mean()
                + s['lam_curv'] * torch.diff(self.field_W, n=2, dim=2).square().mean())

    def _pointwise_latent(self, x):
        parts = torch.split(x, self.dims, dim=1)
        h = torch.cat([enc(p) for enc, p in zip(self.encoders, parts)], dim=1)
        Wt = torch.einsum('kpq,q->kp', self.field_W, self.t_row(self._year))
        add = self.field_S @ Wt.T                                     # (H*W, hidden)
        h = h + add.repeat(h.shape[0] // add.shape[0], 1).to(h.dtype)
        if getattr(self, '_want_private', False):
            h1 = self.mixer[1](self.mixer[0](h))
            return self.mixer[2](h1), self.private_out(h1)
        return self.mixer(h)

    def forward(self, x, mask, with_private=False, *, year):
        self._check(x)
        self._year = year
        try:
            return super().forward(x, mask, with_private)
        finally:
            self._year = None


def make_model(spec, grid_xy, *args, **kwargs):
    kind = spec['kind']
    if kind == 'rff':
        from research.exp.z_coord_models import CoordDESK
        return CoordDESK(*args, coord_spec=spec, grid_xy=grid_xy, **kwargs)
    cls = {'field_output': FieldOutputDESK, 'basis_input': BasisInputDESK}[kind]
    return cls(*args, field_spec=spec, grid_xy=grid_xy, **kwargs)


def is_fast(name):
    return name.startswith('field_C') or name.startswith('field_L') or name.startswith('field_W') or name.startswith('coord_proj')
