"""DESK with explicit space-time coordinate features (in-era interpolation comparison to the z_assim field).

Random Fourier features of (x / spatial scale, y / spatial scale, clamp(year) / temporal scale) enter the
pointwise hidden layer through one zero-initialised linear map, so at initialisation the model is the
covariate-only DESK. Year is clamped to [year_min, year_max] (the observed summer era): before year_min
the coordinate term is constant rather than periodic. Features are computed from the year passed to each
forward call, so checkpointed recomputation reproduces them.
"""
import numpy as np
import torch
from torch import nn

from src.community_encoder.train_DESK.model_arch import MultiStreamAutoencoder


def grid_xy_km(shape):
    """Projected cell centres (km) for the full reference grid, (H, W, 2)."""
    from src.config_utils import load_data_config
    import rasterio
    with rasterio.open(load_data_config()['grid']['ref_raster']) as ds:
        if (ds.height, ds.width) != tuple(shape):
            raise ValueError(f'Reference raster {ds.height}x{ds.width} != covariate grid {shape}')
        rows, cols = np.meshgrid(np.arange(ds.height), np.arange(ds.width), indexing='ij')
        x, y = rasterio.transform.xy(ds.transform, rows.ravel(), cols.ravel())
    return (np.column_stack([x, y]) / 1000.).reshape(*shape, 2).astype('float32')


class CoordDESK(MultiStreamAutoencoder):
    def __init__(self, *args, coord_spec, grid_xy, **kwargs):
        super().__init__(*args, **kwargs)
        g = np.random.default_rng(int(coord_spec['seed']))
        n = int(coord_spec['features'])
        omega = g.normal(size=(3, n)) / np.array([coord_spec['scale_km'], coord_spec['scale_km'],
                                                   coord_spec['scale_years']])[:, None]
        self.register_buffer('xy', torch.as_tensor(grid_xy, dtype=torch.float32))
        self.register_buffer('omega', torch.as_tensor(omega, dtype=torch.float32))
        self.register_buffer('phase', torch.as_tensor(g.uniform(0, 2 * np.pi, n), dtype=torch.float32))
        self.year_range = (float(coord_spec['year_min']), float(coord_spec['year_max']))
        self.coord_proj = nn.Linear(n, self.mixer[0].in_features, bias=False)
        nn.init.zeros_(self.coord_proj.weight)
        self._coord = None

    def features(self, year):
        y = float(min(max(float(year), self.year_range[0]), self.year_range[1])) - 2015.5
        phase = self.xy @ self.omega[:2] + y * self.omega[2] + self.phase
        return torch.cos(phase) * np.sqrt(2. / self.omega.shape[1])

    def _pointwise_latent(self, x):
        parts = torch.split(x, self.dims, dim=1)
        h = torch.cat([enc(p) for enc, p in zip(self.encoders, parts)], dim=1)
        h = h + self.coord_proj(self._coord.to(h.dtype))
        if getattr(self, '_want_private', False):
            h1 = self.mixer[1](self.mixer[0](h))
            return self.mixer[2](h1), self.private_out(h1)
        return self.mixer(h)

    def forward(self, x, mask, with_private=False, *, year):
        B, H, W, _ = x.shape
        if (H, W) != tuple(self.xy.shape[:2]):
            raise ValueError('Coordinate features need the full grid (no spatial tiles)')
        self._coord = self.features(year).reshape(1, H * W, -1).expand(B, -1, -1).reshape(B * H * W, -1)
        try:
            return super().forward(x, mask, with_private)
        finally:
            self._coord = None
