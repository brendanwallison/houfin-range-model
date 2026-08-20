"""Multi-epoch direction panel: the pure cores.

The panel's whole purpose is to decide a 0.03 difference that 36 cells could not, so a quiet
error in WHICH cell-year each side reads would be invisible in the output and would corrupt the
verdict. These pin the selection rules.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.community_encoder.train_DESK.validate_baselines import (
    DEFAULT_EPOCHS, _idw_at, nearest_survey)



def _np_or(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def test_each_cell_uses_its_own_nearest_actual_survey():
    """No averaging and no interpolation in time: the model is read at the same REAL year the
    survey happened, so a cell surveyed in 1986 is compared at 1986, not at 1985."""
    pip = np.array([[0, 0, 1986], [0, 0, 1991], [1, 1, 1984], [2, 2, 1999]], dtype=np.int32)
    near = nearest_survey(pip, None, (1985,), tol=2)
    assert near[1985] == {(0, 0): 1986, (1, 1): 1984}          # 1999 is outside +/-2
    assert (2, 2) not in near[1985]


def test_ties_break_to_the_earlier_year_not_to_row_order():
    """1983 and 1987 are both 2 from 1985. Breaking by whichever row came first would make the
    panel depend on the point set's ordering."""
    a = nearest_survey(np.array([[0, 0, 1983], [0, 0, 1987]], np.int32), None, (1985,), 2)
    b = nearest_survey(np.array([[0, 0, 1987], [0, 0, 1983]], np.int32), None, (1985,), 2)
    assert a[1985][(0, 0)] == b[1985][(0, 0)] == 1983


def test_tolerance_is_respected_exactly_at_the_boundary():
    pip = np.array([[0, 0, 1967], [1, 1, 1969], [2, 2, 1970]], dtype=np.int32)
    near = nearest_survey(pip, None, (1967,), tol=2)
    assert set(near[1967]) == {(0, 0), (1, 1)}                 # 1970 is 3 away
    assert nearest_survey(pip, None, (1967,), tol=3)[1967][(2, 2)] == 1970


def test_unsupervised_rows_never_supply_an_epoch():
    """Duplicate cell-years exist for the ESK basis only. If one supplied the epoch value the
    same cell-year could enter twice, or a non-target row could stand in for a target."""
    pip = np.array([[0, 0, 1985], [1, 1, 1985]], dtype=np.int32)
    near = nearest_survey(pip, np.array([True, False]), (1985,), 2)
    assert set(near[1985]) == {(0, 0)}


def test_the_default_epochs_are_spaced_for_the_learned_ema():
    """~19 years is about two output-EMA half-lives at the learned 10.5 y -- the shortest
    interval over which the EMA is not dominating the predicted change."""
    gaps = np.diff(DEFAULT_EPOCHS)
    assert gaps.min() >= 18 and gaps.max() <= 20, gaps


def test_idw_reads_each_training_cell_at_its_own_epoch_year():
    """The bar must be built the same way the model side is, or the comparison is rigged: a
    training cell contributes its value at ITS nearest year to the epoch."""
    train = {(0, c): (1984 if c % 2 == 0 else 1986) for c in range(10)}
    seen = []

    def z_of(cell, year):
        seen.append((cell, year))
        return np.array([float(year), 1.0])

    out = _idw_at([(5, 5)], train, z_of, k=8)
    assert out.shape == (1, 2)
    assert set(seen) == {(c, y) for c, y in train.items()}
    assert 1984.0 <= out[0, 0] <= 1986.0


def test_idw_declines_when_there_are_too_few_training_cells():
    """Fewer than k neighbours would silently reuse the same cell k times and read as a strong
    baseline built from one point."""
    assert _idw_at([(0, 0)], {(1, 1): 2000, (2, 2): 2000}, lambda c, y: np.zeros(2), k=8) is None
    assert _idw_at([], {(i, i): 2000 for i in range(20)}, lambda c, y: np.zeros(2)) is None


def test_a_constant_field_interpolates_to_that_constant():
    out = _idw_at([(4, 4)], {(i, 0): 2000 for i in range(12)},
                  lambda c, y: np.array([3.0, -1.0]), k=8)
    assert np.allclose(out[0], [3.0, -1.0])


def test_spatial_interp_baseline():
    """The ceiling for a blocked holdout: interpolate the targets, no covariates, no learning."""
    from src.community_encoder.train_DESK.augment import blocked_holdout
    from src.community_encoder.train_DESK.validate_baselines import spatial_interp_baseline

    H, W, L = 60, 80, 16
    valid = np.zeros((H, W), bool); valid[2:58, 3:77] = True
    val, buf = blocked_holdout(valid, block_cells=12, holdout_frac=0.2, buffer_cells=2, seed=0)
    tr = valid & ~val & ~buf
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(0)

    # A smooth field is nearly interpolable, so the baseline must score LOW -- meaning a
    # trained model has to beat a low number to have earned anything from the covariates.
    field = np.stack([np.sin((yy + 3 * d) / 12.0) + np.cos((xx - d) / 15.0)
                      for d in range(L)], -1).astype("float32")
    smooth = {y: (field, torch.tensor(tr), torch.tensor(val)) for y in (1966, 2025)}
    n_s, i_s = spatial_interp_baseline(smooth)
    assert i_s <= n_s, "inverse-distance over 8 neighbours must beat single-nearest"
    # Judge against the field's OWN predict-mean error rather than an arbitrary constant:
    # interpolating a smooth field should remove the large majority of that variance.
    held = field[val]
    predict_mean = float(((held - held.mean(0)) ** 2).sum(-1).mean())
    assert n_s < 0.25 * predict_mean, (n_s, predict_mean)

    # Pure noise is not interpolable at all: nearest-neighbour error approaches 2*L (two
    # independent unit-variance draws per latent dim), and IDW shrinks toward the mean.
    noise = {y: (rng.normal(size=(H, W, L)).astype("float32"),
                 torch.tensor(tr), torch.tensor(val)) for y in (1966, 2025)}
    n_n, i_n = spatial_interp_baseline(noise)
    assert n_n > L, (n_n, L)
    assert i_n < n_n
    assert n_s < n_n / 10, (n_s, n_n)         # smooth vs noise must be worlds apart

    # degenerate inputs return NaN rather than raising inside a training run
    assert all(np.isnan(v) for v in spatial_interp_baseline({}))

    # PER-YEAR masks with zero-filled absent cells. _prepare_trend_targets builds each year as
    # np.zeros and marks only that year's points, so a cell absent in year Y holds 0.0. A
    # baseline that reused one year's masks would score interpolation against those zeros and
    # feed them in as sources -- a different population and denominator from _z_mse. Here year
    # 2025 covers everything while 1966 covers only the top half; the answer must depend only
    # on genuinely-present cells, so it must equal the same call with the zeros left untouched
    # outside each year's own mask.
    # Coverage must SHRINK from the first year onward, otherwise reusing the first year's mask
    # would coincidentally stay inside every later year's mask and the bug would hide.
    half = np.zeros((H, W), bool); half[:H // 2, :] = True
    f2 = field.copy()
    varying = {}
    for y, cov_mask in ((1966, np.ones((H, W), bool)), (2025, half)):
        z = np.where(cov_mask[..., None], f2, 0.0).astype("float32")   # absent -> 0.0
        varying[y] = (z, torch.tensor(tr & cov_mask), torch.tensor(val & cov_mask))
    n_v, i_v = spatial_interp_baseline(varying)
    assert np.isfinite(n_v) and np.isfinite(i_v)
    assert i_v <= n_v
    # The zero-fill must not leak in: poisoning the absent region with a huge value cannot
    # change the result, because those cells are outside every year's mask.
    poisoned = {}
    for y, (z, t_m, v_m) in varying.items():
        zp = z.copy()
        outside = ~(_np_or(t_m) | _np_or(v_m))
        zp[outside] = 1e3
        poisoned[y] = (zp, t_m, v_m)
    n_p, i_p = spatial_interp_baseline(poisoned)
    assert abs(n_p - n_v) < 1e-6 and abs(i_p - i_v) < 1e-6, (n_v, n_p, i_v, i_p)
    print("spatial interpolation baseline OK")


def test_the_direction_baseline_scores_high_when_change_is_smooth_in_space():
    """A spatially smooth change field is exactly what interpolation reproduces, so the
    baseline must score near 1 there -- otherwise it is too weak to be a fair bar and would
    flatter the model."""
    import torch
    from src.community_encoder.train_DESK.validate_baselines import spatial_interp_dir_cos
    H, W, L = 12, 12, 3
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=(H, W, L)).astype("float32") * 0.01
    yy, xx = np.mgrid[0:H, 0:W]
    delta = np.stack([yy / H, xx / W, (yy + xx) / (H + W)], -1).astype("float32")
    tr = np.zeros((H, W), bool); tr[::2, ::2] = True
    va = np.zeros((H, W), bool); va[1::4, 1::4] = True
    tgt = {1966: (torch.tensor(z0), torch.tensor(tr), torch.tensor(va), None),
           2025: (torch.tensor(z0 + delta), torch.tensor(tr), torch.tensor(va), None)}
    dc, n = spatial_interp_dir_cos(tgt, 1966, 2025, torch.tensor(va))
    assert n == int(va.sum()) and dc > 0.95, (dc, n)


def test_the_direction_baseline_reports_nan_when_it_cannot_run():
    import torch
    from src.community_encoder.train_DESK.validate_baselines import spatial_interp_dir_cos
    z = torch.zeros(4, 4, 2); m = torch.zeros(4, 4, dtype=torch.bool)
    tgt = {2025: (z, m, m, None)}
    dc, n = spatial_interp_dir_cos(tgt, None, 2025, m)
    assert np.isnan(dc) and n == 0

