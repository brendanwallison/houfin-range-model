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



# --------- the ladder: each rung isolates one claim ---------

def _grid_points(cells, years):
    pidx = np.array([[r, c, y] for (r, c) in cells for y in years], dtype=np.int32)
    return pidx


def test_cell_trend_is_exact_on_a_linear_in_time_field():
    """The rung that survives a temporal block holdout, so the one the 1900 claim rests on.
    On a field that really is linear in time per cell, extrapolating a line must be exact --
    including BACKWARD, outside the range of the training years."""
    from src.community_encoder.train_DESK.validate_baselines import cell_temporal_baseline
    years = [1970, 1985, 2000, 2015]
    pidx = _grid_points([(0, 0), (0, 1)], years)
    z = np.stack([[float(y) * 0.5, -float(y)] for _r, _c, y in pidx]).astype("float32")
    ho = np.zeros((2, 2), bool)
    target = pidx[:, 2] == 1970                     # earliest -> genuine backward extrapolation
    err = cell_temporal_baseline(pidx, z, ho, target, mode="trend")
    assert np.isfinite(err).all() and np.abs(err).max() < 1e-3, err


def test_cell_trend_is_nan_with_only_one_other_year():
    """A line needs two points. Returning something anyway would put a fake bar in the table."""
    from src.community_encoder.train_DESK.validate_baselines import cell_temporal_baseline
    pidx = np.array([[0, 0, 1970], [0, 0, 2020]], dtype=np.int32)
    z = np.ones((2, 3), "float32")
    err = cell_temporal_baseline(pidx, z, np.zeros((1, 1), bool), pidx[:, 2] == 1970,
                                mode="trend")
    assert np.isnan(err).all()                      # one other year -> no slope


def test_a_point_is_never_its_own_baseline():
    """A target row must be excluded from its own cell's source rows, or the bar reads 0 and
    beats every model by construction."""
    from src.community_encoder.train_DESK.validate_baselines import cell_temporal_baseline
    pidx = _grid_points([(0, 0)], [1970, 1990, 2010])
    z = np.stack([[float(y), 0.0] for _r, _c, y in pidx]).astype("float32")
    err = cell_temporal_baseline(pidx, z, np.zeros((1, 1), bool), pidx[:, 2] == 1990,
                                 mode="nearest")
    assert err[0] > 0.0, "the 1990 point was handed its own value"


def test_borrowed_delta_is_exact_when_every_cell_changed_alike():
    """'It changed like its neighbours' must be exact when that is literally true -- otherwise
    it is too weak to be the competitor to DESK's claim."""
    from src.community_encoder.train_DESK.validate_baselines import borrowed_delta_baseline
    cells = [(r, c) for r in range(5) for c in range(5)]
    pidx = _grid_points(cells, [1980, 2025])
    base = {cell: np.array([r * 1.0, c * 1.0], "float32") for cell, (r, c) in zip(cells, cells)}
    shift = np.array([3.0, -2.0], "float32")        # same delta everywhere
    z = np.stack([base[(int(r), int(c))] + (0.0 if y == 2025 else -shift)
                  for r, c, y in pidx]).astype("float32")
    ho = np.zeros((5, 5), bool); ho[2, 2] = True
    target = (pidx[:, 0] == 2) & (pidx[:, 1] == 2) & (pidx[:, 2] == 1980)
    err = borrowed_delta_baseline(pidx, z, ho, target, recent_year=2025)
    assert np.isfinite(err).any() and np.nanmax(err) < 1e-4, err


def test_borrowed_delta_is_nan_when_the_year_has_no_training_neighbours():
    """Exactly what a temporal block holdout does. It must report n/a, not a number."""
    from src.community_encoder.train_DESK.validate_baselines import borrowed_delta_baseline
    cells = [(0, c) for c in range(12)]
    pidx = _grid_points(cells, [1980, 2025])
    z = np.ones((len(pidx), 2), "float32")
    ho = np.zeros((1, 12), bool)
    ho[0, :] = True                                  # every 1980 cell held out -> no sources
    err = borrowed_delta_baseline(pidx, z, ho, pidx[:, 2] == 1980, recent_year=2025)
    assert np.isnan(err).all()


def _sparse_layout(n_cells=12, spacing=4, years=range(1960, 2040, 10)):
    """Cells spaced far apart in space, several years deep.

    Spacing matters: on a DENSE grid the same-year neighbours are the nearest ones at every
    ratio, so nothing discriminates and the sweep just returns the first candidate. The ratio
    only has meaning when crossing a year can actually cost less than crossing space.
    """
    cells = [(0, c * spacing) for c in range(n_cells)]
    return _grid_points(cells, list(years))


def test_spacetime_ratio_cv_finds_a_time_dominated_field():
    """A field that varies with YEAR and not position. Borrowing across years is what hurts
    here, so the sweep must pick a LARGE ratio -- making a year expensive to cross, hence
    neighbours same-year."""
    from src.community_encoder.train_DESK.validate_baselines import spacetime_idw_baseline
    pidx = _sparse_layout()
    z = np.stack([[float(y) * 0.1, 0.0] for _r, _c, y in pidx]).astype("float32")
    _err, ratio = spacetime_idw_baseline(pidx, z, np.zeros((1, 64), bool),
                                         np.ones(len(pidx), bool), verbose=False)
    assert ratio >= 2.0, ratio


def test_spacetime_ratio_cv_finds_a_space_dominated_field():
    """The converse, on the same layout: a field varying with position and not year must pick a
    SMALL ratio, so a cell's own other years are its nearest neighbours."""
    from src.community_encoder.train_DESK.validate_baselines import spacetime_idw_baseline
    pidx = _sparse_layout()
    z = np.stack([[float(c), 0.0] for _r, c, _y in pidx]).astype("float32")
    _err, ratio = spacetime_idw_baseline(pidx, z, np.zeros((1, 64), bool),
                                         np.ones(len(pidx), bool), verbose=False)
    assert ratio <= 0.25, ratio


def test_the_panel_marks_unavailable_bars_nan_rather_than_inventing_a_number():
    """The whole point of the n/a column. With every historical cell held out there are no
    training sources in that year, so borrowed_delta cannot run -- and must say so."""
    from src.community_encoder.train_DESK.validate_baselines import baseline_panel
    cells = [(0, c) for c in range(12)]
    pidx = _grid_points(cells, [1980, 2025])
    z = np.random.default_rng(0).normal(size=(len(pidx), 3)).astype("float32")
    ho = np.zeros((1, 12), bool); ho[0, :] = True
    out = baseline_panel(pidx, z, z.copy(), ho, recent_year=2025, verbose=False)
    assert np.isnan(out["by_era"]["1980s"]["borrowed_delta"]["desk_beats_frac"])


def test_the_panel_scores_a_perfect_model_as_beating_every_available_bar():
    """Sanity on the direction of the comparison: if z_desk IS z_obs, DESK's error is 0 and it
    must beat every bar that can run. A flipped inequality would read as total failure."""
    from src.community_encoder.train_DESK.validate_baselines import baseline_panel
    cells = [(r, c) for r in range(6) for c in range(6)]
    pidx = _grid_points(cells, [1980, 2000, 2025])
    rng = np.random.default_rng(1)
    z = rng.normal(size=(len(pidx), 3)).astype("float32")
    ho = np.zeros((6, 6), bool); ho[::3, ::3] = True
    out = baseline_panel(pidx, z, z.copy(), ho, recent_year=2025, verbose=False)
    for name, r in out["overall"].items():
        if isinstance(r, dict) and np.isfinite(r["desk_beats_frac"]):
            assert r["desk_beats_frac"] == 1.0, (name, r)


def test_the_two_holdouts_have_COMPLEMENTARY_baseline_sets():
    """The argument for running both. Under a SPATIAL holdout a cell is out in every year, so it
    has no training years of its own and the per-cell temporal rungs cannot run. Under a
    TEMPORAL holdout there are no training points in the withheld years, so borrowed-delta
    cannot run. Neither holdout alone exercises the whole ladder."""
    from src.community_encoder.train_DESK.validate_baselines import baseline_panel
    cells = [(r, c) for r in range(10) for c in range(10)]
    years = [1970, 1980, 1990, 2000, 2025]
    pidx = _grid_points(cells, years)
    rng = np.random.default_rng(0)
    z = rng.normal(size=(len(pidx), 3)).astype("float32")
    ho = np.zeros((10, 10), bool); ho[::3, ::3] = True

    spatial = baseline_panel(pidx, z, z + 0.1, ho, 2025, verbose=False)["overall"]
    assert not np.isfinite(spatial["cell_trend"]["desk_beats_frac"])       # cannot run
    assert np.isfinite(spatial["borrowed_delta"]["desk_beats_frac"])       # can

    is_ho = ho[pidx[:, 0], pidx[:, 1]]
    withheld = [1970, 1980]
    temporal = baseline_panel(pidx, z, z + 0.1, ho, 2025, verbose=False,
                              target_rows=np.isin(pidx[:, 2], withheld) & ~is_ho,
                              exclude_years=withheld)["overall"]
    assert np.isfinite(temporal["cell_trend"]["desk_beats_frac"])          # now it can
    assert not np.isfinite(temporal["borrowed_delta"]["desk_beats_frac"])  # and this cannot


def test_no_rung_may_source_from_a_withheld_year():
    """A baseline reading a withheld year gets information the model never saw and stops being a
    fair bar. cell_trend on a linear-in-time field would be EXACT if it could see the withheld
    endpoints; excluded, it must extrapolate and so cannot be exact for free."""
    from src.community_encoder.train_DESK.validate_baselines import cell_temporal_baseline
    pidx = _grid_points([(0, 0)], [1970, 1975, 1990, 2000, 2010, 2025])
    z = np.stack([[float(y), 0.0] for _r, _c, y in pidx]).astype("float32")
    withheld = [1970, 1975]
    target = np.isin(pidx[:, 2], withheld)
    leaky = cell_temporal_baseline(pidx, z, np.zeros((1, 1), bool), target, mode="nearest")
    fair = cell_temporal_baseline(pidx, z, np.zeros((1, 1), bool), target, mode="nearest",
                                 exclude_years=withheld)
    # leaky can reach the OTHER withheld year (1975 for 1970, 5 units away); fair must reach
    # 1990 at minimum, so its error is strictly larger
    assert np.nanmax(fair) > np.nanmax(leaky), (leaky, fair)
