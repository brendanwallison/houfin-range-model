"""Multi-epoch direction panel: the pure cores.

The panel's whole purpose is to decide a 0.03 difference that 36 cells could not, so a quiet
error in WHICH cell-year each side reads would be invisible in the output and would corrupt the
verdict. These pin the selection rules.
"""
import os
import sys

import numpy as np
import pytest
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
    assert np.isnan(out["by_era"]["1980s"]["win_rate_vs"]["borrowed_delta"])


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
    # DESK is a row in the same table now, so this reads as "every OTHER predictor loses to the
    # null less often than DESK does" only if we ask it against DESK -- ask it directly instead.
    ov = baseline_panel(pidx, z, z, ho, 2025, verbose=False)
    for name, w in ov["overall"]["win_rate_vs"].items():
        if name != "desk" and np.isfinite(w):
            assert w <= ov["overall"]["win_rate_vs"]["desk"], (name, w)
    assert ov["overall"]["win_rate_vs"]["desk"] == 1.0     # a perfect model never loses


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
    assert not np.isfinite(spatial["win_rate_vs"]["cell_trend"])           # cannot run
    assert np.isfinite(spatial["win_rate_vs"]["borrowed_delta"])           # can

    is_ho = ho[pidx[:, 0], pidx[:, 1]]
    withheld = [1970, 1980]
    temporal = baseline_panel(pidx, z, z + 0.1, ho, 2025, verbose=False,
                              target_rows=np.isin(pidx[:, 2], withheld) & ~is_ho,
                              exclude_years=withheld)["overall"]
    assert np.isfinite(temporal["win_rate_vs"]["cell_trend"])              # now it can
    assert not np.isfinite(temporal["win_rate_vs"]["borrowed_delta"])      # and this cannot


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


def test_a_cell_surveyed_near_only_one_epoch_does_not_crash_the_pair():
    """BBS routes come and go: a cell surveyed near 1985 but never near 2005 belongs to
    val_of[1985] and not val_of[2005]. The pair's cell set has to intersect the two epochs
    BEFORE looking up either year, or the per-cell year lookup indexes an epoch that cell was
    never surveyed near -- the KeyError that killed a full validate run."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    epochs = (1985, 2005)
    rows = []
    for i in range(20):                    # 20 cells present in BOTH epochs
        rows += [[0, i, 1985], [0, i, 2005]]
    rows += [[1, 0, 1985]]                 # and one present in 1985 only
    pidx = np.array(rows, dtype=np.int32)
    holdout = np.ones((2, 20), bool)       # every cell held out -> all land in val_of
    buf = np.zeros((2, 20), bool)
    rng = np.random.default_rng(0)
    z_obs = rng.normal(size=(len(pidx), 3)).astype("float32")
    z_model = {(int(r), int(c), int(y)): z_obs[i] for i, (r, c, y) in enumerate(pidx)}

    out = epoch_direction_panel(pidx, None, z_obs, z_model, holdout, buf,
                                epochs=epochs, verbose=False)
    assert out["pairs"]["1985_2005"]["n"] == 20      # the one-epoch cell is dropped, not fatal


def test_the_epoch_panel_idw_bar_may_not_source_from_a_withheld_year():
    """The bug this whole round is about. For an epoch inside a temporal holdout the IDW bar
    would interpolate that year's TRUTH from that year's neighbours, while the model saw no truth
    from that year anywhere. Different information sets are not a bar, so the bar must go away."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    epochs, withheld = (1970, 2020), [1970]
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):                      # 30 TRAIN cells (col 0..29) at both epochs
        rows += [[0, i, 1970], [0, i, 2020]]
    for i in range(30):                      # 30 VAL cells (row 1) at both epochs
        rows += [[1, i, 1970], [1, i, 2020]]
    pidx = np.array(rows, dtype=np.int32)
    holdout = np.zeros((2, 30), bool); holdout[1, :] = True
    buf = np.zeros((2, 30), bool)
    z_obs = rng.normal(size=(len(pidx), 4)).astype("float32")
    z_model = {(int(r), int(c), int(y)): z_obs[i] for i, (r, c, y) in enumerate(pidx)}

    free = epoch_direction_panel(pidx, None, z_obs, z_model, holdout, buf,
                                 epochs=epochs, verbose=False)
    fair = epoch_direction_panel(pidx, None, z_obs, z_model, holdout, buf,
                                 epochs=epochs, verbose=False, exclude_years=withheld)
    key = "1970_2020"
    assert np.isfinite(free["pairs"][key]["idw_dir_cos"])          # bar ran: it saw 1970
    assert not np.isfinite(fair["pairs"][key]["idw_dir_cos"])      # and must not
    # a missing bar is reported as such, never as parity with the model
    assert fair["pairs"][key]["verdict"] == "no-bar"
    assert 1970 in fair["epochs_without_bar"]
    # the model side is untouched -- only the bar changed
    assert free["pairs"][key]["model_dir_cos"] == fair["pairs"][key]["model_dir_cos"]


def test_half_width_zero_reproduces_the_single_year_panel_exactly():
    """The windowed path must be a strict superset: `half_width=0` has to reproduce the pinned
    single-year selection (nearest survey within tol, ties to the earlier year), or every number
    reported before windowing existed becomes unreproducible."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    rng = np.random.default_rng(1)
    rows = []
    for i in range(25):
        for y in (1969, 1971, 2004, 2006):          # off-epoch years, inside tol=2
            rows += [[0, i, y], [1, i, y]]
    pidx = np.array(rows, dtype=np.int32)
    holdout = np.zeros((2, 25), bool); holdout[1, :] = True
    buf = np.zeros((2, 25), bool)
    z_obs = rng.normal(size=(len(pidx), 4)).astype("float32")
    z_model = {(int(r), int(c), int(y)): z_obs[i] for i, (r, c, y) in enumerate(pidx)}
    kw = dict(epochs=(1970, 2005), verbose=False)
    a = epoch_direction_panel(pidx, None, z_obs, z_model, holdout, buf, **kw)
    b = epoch_direction_panel(pidx, None, z_obs, z_model, holdout, buf, half_width=0, **kw)

    def same(x, y):
        """Plain == fails on NaN fields (nan != nan), and rows legitimately carry NaN wherever a
        bar is unavailable -- so compare NaN-aware rather than weakening the assertion."""
        if isinstance(x, float) and isinstance(y, float):
            return (np.isnan(x) and np.isnan(y)) or x == y
        return x == y

    assert set(a["pairs"]) == set(b["pairs"])
    for key, row in a["pairs"].items():
        assert set(row) == set(b["pairs"][key]), key
        for f, v in row.items():
            assert same(v, b["pairs"][key][f]), (key, f, v, b["pairs"][key][f])
    assert a["pairs"]["1970_2005"]["mean_window_depth"] == 1.0    # one year per endpoint


def test_windowing_averages_model_and_target_over_the_same_years():
    """Symmetry is the whole content of the old 'no averaging' rule: smoothing the target while
    reading the model at one year compares two different quantities. With a field that is
    IDENTICAL in model and target, any asymmetry in which years each side averages would drive
    dir-cos off 1.0; equal treatment keeps it exactly 1.0."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    rng = np.random.default_rng(2)
    rows = []
    for i in range(25):
        for y in (1968, 1969, 1970, 1971, 1972, 2003, 2004, 2005):
            rows += [[0, i, y], [1, i, y]]
    pidx = np.array(rows, dtype=np.int32)
    holdout = np.zeros((2, 25), bool); holdout[1, :] = True
    buf = np.zeros((2, 25), bool)
    z_obs = rng.normal(size=(len(pidx), 4)).astype("float32")
    z_model = {(int(r), int(c), int(y)): z_obs[i] for i, (r, c, y) in enumerate(pidx)}
    out = epoch_direction_panel(pidx, None, z_obs, z_model, holdout, buf,
                                epochs=(1970, 2005), verbose=False, half_width=2)
    p = out["pairs"]["1970_2005"]
    assert p["model_dir_cos"] > 0.999, p          # identical fields, identical windows
    assert p["mean_window_depth"] > 1.0, p        # and the window really did average


def test_windowing_recovers_direction_lost_to_survey_noise():
    """The point of windowing: a single-year endpoint is one observer on one morning, and that
    noise attenuates dir-cos toward zero. Averaging the window must recover some of it."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    rng = np.random.default_rng(3)
    years = (1968, 1969, 1970, 1971, 1972, 2003, 2004, 2005, 2006, 2007)
    rows, truth, noisy = [], {}, {}
    for i in range(40):
        d = rng.normal(size=4)                    # this cell's true change direction
        d /= np.linalg.norm(d)
        for y in years:
            rows.append([1, i, y])                # all val cells
            clean = d * (1.0 if y > 2000 else -1.0)
            truth[(1, i, y)] = clean
            noisy[(1, i, y)] = clean + rng.normal(scale=1.4, size=4)
    pidx = np.array(rows, dtype=np.int32)
    z_obs = np.stack([noisy[(int(r), int(c), int(y))] for r, c, y in pidx]).astype("float32")
    z_model = {(int(r), int(c), int(y)): truth[(int(r), int(c), int(y))] for r, c, y in pidx}
    holdout = np.ones((2, 40), bool); buf = np.zeros((2, 40), bool)
    kw = dict(epochs=(1970, 2005), verbose=False)
    single = epoch_direction_panel(pidx, None, z_obs, z_model, holdout, buf, **kw)
    win = epoch_direction_panel(pidx, None, z_obs, z_model, holdout, buf, half_width=2, **kw)
    s = single["pairs"]["1970_2005"]["model_dir_cos"]
    w = win["pairs"]["1970_2005"]["model_dir_cos"]
    assert w > s, (s, w)                          # averaging recovers attenuated direction


def test_attenuation_estimator_recovers_injected_noise():
    """`per_era_attenuation` must measure the noise, not assume it. Built on a field with a known
    linear trend plus known-variance noise, so the attenuation factor is known in closed form."""
    from src.community_encoder.train_DESK.validate_baselines import per_era_attenuation
    rng = np.random.default_rng(4)
    D, L, g, s = 4, 30, 0.30, 0.5                 # dims, span, per-year trend, noise scale
    v = np.zeros(D); v[0] = 1.0
    rows, z = [], []
    for cell in range(60):
        for t in range(L + 1):
            rows.append([0, cell, 1970 + t])
            z.append(v * g * t + rng.normal(scale=s, size=D))
    pidx = np.array(rows, dtype=np.int32)
    z = np.stack(z).astype("float32")
    out = per_era_attenuation(pidx, z, era_width=L)
    era = f"{1970 // L * L}s"
    tau2, noise2 = (g * L) ** 2, 2.0 * D * s ** 2
    expected = np.sqrt(tau2 / (tau2 + noise2))
    got = out[era]["dir_cos_attenuation"]
    # The estimator is very slightly conservative: adjacent-year differences carry one year of
    # real change as well as noise, biasing it by (L^2-1)/L^2 -- 0.1% at L=30.
    assert abs(got - expected) < 0.03, (got, expected)


def test_year_coverage_matches_the_years_the_bar_actually_uses():
    """The reported coverage and the years the bar sums over must come from one predicate, or the
    print drifts from the number it is describing -- which is how the population mismatch hid."""
    from src.community_encoder.train_DESK.validate_baselines import (
        _interp_usable_years, interp_year_coverage)
    H = W = 12
    z = np.zeros((H, W, 3), dtype="float32")
    present = np.ones((H, W), bool)
    ho = np.zeros((H, W), bool); ho[:, :4] = True
    tgt = {1970: (z, np.zeros((H, W), bool), present & ho, np.ones((H, W), "float32")),
           2000: (z, present & ~ho, present & ho, np.ones((H, W), "float32"))}
    assert _interp_usable_years(tgt) == [2000]        # 1970 has no training cells
    assert interp_year_coverage(tgt) == (1, 2)


def test_long_gap_probe_is_used_only_when_years_are_withheld():
    """The bar's anisotropy must be fitted on the gap it is judging. With a temporal holdout the
    probe is the earliest contiguous TRAINING block (a synthetic backward extrapolation); with no
    holdout there is nothing to match and it stays the random half."""
    import io
    import contextlib

    from src.community_encoder.train_DESK.validate_baselines import spacetime_idw_baseline
    rng = np.random.default_rng(5)
    rows = [[r, c, y] for r in range(6) for c in range(6) for y in range(1980, 2011, 2)]
    pidx = np.array(rows, dtype=np.int32)
    z = rng.normal(size=(len(pidx), 3)).astype("float32")
    ho = np.zeros((6, 6), bool); ho[5, :] = True
    target = ho[pidx[:, 0], pidx[:, 1]]

    def _mode(exclude):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            spacetime_idw_baseline(pidx, z, ho, target, exclude_years=exclude)
        return buf.getvalue()

    assert "long-gap probe" in _mode(list(range(1966, 1980)))
    assert "random-half probe" in _mode(())


def test_the_sweep_overlays_are_mutually_consistent():
    """The three temporal-holdout overlays only produce a comparable distance curve if the
    anchors and the common window are IDENTICAL across them and the common window is withheld in
    every run. A drifting anchor is what confounded the first sweep, so pin it with a test."""
    import glob
    import json
    import os

    root = os.path.join(os.path.dirname(__file__), "..", "config", "overlays")
    paths = sorted(glob.glob(os.path.join(root, "desk_tempho_*.json")))
    assert len(paths) >= 3, paths
    seen = {}
    for p in paths:
        t = json.load(open(p))["desk"]["trend"]
        ho, common = set(t["holdout_years"]), set(t["common_holdout_years"])
        # the common window must actually be withheld in this run, or it is not common
        assert common and common <= ho, (p, sorted(common - ho))
        # the trained-era control must NOT be withheld here; the withheld anchor MUST be
        assert t["direction_anchor_year"] not in ho, p
        assert t["direction_withheld_anchor_year"] in ho, p
        for k in ("direction_anchor_year", "direction_withheld_anchor_year",
                  "common_holdout_years"):
            seen.setdefault(k, []).append(json.dumps(t[k], sort_keys=True))
    for k, vals in seen.items():
        assert len(set(vals)) == 1, f"{k} differs across the sweep overlays: {set(vals)}"


def test_the_windowed_and_single_year_panels_cover_the_same_cells():
    """Inclusion is "a survey within +/-tol" for the single-year path and "within +/-half_width"
    for the windowed one, so a half_width below tol silently drops cells -- and the gap between
    the two tables, which is supposed to measure NOISE, would instead be a population change.
    Measured before the fix: n=30 at half_width=0 collapsed to n=0 at half_width=1."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    rng = np.random.default_rng(0)
    pidx = np.array([[1, i, y] for i in range(30) for y in (1968, 1972, 2003, 2007)],
                    dtype=np.int32)
    z = rng.normal(size=(len(pidx), 4)).astype("float32")
    zm = {(int(r), int(c), int(y)): z[i] for i, (r, c, y) in enumerate(pidx)}
    ho = np.ones((2, 30), bool); bf = np.zeros((2, 30), bool)
    ns = []
    for hw in (0, 1, 2, 3):
        p = epoch_direction_panel(pidx, None, z, zm, ho, bf, epochs=(1970, 2005),
                                  verbose=False, half_width=hw)["pairs"].get("1970_2005")
        ns.append(p["n"] if p else 0)
    assert len(set(ns)) == 1 and ns[0] == 30, ns


def test_excluding_years_changes_only_the_bar_never_the_model_or_the_cells():
    """The year filter must land on the IDW source set alone. If it reached val_of it would drop
    exactly the held-out-in-a-withheld-year rows the experiment exists to score."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        rows += [[0, i, 1970], [0, i, 2020], [1, i, 1970], [1, i, 2020]]
    pidx = np.array(rows, dtype=np.int32)
    ho = np.zeros((2, 30), bool); ho[1, :] = True
    bf = np.zeros((2, 30), bool)
    z = rng.normal(size=(len(pidx), 4)).astype("float32")
    zm = {(int(r), int(c), int(y)): z[i] for i, (r, c, y) in enumerate(pidx)}
    kw = dict(epochs=(1970, 2020), verbose=False)
    a = epoch_direction_panel(pidx, None, z, zm, ho, bf, **kw)["pairs"]["1970_2020"]
    b = epoch_direction_panel(pidx, None, z, zm, ho, bf, exclude_years=[1970],
                              **kw)["pairs"]["1970_2020"]
    assert a["n"] == b["n"]                                        # cells unchanged
    assert a["model_dir_cos"] == b["model_dir_cos"]                # model unchanged
    assert a["null_dir_cos"] == b["null_dir_cos"]                  # null unchanged
    assert np.isfinite(a["idw_dir_cos"]) and not np.isfinite(b["idw_dir_cos"])


def test_which_ladder_rungs_survive_each_temporal_bucket():
    """Pins the n/a structure, because it determines WHICH claim each bucket can support. A
    held-out cell has no training years of its own, so no per-cell rung can run there: for the
    doubly-held-out bucket the only admissible bars are no_change and spacetime_idw."""
    from src.community_encoder.train_DESK.validate_baselines import baseline_panel
    rng = np.random.default_rng(0)
    hy = list(range(1966, 1976))
    pidx = np.array([[r, c, y] for r in range(8) for c in range(8)
                     for y in list(range(1966, 1996, 3)) + [2025]], dtype=np.int32)
    z = rng.normal(size=(len(pidx), 4)).astype("float32")
    zd = z + rng.normal(scale=0.1, size=z.shape).astype("float32")
    ho = np.zeros((8, 8), bool); ho[6:, :] = True
    bf = np.zeros((8, 8), bool)
    is_ho, in_hy = ho[pidx[:, 0], pidx[:, 1]], np.isin(pidx[:, 2], np.asarray(hy))
    # the two buckets must be disjoint and together cover every withheld row
    assert not (in_hy & ~is_ho & (in_hy & is_ho)).any()
    assert (((in_hy & ~is_ho) | (in_hy & is_ho)) == in_hy).all()

    def avail(rowsel):
        o = baseline_panel(pidx, z, zd, ho, 2025, buffer_mask=bf, target_rows=rowsel,
                           exclude_years=hy, verbose=False)["overall"]
        return {n for n in ("no_change", "cell_nearest_year", "cell_trend", "borrowed_delta",
                            "spacetime_idw") if o["predictors"][n]["n"] > 0}

    # unseen YEAR, seen cell: the cell has training years elsewhere, so a trend can be fit
    assert avail(in_hy & ~is_ho) == {"no_change", "cell_nearest_year", "cell_trend",
                                     "spacetime_idw"}
    # unseen year AND unseen cell: no training rows for this cell at all
    assert avail(in_hy & is_ho) == {"no_change", "spacetime_idw"}


def test_a_boundary_anisotropy_is_flagged_as_censored():
    """An argmin on the edge of the grid is censored, not measured: the optimum may lie outside,
    so the bar is a lower bound and any DESK margin over it is overstated. This is not
    hypothetical -- the first temporal-holdout sweep selected the grid floor in all three runs,
    which would have made a narrow win look real."""
    import contextlib
    import io

    from src.community_encoder.train_DESK.validate_baselines import (
        SPACETIME_RATIOS, spacetime_idw_baseline)
    # the grid must reach well below 1 cell/yr, which is where the sweep actually landed
    assert min(SPACETIME_RATIOS) <= 0.01, SPACETIME_RATIOS

    rng = np.random.default_rng(0)
    pidx = np.array([[r, c, y] for r in range(6) for c in range(6)
                     for y in range(1980, 2011, 2)], dtype=np.int32)
    ho = np.zeros((6, 6), bool); ho[5, :] = True
    target = ho[pidx[:, 0], pidx[:, 1]]
    # a field that varies ONLY in space: time distance is pure cost, so the winner is the
    # smallest ratio on offer -- i.e. the low boundary, which must be flagged.
    z = np.stack([[float(r), float(c)] for r, c, _y in pidx]).astype("float32")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _err, ratio = spacetime_idw_baseline(pidx, z, ho, target)
    out = buf.getvalue()
    assert ratio == min(SPACETIME_RATIOS), (ratio, out)
    assert "WARNING" in out and "LOW end of the grid" in out, out


def _atten_points(era_start, n_cells, span, noise, seed, trend=0.30, dims=4):
    """Cells whose records all START at era_start and run for `span` years, with known noise."""
    rng = np.random.default_rng(seed)
    v = np.zeros(dims); v[0] = 1.0
    rows, z = [], []
    for c in range(n_cells):
        for t in range(span + 1):
            rows.append([0, c, era_start + t])
            z.append(v * trend * t + rng.normal(scale=noise, size=dims))
    return np.array(rows, dtype=np.int32), np.stack(z).astype("float32")


def test_attenuation_is_controlled_for_record_length():
    """The bug this replaced: binning an era on a cell's FIRST survey year while taking the long
    baseline from that cell's FULL span makes span vary with era by construction. A longer record
    accumulates more real change, so long_sq grows and the attenuation figure rises -- the table
    then ranks RECORD LENGTH, not era noisiness. Two eras with identical noise and identical
    trend, differing only in how long their records run, must report the same attenuation."""
    from src.community_encoder.train_DESK.validate_baselines import per_era_attenuation
    # same noise, same per-year trend; the 1960s cells are surveyed for 50 yr, the 1990s for 25
    p_long, z_long = _atten_points(1960, 60, 50, noise=0.5, seed=0)
    p_short, z_short = _atten_points(1990, 60, 25, noise=0.5, seed=1)
    pidx = np.vstack([p_long, p_short])
    pidx[len(p_long):, 1] += 100                      # keep the two eras on disjoint cells
    z = np.vstack([z_long, z_short])
    out = per_era_attenuation(pidx, z, era_width=10, min_pairs=10)
    a60, a90 = out["1960s"]["dir_cos_attenuation"], out["1990s"]["dir_cos_attenuation"]
    assert abs(a60 - a90) < 0.05, (a60, a90)          # record length must not move it
    # and the long baseline really is the fixed gap, not the cell's span
    assert out["1960s"]["gap_years"] == out["1990s"]["gap_years"] == 20


def test_attenuation_recovers_the_analytic_value_per_era():
    """Having controlled span, the estimator must still measure a genuine noise difference -- and
    measure it CORRECTLY, not merely order it. With a per-year trend g over a fixed gap G and
    per-component noise s in D dims, tau^2 = (g*G)^2 and the noise contributes 2*D*s^2, so the
    attenuation is sqrt(tau^2 / (tau^2 + 2*D*s^2)) in closed form."""
    from src.community_encoder.train_DESK.validate_baselines import (
        ATTEN_GAP, per_era_attenuation)
    g, D = 0.30, 4
    p_quiet, z_quiet = _atten_points(1960, 60, 30, noise=0.2, seed=2, trend=g, dims=D)
    p_noisy, z_noisy = _atten_points(1990, 60, 30, noise=0.9, seed=3, trend=g, dims=D)
    pidx = np.vstack([p_quiet, p_noisy])
    pidx[len(p_quiet):, 1] += 100                     # disjoint cells per era
    z = np.vstack([z_quiet, z_noisy])
    out = per_era_attenuation(pidx, z, era_width=10, min_pairs=10)

    def expected(s):
        tau2 = (g * ATTEN_GAP) ** 2
        return float(np.sqrt(tau2 / (tau2 + 2 * D * s ** 2)))

    for era, s in (("1960s", 0.2), ("1990s", 0.9)):
        got = out[era]["dir_cos_attenuation"]
        assert abs(got - expected(s)) < 0.02, (era, got, expected(s))
    # and the noisier era is the more attenuated one
    assert out["1960s"]["dir_cos_attenuation"] > out["1990s"]["dir_cos_attenuation"], out


def test_the_error_decomposition_is_exact():
    """The identity everything in the magnitude/angular reporting rests on. If magnitude and
    angular do not sum to the total, every derived reading -- cal, the shrinkage profile, the
    epoch panel's attribution -- is measuring something other than it claims."""
    from src.community_encoder.train_DESK.validate_baselines import error_decomposition
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 9)), rng.normal(size=(200, 9))
    total, mag, ang, cos = error_decomposition(a, b)
    assert np.allclose(mag + ang, total, rtol=0, atol=1e-9)
    assert np.allclose(total, np.sum((a - b) ** 2, axis=-1))
    # the angular term must match its closed form where cos is defined
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    assert np.allclose(ang, 2 * na * nb * (1 - cos), rtol=0, atol=1e-9)


def test_the_decomposition_survives_a_zero_length_prediction():
    """A shrunk-to-nothing prediction is exactly the failure mode the magnitude term exists to
    expose, so it must not produce a NaN total or break the identity."""
    from src.community_encoder.train_DESK.validate_baselines import error_decomposition
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
    total, mag, ang, cos = error_decomposition(a, b)
    assert np.allclose(mag + ang, total)
    assert np.isclose(total[0], 25.0) and np.isclose(mag[0], 25.0) and np.isclose(ang[0], 0.0)
    assert np.isnan(cos[0])                      # undefined direction, reported as such
    assert np.isclose(total[1], 0.0) and np.isclose(cos[1], 1.0)


def test_shrinkage_is_the_mse_optimal_answer_to_a_poor_angle():
    """Pins the trade-off the decomposition makes visible: at a fixed direction cosine rho, the
    total is minimised at ||a|| = rho*||b||. This is where rot ~ dcos^2 comes from, and why
    reporting an angle alone cannot explain an error."""
    from src.community_encoder.train_DESK.validate_baselines import error_decomposition
    rng = np.random.default_rng(1)
    b = rng.normal(size=12); b /= np.linalg.norm(b)
    perp = rng.normal(size=12); perp -= (perp @ b) * b; perp /= np.linalg.norm(perp)
    for rho in (0.2, 0.5, 0.8):
        u = rho * b + np.sqrt(1 - rho ** 2) * perp        # unit vector at cosine rho to b
        scales = np.linspace(0.01, 2.0, 4000)
        totals = error_decomposition(scales[:, None] * u[None, :],
                                     np.broadcast_to(b, (len(scales), 12)))[0]
        assert abs(scales[np.argmin(totals)] - rho) < 0.01, (rho, scales[np.argmin(totals)])


def test_the_epoch_panel_reports_the_magnitude_half_not_just_the_angle():
    """dir-cos is one half of an exact split of the change-vector error. Reported alone it cannot
    tell "moved the wrong way" from "barely moved" -- and those are different models, because
    under-moving is the MSE-optimal response to a poor angle. Two predictions with the SAME
    dir-cos and very different magnitudes must be distinguishable in the report."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    rng = np.random.default_rng(0)
    years = (1970, 2020)
    cells = 40
    pidx = np.array([[1, i, y] for i in range(cells) for y in years], dtype=np.int32)
    ho = np.ones((2, cells), bool); bf = np.zeros((2, cells), bool)

    def panel(scale):
        z_obs, model = [], {}
        for i in range(cells):
            d = rng.normal(size=4); d /= np.linalg.norm(d)
            off = rng.normal(size=4) * 0.35                   # fixed angular error budget
            for y in years:
                z_obs.append(d * (1.0 if y > 2000 else -1.0))
            # model change = scale * (truth direction + the same angular perturbation)
            model[(1, i, years[0])] = -(d + off) * scale / 2.0
            model[(1, i, years[1])] = (d + off) * scale / 2.0
        zo = np.stack(z_obs).astype("float32")
        return epoch_direction_panel(pidx, None, zo, model, ho, bf, epochs=years,
                                     verbose=False)["pairs"]["1970_2020"]

    small, big = panel(0.2), panel(2.0)
    # same direction error, so the ANGLE is unchanged and cannot separate them...
    assert abs(small["model_dir_cos"] - big["model_dir_cos"]) < 0.02, (small, big)
    # ...but the magnitude half does, which is the whole point of reporting it
    assert big["change_magnitude_ratio"] > 5 * small["change_magnitude_ratio"]
    # The SHARE is large exactly when the magnitude is WRONG: `small` barely moves (ratio ~0.1)
    # so its error is almost all magnitude, while `big` is scaled about right (ratio ~1.1) and
    # its error is mostly angular. So the badly-scaled one carries the larger magnitude share.
    assert small["err_magnitude_share"] > 0.8, small
    assert big["err_magnitude_share"] < 0.5, big
    # the two shares must partition the error
    for r in (small, big):
        assert abs(r["err_magnitude_share"] + r["err_angular_share"] - 1.0) < 1e-9


def test_the_spacetime_bar_reaches_epochs_the_spatial_bar_cannot():
    """The per-epoch spatial bar needs training cells IN that year, so it goes n/a for any epoch
    inside the temporal holdout -- which is precisely the deep-past epoch the experiment exists to
    measure. Those pairs therefore had no bar at all. The spacetime bar borrows across years too,
    so it must produce a finite value exactly where the spatial one cannot."""
    from src.community_encoder.train_DESK.validate_baselines import epoch_direction_panel
    epochs, withheld = (1970, 2020), [1970]
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        rows += [[0, i, 1970], [0, i, 2020], [1, i, 1970], [1, i, 2020]]
    pidx = np.array(rows, dtype=np.int32)
    ho = np.zeros((2, 30), bool); ho[1, :] = True
    bf = np.zeros((2, 30), bool)
    z_obs = rng.normal(size=(len(pidx), 4)).astype("float32")
    z_model = {(int(r), int(c), int(y)): z_obs[i] for i, (r, c, y) in enumerate(pidx)}
    z_st = rng.normal(size=(len(pidx), 4)).astype("float32")     # a stand-in bar at every row
    kw = dict(epochs=epochs, verbose=False, exclude_years=withheld)

    without = epoch_direction_panel(pidx, None, z_obs, z_model, ho, bf, **kw)
    with_bar = epoch_direction_panel(pidx, None, z_obs, z_model, ho, bf,
                                     z_spacetime=z_st, **kw)
    k = "1970_2020"
    # the spatial bar is unavailable in both -- 1970 is withheld
    assert not np.isfinite(without["pairs"][k]["idw_dir_cos"])
    assert not np.isfinite(with_bar["pairs"][k]["idw_dir_cos"])
    # but the spacetime bar reaches it
    assert not np.isfinite(without["pairs"][k]["spacetime_idw_dir_cos"])
    assert np.isfinite(with_bar["pairs"][k]["spacetime_idw_dir_cos"]), with_bar["pairs"][k]
    # and adding a bar must not perturb the model's own numbers
    assert without["pairs"][k]["model_dir_cos"] == with_bar["pairs"][k]["model_dir_cos"]
    assert without["pairs"][k]["n"] == with_bar["pairs"][k]["n"]


def test_per_dimension_split_recovers_a_planted_spectrum():
    """Without this, a flat result is uninformative -- it could mean the estimator is blind rather
    than the signal being absent. Same reason the ESK oracle needs a detect-a-high-ceiling test.

    Signal is planted in KNOWN leading dimensions and noise spread over all of them, so the
    estimator must put signal_var where the signal is and noise_var everywhere."""
    from src.community_encoder.train_DESK.validate_baselines import (
        ATTEN_GAP, per_dimension_signal_noise)
    rng = np.random.default_rng(0)
    L, sig_dims, noise_sd, g = 16, 4, 0.4, 0.25
    rows, z = [], []
    for cell in range(80):
        d = np.zeros(L); d[:sig_dims] = rng.normal(size=sig_dims)   # this cell's drift direction
        for t in range(ATTEN_GAP + 3):
            rows.append([0, cell, 1970 + t])
            z.append(d * g * t + rng.normal(scale=noise_sd, size=L))
    out = per_dimension_signal_noise(np.array(rows, dtype=np.int32),
                                     np.stack(z).astype("float32"), min_pairs=10)
    sig = np.array(out["signal_var"]); noi = np.array(out["noise_var"])
    # signal concentrates on the planted dimensions
    assert sig[:sig_dims].mean() > 20 * sig[sig_dims:].mean(), (sig[:sig_dims], sig[sig_dims:])
    # noise is spread across all of them, so the empty dims are noise-only
    assert noi[sig_dims:].mean() > 0.5 * noi[:sig_dims].mean()
    # signal is in the LEADING half here, so SNR must fall along the basis
    assert out["snr_slope"] < 0, out["snr_slope"]
    assert out["snr_leading_8"] > out["snr_trailing_8"]
    assert out["signal_share_leading_half"] > 0.9, out["signal_share_leading_half"]


def test_per_dimension_split_detects_signal_in_the_TRAILING_directions():
    """The reading that would make the shrinkage tilt a real defect. If the estimator could only
    ever report signal-in-the-leading-dims it would confirm the convenient answer by construction,
    so plant the opposite arrangement and require it to be found."""
    from src.community_encoder.train_DESK.validate_baselines import (
        ATTEN_GAP, per_dimension_signal_noise)
    rng = np.random.default_rng(1)
    L, g = 16, 0.25
    rows, z = [], []
    for cell in range(80):
        d = np.zeros(L); d[-4:] = rng.normal(size=4)                # signal in the TAIL
        for t in range(ATTEN_GAP + 3):
            rows.append([0, cell, 1970 + t])
            z.append(d * g * t + rng.normal(scale=0.4, size=L))
    out = per_dimension_signal_noise(np.array(rows, dtype=np.int32),
                                     np.stack(z).astype("float32"), min_pairs=10)
    assert out["snr_slope"] > 0, out["snr_slope"]
    assert out["snr_trailing_8"] > out["snr_leading_8"]
    assert out["signal_share_leading_half"] < 0.1, out["signal_share_leading_half"]


def test_per_dimension_split_refuses_on_too_few_pairs():
    from src.community_encoder.train_DESK.validate_baselines import per_dimension_signal_noise
    pidx = np.array([[0, 0, 1970], [0, 0, 1971]], dtype=np.int32)
    out = per_dimension_signal_noise(pidx, np.zeros((2, 5), "float32"), min_pairs=30)
    assert "note" in out and "signal_var" not in out


def test_the_direction_panel_ceiling_is_an_independent_observation():
    """A dir-cos has no scale without it: 0.23 reads as poor against 1.0 and as good against a
    ceiling of 0.35, and only the second comparison means anything. The ceiling is built from a
    DISJOINT half of each window's years, so it is an independent look at the same place rather
    than the target restated."""
    from src.community_encoder.train_DESK.validate_baselines import _half_years, _ceiling_row
    # halves alternate through the sorted years -- an early/late split would put a real time
    # gradient between them and the ceiling would understate itself
    assert _half_years([1970, 1971, 1972, 1973, 1974], 0) == [1970, 1972, 1974]
    assert _half_years([1970, 1971, 1972, 1973, 1974], 1) == [1971, 1973]
    a, b = _half_years(range(1970, 1980), 0), _half_years(range(1970, 1980), 1)
    assert not (set(a) & set(b)), (a, b)              # DISJOINT is the whole point
    with pytest.raises(ValueError):
        _half_years([1970], 0)                       # cannot split; caller must drop the pair

    rng = np.random.default_rng(0)
    n, L = 200, 8
    signal = rng.normal(size=(n, L))
    # two independent noisy looks at the same underlying change
    dtA = signal + rng.normal(scale=0.6, size=(n, L))
    dtB = signal + rng.normal(scale=0.6, size=(n, L))
    row = _ceiling_row((dtA, dtB), null_cos=0.02, model_cos=0.30)
    assert 0.2 < row["ceiling_dir_cos"] < 0.95, row  # noisy, so well below 1.0
    assert row["room"] == row["ceiling_dir_cos"] - 0.02
    assert 0.0 < row["share_of_room"] < 1.5, row

    # a noiseless target gives a ceiling of ~1: then the model really is being read against 1.0
    clean = _ceiling_row((signal, signal), null_cos=0.0, model_cos=0.5)
    assert clean["ceiling_dir_cos"] > 0.999, clean

    # not enough splittable cells -> a stated reason, never a silent number
    assert "fewer than 4 cells" in _ceiling_row(None, 0.0, 0.5)["ceiling_note"]


def test_the_direction_panel_flags_a_narrow_comparison():
    from src.community_encoder.train_DESK.validate_baselines import _ceiling_row
    rng = np.random.default_rng(1)
    d = rng.normal(size=(100, 6))
    # ceiling barely above the null -> nothing to resolve
    row = _ceiling_row((d, d * 0.0 + rng.normal(scale=5.0, size=(100, 6))),
                       null_cos=0.02, model_cos=0.05)
    if np.isfinite(row.get("room", np.nan)) and row["room"] < 0.15:
        assert "NARROW" in row["room_verdict"], row


def test_one_smoothing_length_by_default_and_divergences_must_say_why():
    """Same number of years averaged everywhere unless there is a stated reason.

    A measurement that averages a different span than its neighbours is not comparable to them,
    and every instance of that here so far was an accident: three constants for one concept, two
    modules disagreeing on whether to average raw counts or z at the same width, and the two ends
    of a single difference averaged over 5 and 16 surveys.
    """
    from src.community_encoder.train_DESK.validate_baselines import (
        smoothing_half_width, smoothing_manifest, SMOOTHING_DIVERGENCES)
    cfg = {"target": {"smooth_half_width": 2}}
    assert smoothing_half_width(cfg) == 2
    assert smoothing_half_width({"bbs_routes": {"window_half_width": 3}}) == 3   # legacy fallback
    assert smoothing_half_width({}) == 2

    m = smoothing_manifest(cfg, {"a": 2, "bbs_routes.epoch_eras": "whole era (~13 surveys)"})
    assert m["default_half_width_yr"] == 2
    assert m["measurements"]["a"]["matches_default"] is True
    era = m["measurements"]["bbs_routes.epoch_eras"]
    assert era["matches_default"] is False and era["reason"]
    assert m["unjustified"] == []                       # declared, so it passes

    # an undeclared difference is reported, not accepted
    bad = smoothing_manifest(cfg, {"somewhere_new": 9})
    assert len(bad["unjustified"]) == 1
    assert "no reason is declared" in bad["unjustified"][0]

    # every declared divergence carries a real reason, not a placeholder
    for name, (width, reason) in SMOOTHING_DIVERGENCES.items():
        assert width and len(reason) > 80, name


def test_one_unsplittable_cell_does_not_destroy_the_whole_ceiling():
    """Regression. The first version wrapped the per-cell loop in a single try/except, so one cell
    with a one-year window returned no ceiling for the entire epoch pair. With a mean window depth
    of 3.1 surveys that fired on every pair of every run, and the ceiling column came back empty.
    """
    from src.community_encoder.train_DESK.validate_baselines import _half_years
    windows = {"a": [1970, 1971, 1972], "b": [1985], "c": [2000, 2001]}
    ok, dropped = [], []
    for name, yrs in windows.items():
        try:
            _half_years(yrs, 0), _half_years(yrs, 1)
            ok.append(name)
        except ValueError:
            dropped.append(name)
    assert ok == ["a", "c"] and dropped == ["b"]     # the short one goes, the others stay


# ---------------------------------------------------------------------------------------------
# Species space -- which species rise and fall
# ---------------------------------------------------------------------------------------------

def _species_world(n=400, n_sp=30, latent=6, seed=0):
    """Communities generated from latent coordinates, so a linear readout can in principle work."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, latent))
    W = rng.normal(size=(latent, n_sp)) * 0.8
    X = np.clip(Z @ W + 3.0 + rng.normal(scale=0.1, size=(n, n_sp)), 0.0, None)
    return Z, X


def test_the_species_readout_is_reported_in_sample_before_anything_else():
    """If the coordinates cannot reproduce abundances on the data the map was fitted to, nothing
    computed from the readout is about DESK."""
    from src.community_encoder.train_DESK.validate_baselines import (
        fit_species_readout, apply_species_readout)
    Z, X = _species_world()
    r = fit_species_readout(Z[:300], X[:300])
    assert r["train_r2"] > 0.9, r                       # generated linearly, so it must fit
    assert r["n_train"] == 300 and r["n_species"] == 30
    # and it generalises to rows it never saw
    held = apply_species_readout(r, Z[300:])
    ss_res = ((X[300:] - held) ** 2).sum()
    ss_tot = ((X[300:] - X[300:].mean(0)) ** 2).sum()
    assert 1 - ss_res / ss_tot > 0.85

    # coordinates that carry NO species information must show it in-sample, not silently pass
    rng = np.random.default_rng(1)
    junk = fit_species_readout(rng.normal(size=(300, 6)), X[:300])
    assert junk["train_r2"] < 0.2, junk


def test_direction_skill_is_not_fooled_by_guessing_the_commoner_direction():
    """BBS declines are widespread, so 'everything declined' scores well against a coin flip and
    would read as skill. The correction against the majority guess is the whole point."""
    from src.community_encoder.train_DESK.validate_baselines import species_change_agreement
    rng = np.random.default_rng(0)
    n_sp = 200
    xe = np.full((1, n_sp), 3.0)
    drop = rng.random(n_sp) < 0.8                       # 80% of species decline
    xm = xe - np.where(drop, 1.0, -1.0) * rng.uniform(0.3, 1.0, n_sp)

    lazy = species_change_agreement(xe, xm, xe, xe - 1.0)          # predicts decline for all
    # By construction the lazy predictor achieves EXACTLY the majority rate -- that is what makes
    # it lazy -- so assert against the realised rate rather than the intended 0.8, which the draw
    # will not hit exactly.
    assert lazy["direction_hit_rate"] == lazy["majority_direction_rate"], lazy
    assert lazy["direction_hit_rate"] > 0.6, lazy                   # looks respectable raw...
    assert abs(lazy["direction_skill"]) < 0.05, lazy                # ...and scores ~0 corrected

    skilled = species_change_agreement(xe, xm, xe, xm)              # gets every species right
    assert skilled["direction_skill"] > 0.99, skilled


def test_direction_and_rank_separate():
    """A predictor can get every sign right and still rank the magnitudes backwards."""
    from src.community_encoder.train_DESK.validate_baselines import species_change_agreement
    rng = np.random.default_rng(2)
    n_sp = 120
    xe = np.full((1, n_sp), 4.0)
    mag = rng.uniform(0.2, 2.0, n_sp)
    sign = np.where(rng.random(n_sp) < 0.5, 1.0, -1.0)
    xm = xe + sign * mag
    # right sign, magnitude ordering exactly reversed: the species that moved most is predicted
    # to have moved least. (Indexing mag by its own ranks is a scramble, not a reversal -- tau
    # 0.02 -- which is a fixture bug worth not repeating.)
    order = np.argsort(mag)
    rev = np.empty_like(mag)
    rev[order] = np.sort(mag)[::-1]
    pm = xe + sign * rev
    r = species_change_agreement(xe, xm, xe, pm)
    assert r["direction_skill"] > 0.99, r               # every direction correct
    # ...and the magnitude ordering inverted. This is why the rank statistic uses the ABSOLUTE
    # change: on the signed change tau reads +0.51 here, dominated by the signs being right, and
    # the reversal is invisible.
    assert r["rank_tau"] < -0.9, r


def test_the_noise_floor_excludes_coin_flips_not_signal():
    from src.community_encoder.train_DESK.validate_baselines import species_change_agreement
    rng = np.random.default_rng(3)
    n_sp = 150
    xe = np.full((1, n_sp), 3.0)
    real = np.zeros(n_sp)
    real[:50] = rng.uniform(0.5, 1.5, 50) * np.where(rng.random(50) < 0.5, 1, -1)
    xm = xe + real + rng.normal(scale=0.02, size=n_sp)   # 100 species change by noise only
    # The predictor gets the real 50 exactly and COMMITS to a direction on the noise species too,
    # guessing at random -- which is what a real predictor does. (If it predicted exactly zero for
    # them it would abstain and the threshold would have nothing to remove.)
    pm = xe + real + np.where(real == 0, rng.normal(scale=0.01, size=n_sp), 0.0)
    loose = species_change_agreement(xe, xm, xe, pm, noise_floor_abs=0.0)
    tight = species_change_agreement(xe, xm, xe, pm, noise_floor_abs=0.1)
    assert tight["n_species_scored"] == 50, tight        # only the real movers survive
    assert loose["n_species_scored"] > 100
    assert tight["direction_skill"] > loose["direction_skill"]   # coin flips were diluting it


def test_a_species_absent_from_both_endpoints_is_not_scored():
    from src.community_encoder.train_DESK.validate_baselines import species_change_agreement
    xe = np.array([[0.0, 0.0, 2.0, 0.0] + [1.0] * 20])
    xm = np.array([[0.0, 3.0, 0.0, 0.0] + [2.0] * 20])
    r = species_change_agreement(xe, xm, xe, xm)
    # species 0 and 3 are absent from both; 1 appeared, 2 disappeared, and 20 rose
    assert r["n_species_scored"] == 22, r
def test_a_predictor_of_no_change_abstains_rather_than_scoring_worst():
    """The frozen-at-recent null predicts zero change for every species by construction. Counting
    that as wrong every time read as -1.007 -- 'worse than useless' -- when the honest description
    is that it declines to answer."""
    from src.community_encoder.train_DESK.validate_baselines import species_change_agreement
    rng = np.random.default_rng(0)
    n_sp = 60
    xe = np.full((1, n_sp), 3.0)
    xm = xe + rng.normal(size=n_sp)
    silent = species_change_agreement(xe, xm, xe, xe)      # predicts identical -> zero change
    assert "abstention" in silent["note"], silent
    assert "direction_skill" not in silent                 # no score is offered at all

    # a predictor that commits on only some species is scored on those, and says how often
    partial = xe.copy()
    partial[0, :30] = xe[0, :30] + np.sign(xm[0, :30] - xe[0, :30])
    r = species_change_agreement(xe, xm, xe, partial)
    assert r["n_species_committed"] == 30, r
    assert 0.4 < r["commit_rate"] < 0.6
    assert r["direction_skill"] > 0.9                      # right on every one it committed to
def test_a_comparison_where_every_species_moves_the_same_way_reports_no_skill():
    """The lazy guess is already perfect, so there is nothing for a direction score to
    discriminate. Dividing by what remains of it produced -5e10 on a real fixture."""
    from src.community_encoder.train_DESK.validate_baselines import species_change_agreement
    xe = np.full((1, 40), 2.0)
    xm = xe + 1.0                                    # every species rises
    r = species_change_agreement(xe, xm, xe, xm)
    assert "direction_skill" not in r, r
    assert "no information" in r["note"]
    assert r["share_declining_observed"] == 0.0
