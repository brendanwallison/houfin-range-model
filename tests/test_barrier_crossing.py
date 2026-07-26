import numpy as np
import pytest

from src.data.preprocess.great_plains import (
    ZONE_BARRIER, ZONE_EAST, ZONE_WEST, corridor_zones, read_zone_raster,
)
from src.vis.barrier_crossing import (
    annual_operator, crossing_gain, directional_q_contrast, edge_correction_summary,
    low_density_departure_probability,
)


def _fixture(ny=16, nx=36, mdd=330.0, q_east=0.55, q_west=0.55, fmax_barrier=0.8,
             barrier_frac=0.30):
    """Small grid, but the REAL kernel builder and the REAL dispersal operator."""
    import jax.numpy as jnp
    from src.model.build_kernels import build_simulation_struct

    land = np.ones((ny, nx))
    ss = build_simulation_struct(jnp.asarray(land), 27.0, 100.0, mdd, 0.468, 0.468,
                                 [0.0, 0.402 * mdd, 1.002 * mdd, 1e9])
    labels = [str(s) for s in ss["labels"]]
    n_k = len(labels)
    x = np.arange(nx, dtype=float)
    w = float(int(nx * (0.5 - barrier_frac / 2)))
    e = float(int(nx * (0.5 + barrier_frac / 2)))
    zw, zb, ze = corridor_zones(np.full(ny, w), np.full(ny, e), x, ny, nx)
    zones = {"west": zw, "barrier": zb, "east": ze}

    pop = 20.0
    gamma = np.log(2) / (0.20 / pop)
    sa = np.full((ny, nx), 0.62)
    sj = np.full((ny, nx), 0.42)
    fmax = np.full((ny, nx), 3.0)
    fmax[zb] = fmax_barrier
    k = np.full((ny, nx), 0.10)
    q = np.empty((n_k, ny, nx))
    for i, lab in enumerate(labels):
        q[i] = q_east if lab.startswith("to_EAST") else (
            q_west if lab.startswith("to_WEST") else 0.5 * (q_east + q_west))
    a0 = np.zeros((ny, nx))
    j0 = np.zeros((ny, nx))
    a0[zw] = a0[ze] = 0.08
    j0[zw] = j0[ze] = 0.04
    fields = {"Sa": sa, "Sj": sj, "Fmax": fmax, "K": k, "Q": q,
              "a0": a0, "j0": j0, "land": land}
    rows, cols = np.nonzero(land)
    data = {
        "land_rows": jnp.asarray(rows), "land_cols": jnp.asarray(cols),
        "land_mask": jnp.asarray(land),
        "adult_fft_kernel": ss["adult_fft_kernel"],
        "juvenile_fft_kernel_stack": ss["juvenile_fft_kernel_stack"],
        "adult_edge_correction": ss["adult_edge_correction"],
        "juvenile_edge_correction_stack": ss["juvenile_edge_correction_stack"],
        "dispersal_target_fraction": 0.8, "pop_scalar": pop,
    }
    return fields, data, zones, labels, gamma


# --------------------------------------------------------------------- zones

def test_corridor_zones_partition_every_cell_exactly_once():
    """Gap-free and disjoint. A hole would silently absorb dispersing mass."""
    ny, nx = 7, 11
    west_edge = np.array([2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
    east_edge = west_edge + 3.0
    w, b, e = corridor_zones(west_edge, east_edge, np.arange(nx, dtype=float), ny, nx)
    assert np.all(w.astype(int) + b.astype(int) + e.astype(int) == 1)
    # The corridor must follow the per-row tilt, not a single global threshold.
    assert b[0].argmax() < b[-1].argmax()


def test_read_zone_raster_rejects_unknown_codes(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "zones.tif"
    bad = np.full((4, 5), 9, dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", height=4, width=5, count=1,
                       dtype="uint8", crs="EPSG:3857",
                       transform=from_origin(0, 0, 1, 1), nodata=0) as dst:
        dst.write(bad, 1)
    with pytest.raises(ValueError, match="unexpected codes"):
        read_zone_raster(path)


def test_read_zone_raster_rejects_wrong_shape(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "zones.tif"
    ok = np.full((4, 5), ZONE_BARRIER, dtype="uint8")
    ok[:, 0] = ZONE_WEST
    ok[:, -1] = ZONE_EAST
    with rasterio.open(path, "w", driver="GTiff", height=4, width=5, count=1,
                       dtype="uint8", crs="EPSG:3857",
                       transform=from_origin(0, 0, 1, 1), nodata=0) as dst:
        dst.write(ok, 1)
    with pytest.raises(ValueError, match="expected"):
        read_zone_raster(path, expected_shape=(9, 9))


# ------------------------------------------------------------------ operator

def test_departure_probability_collapses_to_a_scalar():
    """At N/K -> 0 the density term is -beta1*tau, so p0 is one number, not a field."""
    latents = {"dispersal_logit_intercept": 2.0, "dispersal_logit_slope": 4.0,
               "dispersal_random": np.zeros(10)}
    p0 = low_density_departure_probability(latents, 0.8)
    assert np.isclose(p0, 1.0 / (1.0 + np.exp(-(2.0 - 4.0 * 0.8))))
    assert 0.0 < p0 < 1.0


def test_operator_is_linearized_against_the_1e9_pad():
    """age_forward.rightpad pads with 1e-9, making the raw operator AFFINE.

    Regression test for the bug this module was built around: an affine operator has
    a non-zero fixed point, so the resolvent iteration floors instead of decaying and
    rho pins at exactly 1.0. The linearized operator must map zero to zero.
    """
    fields, data, zones, _, _ = _fixture()
    step = annual_operator(fields, data, p0=0.23, use_edge_correction=False)
    zero = np.zeros_like(fields["land"])
    a, j = (np.asarray(x) for x in step(zero, zero))
    assert a.max() == 0.0 and j.max() == 0.0

    # And the un-subtracted operator really is affine -- if this ever becomes exactly
    # zero the pad has changed and the subtraction is redundant, not wrong.
    from src.model.age_forward import rightpad_convolution
    import jax.numpy as jnp
    pad_response = np.asarray(
        rightpad_convolution(jnp.zeros_like(jnp.asarray(fields["land"])),
                             jnp.asarray(data["adult_fft_kernel"])))
    assert pad_response.max() > 0.0


# -------------------------------------------------------------- crossing gain

def test_crossing_gain_is_symmetric_when_Q_has_no_direction():
    """THE decisive test: uniform Q + uniform habitat => no directional preference.

    Isolates genuine Z_disp asymmetry from an index-convention error, since Q is
    indexed by ARRIVAL cell while the edge correction is indexed by source cell.
    """
    fields, data, zones, _, _ = _fixture(q_east=0.55, q_west=0.55)
    g = {d: crossing_gain(fields, data, zones, d, p0=0.23, use_edge_correction=False)
         for d in ("east_to_west", "west_to_east")}
    ew, we = g["east_to_west"]["G_horizon"], g["west_to_east"]["G_horizon"]
    assert ew > 0 and we > 0
    assert abs(ew - we) / max(ew, we) < 1e-6


def test_crossing_gain_follows_the_direction_Q_favours():
    """Cheaper westward journeys must raise G(east->west), not lower it."""
    fields, data, zones, _, _ = _fixture(q_east=0.40, q_west=0.70)
    g = {d: crossing_gain(fields, data, zones, d, p0=0.23, use_edge_correction=False)
         for d in ("east_to_west", "west_to_east")}
    assert g["east_to_west"]["G_horizon"] > g["west_to_east"]["G_horizon"]


def test_q_override_removes_the_asymmetry():
    """With Q forced to 1 the same asymmetric landscape must become symmetric."""
    fields, data, zones, _, _ = _fixture(q_east=0.40, q_west=0.70)
    ones = np.ones_like(fields["Q"])
    g = {d: crossing_gain(fields, data, zones, d, p0=0.23, use_edge_correction=False,
                          q_override=ones)
         for d in ("east_to_west", "west_to_east")}
    ew, we = g["east_to_west"]["G_horizon"], g["west_to_east"]["G_horizon"]
    assert abs(ew - we) / max(ew, we) < 1e-6


def test_rho_is_direction_independent_and_below_the_local_eigenvalue():
    """rho is a property of the barrier-restricted operator, not of the direction.

    It must also sit at or below the single-cell dominant eigenvalue, since dispersal
    out of the barrier can only remove mass from a barrier-confined lineage.
    """
    fields, data, zones, _, _ = _fixture(q_east=0.40, q_west=0.70, fmax_barrier=0.8)
    g = {d: crossing_gain(fields, data, zones, d, p0=0.23, use_edge_correction=False)
         for d in ("east_to_west", "west_to_east")}
    rhos = [g[d]["rho"] for d in g]
    assert abs(rhos[0] - rhos[1]) < 1e-3
    sa, sj, fmax, p0 = 0.62, 0.42, 0.8, 0.23
    local = (sa + np.sqrt(sa ** 2 + 4 * (sj * (1 - p0) + 0.7 * p0) * sa * fmax)) / 2
    assert 0.0 < rhos[0] <= local + 1e-6


def test_self_sustaining_barrier_is_reported_not_summed():
    """rho >= 1 must yield G_total = inf and a finite G_horizon, never a silent number.

    A barrier patch with R0 > 1 self-sustains under the Allee-optimistic
    linearization, so the Neumann series genuinely diverges; truncating it would
    report a horizon artifact as if it were the infinite-horizon gain.
    """
    fields, data, zones, _, _ = _fixture(fmax_barrier=3.0)
    g = crossing_gain(fields, data, zones, "east_to_west", p0=0.23,
                      use_edge_correction=False)
    assert g["barrier_self_sustaining"] is True
    assert not np.isfinite(g["G_total"])
    assert np.isfinite(g["G_horizon"])


def test_crossing_gain_raises_on_an_unoccupied_source():
    """0/0 must be an error, not a NaN that propagates into the sweep comparison."""
    fields, data, zones, _, _ = _fixture()
    fields = {**fields, "a0": np.zeros_like(fields["a0"]),
              "j0": np.zeros_like(fields["j0"])}
    with pytest.raises(ValueError, match="no adults establish"):
        crossing_gain(fields, data, zones, "east_to_west", p0=0.23)


def test_bad_direction_is_rejected():
    fields, data, zones, _, _ = _fixture()
    with pytest.raises(ValueError, match="direction must be one of"):
        crossing_gain(fields, data, zones, "north_to_south", p0=0.23)


# ------------------------------------------------------------------ reporting

def test_directional_q_contrast_reads_cohorts_from_labels():
    fields, data, zones, labels, _ = _fixture(q_east=0.40, q_west=0.70)
    c = directional_q_contrast(fields, labels)
    assert np.allclose(c["q_to_east"], 0.40)
    assert np.allclose(c["q_to_west"], 0.70)
    assert np.allclose(c["q_west_minus_east"], 0.30)
    with pytest.raises(ValueError, match="no to_EAST/to_WEST"):
        directional_q_contrast(fields, ["Kernel_0", "Kernel_1"])


def test_edge_correction_summary_reports_share_not_just_extremes():
    """The gating check needs a share below threshold; a bare min is uninformative."""
    fields, data, zones, _, _ = _fixture()
    s = edge_correction_summary(data, zones, fields["land"])
    assert 0.0 <= s["frac_juvenile_edge_correction_below_half"] <= 1.0
    assert 0.0 < s["mean_juvenile_edge_correction_in_barrier"] <= 1.0
    assert 0.0 < s["mean_adult_edge_correction_in_barrier"] <= 1.0
