import numpy as np

from scripts.viz.encoder_diagnostics import (
    component_metrics,
    kernel_dimension_curve,
    paired_turnover,
    plot_component_atlas,
    plot_component_fidelity,
    plot_kernel_curve,
    plot_structure,
    plot_turnover,
    structure_retention,
)


def _fixture():
    rows, cols, years = [], [], []
    for year in (2000, 2005, 2010):
        for row in range(3):
            for col in range(4):
                rows.append(row); cols.append(col); years.append(year)
    rows = np.asarray(rows); cols = np.asarray(cols); years = np.asarray(years)
    z = np.column_stack([
        0.4 * rows + 0.1 * (years - 2000),
        0.3 * cols - 0.05 * (years - 2000),
        np.sin(rows + cols) + 0.02 * (years - 2000),
    ]).astype("float32")
    x = np.exp(np.column_stack([z[:, 0] + 1, z[:, 1] + 1, z[:, 2] + 2])).astype("float32")
    return x, z, rows, cols, years


def test_identical_desk_preserves_components_and_structure():
    _, z, rows, cols, years = _fixture()
    metrics = component_metrics(z, z.copy(), rows, cols, years)
    assert np.allclose(metrics[["global_corr", "spatial_corr", "temporal_corr"]], 1)
    assert np.allclose(metrics["nrmse"], 0)
    assert np.allclose(metrics["variance_ratio"], 1)

    retention = structure_retention(z, z.copy(), rows, cols, years)
    assert retention["n_spatial_pairs"] > 0
    assert retention["n_temporal_pairs"] > 0
    assert np.allclose(retention["spatial_ratio"], 1)
    assert np.allclose(retention["temporal_ratio"], 1)


def test_kernel_curve_and_turnover_are_well_formed():
    x, z, rows, cols, years = _fixture()
    curve = kernel_dimension_curve(x, z, z.copy(), [1, 2, 3], seed=4, n_pairs=500)
    assert curve["dimension"].tolist() == [1, 2, 3]
    assert np.allclose(curve["esk_rmse_norm"], curve["desk_rmse_norm"])
    assert np.allclose(curve["desk_vs_esk_corr"], 1)

    r, c, fused, esk, desk = paired_turnover(x, z, z.copy(), rows, cols, years, 2000, 2010)
    assert len(r) == len(c) == len(fused) == 12
    assert np.allclose(esk, desk)


def test_all_diagnostic_figures_render(tmp_path):
    x, z, rows, cols, years = _fixture()
    desk = z + 0.01 * np.cos(np.arange(z.size)).reshape(z.shape)
    data = {"X": x, "Z_esk": z, "Z_desk": desk,
            "rows": rows, "cols": cols, "years": years}
    comp = component_metrics(z, desk, rows, cols, years)
    retention = structure_retention(z, desk, rows, cols, years)
    curve = kernel_dimension_curve(x, z, desk, [1, 2, 3], seed=2, n_pairs=500)

    outputs = [
        tmp_path / "components.png", tmp_path / "structure.png",
        tmp_path / "kernel.png", tmp_path / "turnover.png", tmp_path / "atlas.png",
    ]
    plot_component_fidelity(comp, outputs[0])
    plot_structure(retention, outputs[1])
    plot_kernel_curve(curve, outputs[2])
    plot_turnover(data, 3, 4, 2000, 2010, outputs[3])
    plot_component_atlas(data, 3, 4, 2010, [1, 2, 3], outputs[4])
    assert all(path.stat().st_size > 10_000 for path in outputs)
