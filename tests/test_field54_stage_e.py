import json
from types import SimpleNamespace

import numpy as np

import research.exp.z_repair_evaluate as zre
import research.exp.z_assim as za


def test_block_bootstrap_size_is_a_module_setting_default_six():
    cells = np.array([[0, 0], [0, 4], [4, 0], [5, 5]])
    assert zre.BLOCK_CELLS == 6
    assert zre.weights_for_cells(cells)[1] == 1
    old = zre.BLOCK_CELLS
    try:
        zre.BLOCK_CELLS = 3
        assert zre.weights_for_cells(cells)[1] == 4
    finally:
        zre.BLOCK_CELLS = old


def test_export_contract_copies_the_d_cube_meta_and_adds_field_provenance(tmp_path):
    cube = tmp_path/'cubes'/'backcast40'/'desk_interp_s1'
    cube.mkdir(parents=True)
    (cube/'cube_meta.json').write_text(json.dumps({'latent_dim': 128, 'kernel': 'ruzicka_sum'}))
    d = SimpleNamespace(control='desk_interp_s1')
    c = za.cube_meta(tmp_path, d, 'field_d_interp', {'summer': 'interp'})
    assert c['latent_dim'] == 128 and c['assim_arm']['arm'] == 'field_d_interp'
    assert c['assim_arm']['summer'] == 'interp' and c['source_cube_dir'].endswith('desk_interp_s1')
    assert za.cube_meta(tmp_path, SimpleNamespace(control='missing'), 'x', {}) is None


def test_rbf_basis_is_in_km_not_cells():
    from research.exp.z_repair_models import GaussianRegionalBasis
    xy = np.array([[0., 0.], [200., 0.]])
    b = GaussianRegionalBasis(xy, n_centers=2, scale_km=200.)
    v = b(np.array([[0., 0.]]))
    # one constant + two Gaussians; the far centre is exactly one width (200 km) away
    assert np.isclose(sorted(v[0, 1:])[0], np.exp(-.5))


def test_inner_choice_keeps_the_27km_default_unless_clearly_better():
    from research.exp.field54_stage_e import select, inner_grid
    g = inner_grid()
    assert len(g) == 15 and ['field_d', {'lam_lap': 1.0, 'lam_u': 1e-3}] in g
    rows = [{'arm': 'control', 'cfg': None, 'regional': .9}] + \
           [{'arm': 'field_d', 'cfg': c, 'regional': .5} for _, c in g]
    rows[-1]['regional'] = .504                     # within the tie band
    assert select(rows)[0] == {'lam_lap': 1.0, 'lam_u': 1e-3}
    rows[-1]['regional'] = .52
    assert select(rows)[0] == {'lam_lap': 4.0, 'lam_u': 1e-2}
