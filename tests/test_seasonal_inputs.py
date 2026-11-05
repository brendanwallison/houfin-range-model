import pickle

import numpy as np

from research.exp.final_seasonal_inputs import derive
from src.community_encoder.cube_contract import export_contract, validate_model_feature_contract


def test_seasonal_derivation_preserves_tensors_and_observations(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    contract = export_contract(2, 'summer_basis', 2, [
        {'label': 'winter_half', 'private': 2, 'private_only': True, 'identity': True,
         'coord_target': {'basis_dir': 'winter_basis', 'path': 'winter_target.npz'}}])
    contract.update(feature_prior='isotropic', source_latent_dim=4, truncation='none')
    rng = np.random.default_rng(123)
    z = rng.normal(size=(3, 5, 4)).astype('float32')
    # A fixed spatial path operator acts on each coordinate independently.
    operator = rng.uniform(size=(5, 5)).astype('float32')
    disp = np.einsum('ij,tjm->tim', operator, z)[:, :, None, :]
    z.tofile(source / 'z.dat'); disp.tofile(source / 'disp.dat')
    meta = dict(time=3, N_land=5, K=1, M=4, z_gathered_path='z.dat',
                z_disp_gathered_path='disp.dat', z_kernel_contract=contract,
                observed_results=np.arange(7), years=np.arange(2000, 2003))
    with (source / 'metadata.pkl').open('wb') as f:
        pickle.dump(meta, f)
    for name in ('obs_route_ids.npz', 'subcell_strata.npz'):
        np.savez(source / name, values=np.arange(4))
    for season, sl in [('summer', slice(0, 2)), ('winter_half', slice(2, 4))]:
        out = tmp_path / season; derive(source, out, season)
        with (out / 'metadata.pkl').open('rb') as f:
            new = pickle.load(f)
        validate_model_feature_contract(new['z_kernel_contract'], 2)
        actual = np.memmap(out / new['z_disp_gathered_path'], dtype='float32',
                           shape=(3, 5, 1, 2), mode='r')
        np.testing.assert_array_equal(actual, disp[..., sl])
        np.testing.assert_allclose(actual[:, :, 0], np.einsum('ij,tjm->tim', operator, z[..., sl]), atol=1e-6)
        np.testing.assert_array_equal(new['observed_results'], meta['observed_results'])
        np.testing.assert_array_equal(new['years'], meta['years'])
        assert new['z_kernel_contract']['kernel'] == 'ruzicka'
        assert new['z_kernel_contract']['feature_blocks'][0]['name'] == season
        assert (out / 'DERIVED_DONE.json').exists()
