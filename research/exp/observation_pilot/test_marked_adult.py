import importlib.util
from pathlib import Path
import unittest
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from observation_pilot.marked_adult import adult_step


class MarkedAdultTests(unittest.TestCase):
    def test_native_departure_and_destination_survival_against_enumeration(self):
        path=Path('/mnt/e/Datasets/houfin/grid54/runs/step33_nestcbc_s41/code/src/model/age_forward.py')
        if not path.exists():self.skipTest('frozen source unavailable')
        spec=importlib.util.spec_from_file_location('audited_marked_adult_forward',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        # The current candidate declares zero FFT padding and no floor. Private
        # imported test module only; no production globals/config are changed.
        module.FFT_PAD_VALUE=0.;module.FFT_FLOOR_ULPS=None
        land=np.array([[1,1,0],[1,1,1],[0,1,1]],bool);cells=np.argwhere(land);n=len(cells)
        kernel=np.zeros((5,5));kernel[0,0]=.3;kernel[0,1]=.3;kernel[0,-1]=.2;kernel[1,0]=.2
        raw=np.array([[kernel[(v-y)%5,(u-x)%5] for v,u in cells] for y,x in cells])
        edge=raw.sum(1);edge_grid=np.ones(land.shape);edge_grid[land]=edge
        ambient=np.zeros(land.shape);ambient[land]=np.linspace(.1,.9,n)
        native=np.ones(land.shape);native[land]=np.linspace(.4,1,n)
        survival=np.zeros(land.shape);survival[land]=np.linspace(.3,.8,n)
        p=np.clip(ambient[land]-np.logaddexp(0,10*(ambient[land]-1))/10,1e-6,1)*native[land]
        expected=((np.diag(1-p)+p[:,None]*raw/(edge[:,None]+1e-6))*survival[land][None,:])
        fft=jnp.asarray(np.fft.fft2(kernel))
        kwargs=dict(dispersal_logit_intercept=0.,dispersal_logit_slope=0.,target_fraction=.8,
          adult_edge_correction=jnp.asarray(edge_grid),juvenile_edge_correction_stack=jnp.asarray(edge_grid[None]),
          adult_fft_kernel=fft,juvenile_fft_kernel_stack=fft[None],Q_grid=jnp.ones((1,)+land.shape),
          veit_departure=True,departure_scale=jnp.asarray(native),eps=1e-6)
        actual=[]
        for y,x in cells:
            alive=np.zeros(land.shape);alive[y,x]=1.
            result=adult_step(jnp.asarray(alive),jnp.asarray(ambient),jnp.ones(land.shape),
               jnp.asarray(survival),jnp.asarray(land),module.dispersal_step_age_structured,kwargs)
            actual.append(np.asarray(result['alive'])[land])
            self.assertAlmostEqual(float(result['mass_balance_error']),0.,places=12)
            # Numerical/edge movement loss is NOT misclassified as a reported death.
            self.assertGreater(float(result['movement_mass_loss']),0.)
            self.assertAlmostEqual(float(result['alive'].sum()+result['deaths_by_destination'].sum()+result['movement_mass_loss']),1.,places=12)
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)


if __name__=='__main__':unittest.main()
