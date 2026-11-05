import unittest
import numpy as np
from hmc_geometry_math import lanczos,spectrum_report


class GeometryTests(unittest.TestCase):
    def test_indefinite_spectrum_against_exact_eigendecomposition(self):
        rng=np.random.default_rng(41);q,_=np.linalg.qr(rng.normal(size=(8,8)))
        values=np.array([-3.,-.1,.001,.1,1.,2.,5.,100.]);h=(q*values)@q.T
        result=lanczos(lambda v:h@v,8,8)
        np.testing.assert_allclose(result['values'],values,atol=1e-12)
        self.assertLess(result['orthogonality_error'],1e-12)
        self.assertEqual(spectrum_report(result)['observed_negative_ritz_values'],2)

    def test_scaling_can_remove_unit_conditioning(self):
        h=np.diag([1e-6,1.,1e6]);scale=np.array([1e3,1.,1e-3])
        result=lanczos(lambda v:scale*(h@(scale*v)),3,3)
        np.testing.assert_allclose(result['values'],1.,atol=1e-12)

    def test_readonly_hvp_result(self):
        h=np.diag([1.,2.,3.])
        def op(v):
            out=h@v;out.flags.writeable=False;return out
        np.testing.assert_allclose(lanczos(op,3,3)['values'],[1.,2.,3.],atol=1e-12)


if __name__=='__main__':unittest.main()
