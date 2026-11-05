import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,'/mnt/c/Dev/houfin-range-model/.research_runtime/observation_packages')
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
from statsmodels.genmod.families.links import CLogLog
from scipy.stats import nbinom
from observation_pilot.population_link import align,paired_blocks,encounter_intercept


class PopulationLinkTests(unittest.TestCase):
    def test_cell_year_alignment_rejects_unknown_instead_of_clipping(self):
        state={'land_rows':np.array([1,3]),'land_cols':np.array([2,4]),'years':np.array([2010,2011])}
        f=pd.DataFrame({'row':[3,1,2,1],'col':[4,2,2,2],'year':[2011,2010,2010,2009]})
        cells,years,keep=align(f,state)
        np.testing.assert_array_equal(cells,[1,0,-1,0])
        np.testing.assert_array_equal(years,[1,0,0,-1])
        np.testing.assert_array_equal(keep,[True,True,False,False])

    def test_poisson_encounter_link_and_scale_gauge(self):
        a=np.array([.1,1.,10.]);q=.2
        expected=-np.expm1(-q*a)
        np.testing.assert_allclose(CLogLog().inverse(np.log(a)+np.log(q)),expected)
        np.testing.assert_allclose(CLogLog().inverse(np.log(a*20)+np.log(q/20)),expected)

    def test_initialization_matches_training_rate_with_offset(self):
        offset=np.log(np.geomspace(1e-5,10,100))
        b=encounter_intercept(offset,.05)
        self.assertAlmostEqual(float(np.mean(-np.expm1(-np.exp(b+offset)))),.05,places=12)

    def test_count_validation_retains_population_offset(self):
        a=np.array([.2,1.,3.]);q=2.;alpha=.4;y=np.array([1,2,6])
        f=pd.DataFrame({'count':y})
        model=TruncatedLFNegativeBinomialP.from_formula('count ~ 1',f,offset=np.log(a),truncation=0,p=2)
        actual=model.loglikeobs(np.array([np.log(q),alpha]))
        n=1/alpha;prob=n/(n+q*a)
        expected=nbinom.logpmf(y,n,prob)-nbinom.logsf(0,n,prob)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_bootstrap_is_paired_by_block(self):
        result=paired_blocks(np.full(5,.2),np.array(['a','a','a','b','b']))
        self.assertEqual(result['blocks'],2)
        np.testing.assert_allclose(result['interval90'],[.2,.2])


if __name__=='__main__':unittest.main()
