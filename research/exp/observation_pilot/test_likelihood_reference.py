import sys
import unittest
import numpy as np
from scipy.stats import nbinom
sys.path.insert(0,'/mnt/c/Dev/houfin-range-model/.research_runtime/observation_packages')
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
from observation_pilot.calibrate import positive_logscore
import pandas as pd


class TruncatedLikelihoodTests(unittest.TestCase):
    def test_library_positive_count_likelihood_matches_nb2_conditioning(self):
        y=np.array([1,2,4,9]);mu=3.;alpha=.4
        model=TruncatedLFNegativeBinomialP(y,np.ones((len(y),1)),truncation=0,p=2)
        actual=model.loglikeobs(np.array([np.log(mu),alpha]))
        n=1/alpha;prob=n/(n+mu)
        reference=nbinom.logpmf(y,n,prob)-nbinom.logsf(0,n,prob)
        np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-12)

    def test_formula_based_validation_uses_public_api(self):
        frame=pd.DataFrame({'count':[1,2,3,4,1,6],'x':[-1,-.5,0,.5,1,2]})
        model=TruncatedLFNegativeBinomialP.from_formula('count ~ x',frame,truncation=0,p=2)
        # A minimal result-like object isolates the scoring API from optimization.
        from types import SimpleNamespace
        params=np.array([np.log(3),.1,.4])
        result=SimpleNamespace(model=model,params=params)
        actual=positive_logscore(result,'count ~ x',frame.iloc[:3])
        reference=TruncatedLFNegativeBinomialP.from_formula('count ~ x',frame.iloc[:3],truncation=0,p=2).loglikeobs(params)
        np.testing.assert_allclose(actual,reference)


if __name__=='__main__':unittest.main()
