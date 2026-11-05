import tempfile
from pathlib import Path
import pickle
import unittest
import numpy as np
from full_pathology_audit import observation_context


class ObservationContextTests(unittest.TestCase):
    def test_modern_positive_records_and_route_count_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            metadata={'years':np.array([2009,2010,2020,2021]),'obs_time_indices':np.arange(4),
                      'obs_rows':np.zeros(4,int),'obs_cols':np.array([0,1,0,1]),
                      'observed_results':np.array([20,0,2,4]),'pop_scalar':20.}
            with (p/'metadata.pkl').open('wb') as f:pickle.dump(metadata,f)
            np.savez(p/'fields.npz',K_modern=np.array([[.001,.1]]),
                     allee_dead_mask=np.array([[True,False]]),lam_fundamental_modern=np.array([[1.2,.8]]))
            result=observation_context(str(p),p/'fields.npz',{})
            self.assertEqual(result['positive_modern_records'],2)
            self.assertEqual(result['positive_record_share_K_below_005_route_counts'],.5)
            self.assertEqual(result['positive_record_share_allee_dead'],.5)
            self.assertEqual(result['positive_record_share_fundamental_sink'],.5)


if __name__=='__main__':unittest.main()
