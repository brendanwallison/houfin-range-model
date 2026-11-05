import ast
from pathlib import Path
import unittest
import numpy as np
from observation_pilot.census import breeding_grid


class CensusTests(unittest.TestCase):
    def test_exact_frozen_source_mapping(self):
        source=Path('/mnt/e/Datasets/houfin/grid54/runs/step33_nestcbc_s41/code/src/model/age_priors.py')
        if not source.exists():self.skipTest('frozen source unavailable')
        tree=ast.parse(source.read_text())
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='OBS_BREEDING')
        statements=[n for n in branch.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['breeders','densities_obs']]
        rng=np.random.default_rng(41);total=rng.uniform(0,3,(5,2,3));young=rng.uniform(0,1,(5,2,3));initial=rng.uniform(0,2,(2,3))
        t=np.array([0,1,3,4]);r=np.array([0,1,1,0]);c=np.array([1,0,2,2]);phi=.2
        env={'jnp':np,'t_idx':t,'rows':r,'cols':c,'densities':total,'Nj_grid':young,
             'initpop_seeded':initial,'phi':phi,'data':{'pop_scalar':20.}}
        exec(compile(ast.Module(body=statements,type_ignores=[]),str(source),'exec'),env)
        actual=breeding_grid(total,young,phi,20.,initial)[t,r,c]
        np.testing.assert_array_equal(actual,env['densities_obs'])

    def test_does_not_silently_substitute_total_or_first_year(self):
        total=np.array([[10.],[20.],[30.]]);young=np.array([[1.],[2.],[3.]])
        mapped=breeding_grid(total,young,.5,1.)
        self.assertTrue(np.isnan(mapped[0,0]))
        self.assertEqual(mapped[1,0],11.)
        self.assertNotEqual(mapped[1,0],total[1,0])


if __name__=='__main__':unittest.main()
