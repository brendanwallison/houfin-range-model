import numpy as np
from scipy.optimize import minimize_scalar

from research.exp.local_profile_point import multiplier_step


def test_augmented_constraint_recovers_planted_feasible_answer():
    # Data alone prefers x=3; the diagnostic equality fixes x=1.
    rho, multiplier, previous = 10.,0.,float("inf")
    for _ in range(8):
        objective = lambda x: (x-3.)**2 + multiplier*(x-1.) + .5*rho*(x-1.)**2
        solution = minimize_scalar(objective).x
        residual = solution-1.
        multiplier,rho = multiplier_step(multiplier,rho,residual,previous,.0001)
        previous = abs(residual)
    np.testing.assert_allclose(solution,1.,atol=1e-4)
