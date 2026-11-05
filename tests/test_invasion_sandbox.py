import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "research", "exp", "invasion_sandbox")))
import proto  # noqa: E402


@pytest.mark.parametrize("ndim,dx,extent", [(1, 5.0, 6000.0), (2, 10.0, 3000.0)])
def test_kernel_mass_and_mean(ndim, dx, extent):
    g = proto.grid(ndim, dx, extent)
    for a, b, stay in [(.859, 76.893, .22), (.468, 41.056, .27)]:
        k = proto.weibull_kernel(g, a, b, stay=stay)
        assert abs(k.sum() - 1) < 1e-12
        assert k.min() >= 0
        # discretised movers' mean within 6% of the analytic Veit & Lewis mean (tail truncation at the domain edge)
        assert abs(proto.kernel_mean_distance(g, k, stay) / proto.weibull_mean(a, b) - 1) < .06


def test_paper_means():
    assert abs(proto.weibull_mean(.468, 41.056) - 329.3) < 1.
    assert abs(proto.weibull_mean(.859, 76.893) - 98.3) < 1.


def _small(ndim=1):
    g = proto.grid(ndim, 10.0, 1500.0 if ndim == 1 else 600.0)
    kA = proto.weibull_kernel(g, .859, 76.893, stay=.22)
    kJ = proto.weibull_kernel(g, .468, 41.056, stay=.27)
    N0 = proto.release_patch(g, .5, 10.0)
    return g, kA, kJ, N0


def test_vl_lag_ordering():
    g, kA, kJ, N0 = _small(1)
    lag = {}
    for r in (0.0, .15, .25):
        h, _ = proto.simulate('vl', proto.vl_params(r=r), g, kA, kJ, years=41, N0=N0)
        lag[r] = proto.lag_and_speeds(h)['lag_year'] or 99
    assert lag[0.0] < lag[.15] < lag[.25]


@pytest.mark.parametrize("mech", [{'local_sink': (0.0, 5.0)}, {'establish': 1e-12}, {'floor': 0.0},
                                   {'departure': 'power', 'theta': 1.0}])
def test_mechanism_off_is_identity(mech):
    g, kA, kJ, N0 = _small(1)
    P = proto.hofi_params(Sa=.6, Sj=.4, Fmax=3., K=4., n50=.01)
    h0, s0 = proto.simulate('hofi', P, g, kA, kJ, years=25, N0=N0, keep=(24,))
    h1, s1 = proto.simulate('hofi', P, g, kA, kJ, years=25, N0=N0, mech=mech, keep=(24,))
    np.testing.assert_allclose(s1[24], s0[24], rtol=1e-12, atol=1e-15)


def test_equilibrium_at_K():
    # a full uniform field at K stays at K (c makes lambda = 1 at K; Allee off)
    g = proto.grid(1, 10.0, 500.0)
    k0 = np.zeros_like(g['x']); k0[len(k0) // 2] = 1.
    P = proto.hofi_params(Sa=.55, Sj=.3, Fmax=6., K=4., n50=float('nan'))
    # at the post-breeding census the stable composition is adults B and young B*F(K)
    Fk = P['Fmax'] / (1 + P['c'])
    A, J = np.full_like(g['x'], 4. / (1 + Fk)), np.full_like(g['x'], 4. * Fk / (1 + Fk))
    D = A + J
    B, young = proto.hofi_step((A, J), P, k0, k0, {}, 0)
    # breeders after survival: Sa A + Sj J; with D = K the brake gives F(K) = Fmax/(1+c)
    np.testing.assert_allclose(B, P['Sa'] * A + P['Sj'] * J)
    np.testing.assert_allclose(young / B, Fk)
    assert np.allclose(D, 4.)


def test_establishment_reduces_colonisation_only():
    g, kA, kJ, N0 = _small(1)
    P = proto.hofi_params(Sa=.6, Sj=.4, Fmax=3., K=4., n50=float('nan'))
    h0, _ = proto.simulate('hofi', P, g, kA, kJ, years=20, N0=N0)
    h1, _ = proto.simulate('hofi', P, g, kA, kJ, years=20, N0=N0, mech={'establish': 50.})
    assert h1[19]['radius_km'] <= h0[19]['radius_km']
    assert h1[19]['total'] <= h0[19]['total']
