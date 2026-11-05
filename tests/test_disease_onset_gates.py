import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pytest

from src.model.disease_onset import onset_gate, resolve_onset_spec
from src.model import age_fields as af


def test_default_is_historical_sigmoid_exactly():
    x = jnp.linspace(-20., 20., 501)
    np.testing.assert_array_equal(onset_gate(x, 1.3, resolve_onset_spec({})), jnn.sigmoid(x / 1.3))


@pytest.mark.parametrize("form,start,end", [("compact_start", 0., 3.), ("compact_midpoint", -1.5, 1.5)])
def test_compact_bounds_plateaus_and_boundary_gradients(form, start, end):
    spec = resolve_onset_spec({"onset_gate": {"form": form, "duration_years": 3.}})
    f = lambda t: onset_gate(t, 1., spec)
    x = jnp.linspace(start - 10., end + 10., 2001)
    y = np.asarray(f(x))
    assert np.all(y >= 0.) and np.all(y <= 1.)
    assert np.all(np.diff(y) >= -1e-7)
    assert float(f(start - 1.)) == 0. and float(f(end + 1.)) == 1.
    for t in (start - 1., start, end, end + 1.):
        assert abs(float(jax.grad(f)(t))) < 1e-6
        assert abs(float(jax.grad(jax.grad(f))(t))) < 1e-5
    assert np.isfinite(np.asarray(jax.vmap(jax.grad(f))(x))).all()


def test_fixed_sigmoid_ignores_fitted_tau():
    spec = resolve_onset_spec({"onset_gate": {"form": "sigmoid", "tau_years": .35}})
    np.testing.assert_array_equal(onset_gate(jnp.arange(4.), 99., spec), jnn.sigmoid(jnp.arange(4.) / .35))


def test_matched_control_has_same_midpoint_and_ten_ninety_width():
    compact = {"form": "compact_midpoint", "duration_years": 3.}
    sigmoid = {"form": "sigmoid", "tau_years": .3459321582212081}
    ages = jnp.array([-.7600906401346002, 0., .7600906401346002])
    for spec in (compact, sigmoid):
        np.testing.assert_allclose(onset_gate(ages, 99., spec), [.1, .5, .9], atol=1e-6)


def test_mortality_gate_preserves_chronic_disease_and_recovery(monkeypatch):
    monkeypatch.setattr(af, "DISEASE_ONSET_SPEC", {"form": "compact_start", "duration_years": 3.})
    disease = dict(mu_sev=0., b_late=0., onset_decades=jnp.zeros(2), b_native=0.,
                   native_shape=jnp.zeros(2), sev_basis=jnp.zeros((1, 2)), w_sev=jnp.zeros(1),
                   tau=1., rec=.4, tau_rec=12., ceiling=.5)
    onset = jnp.zeros(2)
    np.testing.assert_array_equal(af.disease_mortality_amplitude(disease, -1., onset), 0.)
    late = af.disease_mortality_amplitude(disease, 300., onset)
    np.testing.assert_allclose(late, .5 * .5 * (1. - .4), rtol=1e-6)
    assert np.all(np.asarray(af.disease_mortality_amplitude(disease, 3., onset)) > late)


@pytest.mark.parametrize("gate", [
    {"form": "other"}, {"form": "compact_start", "duration_years": 0.},
    {"form": "compact_start", "duration_years": float("nan")},
    {"form": "sigmoid", "tau_years": -1.},
])
def test_invalid_gate_rejected(gate):
    with pytest.raises(ValueError):
        resolve_onset_spec({"onset_gate": gate})
