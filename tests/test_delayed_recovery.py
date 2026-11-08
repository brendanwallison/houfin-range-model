"""disease_prior.recovery: the default (exponential) is the historical formula exactly; "delayed" is host adaptation
after a learned delay (RECOVERY-LIT, 2026-11-05)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)    # the model runs in float64; exactness is checked there

from src.model.disease_onset import recovery_fraction, resolve_recovery_spec


def test_default_is_exponential_and_identical_to_the_historical_formula():
    spec = resolve_recovery_spec({})
    assert spec == {"form": "exponential"}
    age = np.linspace(0, 40, 81)
    for rec, tau in ((.38, 12.), (.01, 18.), (.9, 3.)):
        new = np.asarray(recovery_fraction(age, rec, tau, spec))
        old_jax = np.asarray(rec * (-jax.numpy.expm1(-jax.numpy.asarray(age) / tau)))   # the replaced in-model expression
        np.testing.assert_array_equal(new, old_jax)
        np.testing.assert_allclose(new, rec * (-np.expm1(-age / tau)), rtol=1e-14, atol=0)


def test_delayed_shape():
    spec = resolve_recovery_spec({"recovery": {"form": "delayed"}})
    assert spec == {"form": "delayed", "delay_median_years": 22.0, "delay_log_scale": 0.2, "width_years": 3.0}
    age = np.linspace(0, 60, 601)
    r = np.asarray(recovery_fraction(age, .5, 12., spec, delay=22.))
    assert r[0] == 0.0
    assert np.all(np.diff(r) >= -1e-12)                       # monotone
    assert r[np.searchsorted(age, 10.)] < .01                 # plateau: < 2% of rec ~4 widths before the delay
    assert abs(r[np.searchsorted(age, 22.)] - .25) < .01      # about half of rec at the delay
    assert abs(r[-1] - .5) < 1e-4                             # saturates at rec
    assert np.all(r <= .5 + 1e-12)


@pytest.mark.parametrize("bad", [{"form": "logistic"}, {"form": "delayed", "width_years": 0},
                                 {"form": "delayed", "delay_median_years": -1}])
def test_bad_specs_raise(bad):
    with pytest.raises(ValueError):
        resolve_recovery_spec({"recovery": bad})
