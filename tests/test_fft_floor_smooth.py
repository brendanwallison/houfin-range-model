"""numerics.fft_floor_smooth: the soft FFT floor removes sub-resolution values but keeps the output continuous."""
import jax
import jax.numpy as jnp
import numpy as np

import src.model.age_forward as af


def _conv(pop):
    # float32 throughout: another test may have switched x64 on globally, and a float64 kernel would promote the
    # convolution to float64, whose floor sits far below the float32-scale threshold scanned here.
    k = jnp.zeros((15, 15), jnp.float32).at[0, 0].set(0.6).at[0, 1].set(0.2).at[1, 0].set(0.2)
    return af.rightpad_convolution(pop, jnp.fft.fft2(k))


def test_smooth_floor_is_continuous_and_still_kills_the_floor():
    old = (af.FFT_FLOOR_ULPS, af.FFT_FLOOR_SMOOTH, af.FFT_PAD_VALUE)
    try:
        af.FFT_PAD_VALUE, af.FFT_FLOOR_ULPS = 0.0, 10.0
        pop = jnp.zeros((8, 8), jnp.float32).at[3, 3].set(1.0)
        # Scan the input through the threshold. The hard floor steps by ~tol somewhere; the smooth one only moves by
        # small increments (FFT round-off is ~.1 tol, so compare against tol, not zero).
        tol = 10 * np.finfo(np.float32).eps * 0.6
        a0 = tol / 0.6                                        # output = .6 x input crosses tol here
        grid = np.linspace(0.3 * a0, 2.5 * a0, 400)
        for smooth in (False, True):
            af.FFT_FLOOR_SMOOTH = smooth
            vals = np.array([float(_conv(pop.at[5, 5].set(a))[5, 5]) for a in grid])
            jump = np.max(np.abs(np.diff(vals)))
            if smooth:
                assert jump < 0.3 * tol
                f = lambda a: _conv(pop.at[5, 5].set(a)).sum()
                assert np.isfinite(float(jax.grad(f)(a0)))
            else:
                assert jump > 0.7 * tol                       # the hard version steps
        af.FFT_FLOOR_SMOOTH = True
        out = np.asarray(_conv(pop))
        assert out.min() >= 0.0                              # no negative round-off survives
        assert np.all(out[out < 1e-6] < 1e-9)                # sub-resolution values are effectively zero
        np.testing.assert_allclose(out[3, 3], 0.6, rtol=1e-5)   # real mass untouched
    finally:
        af.FFT_FLOOR_ULPS, af.FFT_FLOOR_SMOOTH, af.FFT_PAD_VALUE = old
