"""Unexecuted JAX parity checks. Synthetic fixtures are not study evidence."""
import numpy as np
import pytest

pytest.importorskip('jax')
import jax

from bo_vs_es.objectives.models import LOWER, UPPER, physical, predict, rc_voltage
from bo_vs_es.v5_jax import cpu_float64, loss_jax, predict_jax, rc_voltage_jax
from bo_vs_es.v5_plots import saved_curves


def fixture_arrays():
    times = np.array([0., .1, .7, 2., 7., 19., 100., 180., 400.])
    current = np.array([0., 1., 1., -2., 0., .5, -.8, 1.2, 0.])
    ocv = np.linspace(4.1, 3.9, len(times))
    device = cpu_float64()
    return tuple(jax.device_put(x, device) for x in (times, current, ocv))


@pytest.mark.parametrize('tau', [1., 50., 2000.])
def test_scan_matches_native_irregular_time_recurrence(tau):
    times, current, _ = fixture_arrays()
    actual = np.asarray(rc_voltage_jax(times, current, .03, tau).block_until_ready())
    expected = rc_voltage(np.asarray(times), np.asarray(current), .03, tau)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('u', [np.zeros(5), np.array([1., 0., 0., 0., 0.]),
                              np.array([0., 0., -1., 0., 0.])])
def test_voltage_and_loss_preserve_the_original_model(u):
    times, current, ocv = fixture_arrays()
    theta = physical(u)
    expected = predict(np.asarray(times), np.asarray(current), np.asarray(ocv), theta)
    actual = np.asarray(predict_jax(times, current, ocv, jax.device_put(theta, cpu_float64())).block_until_ready())
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    measured = np.asarray(ocv) - .01
    device = cpu_float64()
    arguments = [u, measured, (np.log(LOWER) + np.log(UPPER)) / 2,
                 (np.log(UPPER) - np.log(LOWER)) / 2]
    point, voltage, midpoint, half_range = [jax.device_put(x, device) for x in arguments]
    result = float(loss_jax(point, times, current, ocv, voltage, midpoint, half_range).block_until_ready())
    assert result == pytest.approx(np.mean((expected - measured) ** 2), rel=1e-10, abs=1e-12)


def test_jax_saved_curves_match_numpy_and_keep_best_observations():
    losses = np.array([[.04, .09, .01, .02], [.01, .008, .008, .003]])
    result = saved_curves(losses, 'jax')
    np.testing.assert_allclose(result, saved_curves(losses, 'numpy'), rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(result, axis=1) <= 0)
