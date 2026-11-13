"""Optional CPU JAX implementation of the existing battery equations.

This module changes the numerical backend, not the model or optimisation rule.
Float64 and explicit CPU placement are configured before creating arrays.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np


def cpu_float64():
    jax.config.update('jax_enable_x64', True)
    return jax.devices('cpu')[0]


@jax.jit
def rc_voltage_jax(times, current, resistance, tau):
    dt = jnp.diff(times)
    decay = jnp.exp(-dt / tau)
    drive = -jnp.expm1(-dt / tau) * resistance * current[:-1]

    def transition(previous, inputs):
        attenuation, forcing = inputs
        voltage = attenuation * previous + forcing
        return voltage, voltage

    _, history = jax.lax.scan(transition, jnp.zeros((), dtype=times.dtype), (decay, drive))
    return jnp.concatenate((jnp.zeros((1,), dtype=times.dtype), history))


@jax.jit
def predict_jax(times, current, ocv, theta):
    voltage = ocv - theta[0] * current
    for index in range(1, theta.shape[0], 2):
        voltage = voltage - rc_voltage_jax(times, current, theta[index], theta[index + 1])
    return voltage


@jax.jit
def loss_jax(u, times, current, ocv, measured, log_midpoint, log_half_range):
    theta = jnp.exp(log_midpoint + u * log_half_range)
    residual = predict_jax(times, current, ocv, theta) - measured
    return jnp.mean(residual ** 2)


@jax.jit
def incumbent_rmse_mv(losses):
    """Vectorised saved-trace calculation; no new battery evaluations."""
    return jax.vmap(lambda row: 1000 * jnp.sqrt(
        jax.lax.associative_scan(jnp.minimum, row)))(losses)


class JaxBattery:
    def __init__(self, reference):
        from bo_vs_es.objectives.models import LOWER, UPPER
        self.device = cpu_float64()
        # Use the original preprocessing, OCV correction and parameter transform.
        self.arguments = tuple(jax.device_put(np.asarray(value, dtype=np.float64), self.device)
            for value in (reference.time, reference.current, reference.ocv, reference.measured,
                          (np.log(LOWER) + np.log(UPPER)) / 2,
                          (np.log(UPPER) - np.log(LOWER)) / 2))

    def __call__(self, u):
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (5,) or not np.isfinite(u).all():
            raise ValueError('Expected five finite dimensionless parameters')
        result = loss_jax(jax.device_put(u, self.device), *self.arguments)
        # Synchronise the CPU dispatch so objective timings include execution.
        return float(result.block_until_ready())


def prepare_objective():
    """Compile and check losses before any timed optimisation trajectory.

    Eleven fixed physical points, including the surface, require 11 evaluations
    per backend. These 22 setup evaluations are separate from search budgets.
    """
    from bo_vs_es.objectives.models import Battery
    start = time.perf_counter()
    reference = Battery()
    objective = JaxBattery(reference)
    points = np.vstack([np.zeros(5), np.eye(5), -np.eye(5)])
    report = dict(status='running', numpy_calls=0, jax_calls=0)
    try:
        expected, actual = [], []
        for point in points:
            report['numpy_calls'] += 1
            expected.append(reference(point))
            report['jax_calls'] += 1
            actual.append(objective(point))
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError('Nonfinite loss during objective parity check')
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=False)
        report.update(status='passed', max_absolute_loss_gap=float(np.max(np.abs(
            np.asarray(actual) - expected))), rtol=1e-10, atol=1e-12)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        if isinstance(exc, KeyboardInterrupt):
            exc.report = report
            raise
        raise ObjectivePreparationError(report) from exc
    finally:
        report['seconds'] = time.perf_counter() - start
    return objective, report


class ObjectivePreparationError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('JAX/NumPy parity failed; no optimisation trajectory started')
