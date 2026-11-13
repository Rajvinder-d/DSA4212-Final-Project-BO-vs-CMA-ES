"""Exact RC transitions, fixed physical coordinates, and quadratic controls."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd

from bo_vs_es import ROOT

LOWER = np.array([.001, .0001, 1., .0001, 1.])
UPPER = np.array([.2, .2, 2000., .2, 2000.])
PARAMETERS = ["R0_ohm", "R1_ohm", "tau1_s", "R2_ohm", "tau2_s"]


def physical(u):
    return np.exp((np.log(LOWER) + np.log(UPPER)) / 2 + np.asarray(u) * (np.log(UPPER) - np.log(LOWER)) / 2)


def dimensionless(theta):
    return (2 * np.log(theta) - np.log(LOWER) - np.log(UPPER)) / (np.log(UPPER) - np.log(LOWER))


def rc_voltage(time, current, resistance, tau):
    """Zero initial overpotential; I[k-1] held over [t[k-1], t[k])."""
    dt = np.diff(time)
    decay = np.exp(-dt / tau)
    drive = -np.expm1(-dt / tau) * resistance * current[:-1]
    output = np.zeros(len(time))
    # Native irregular sample times are preserved; no resampling approximation.
    for k in range(len(dt)):
        output[k + 1] = decay[k] * output[k] + drive[k]
    return output


def predict(time, current, ocv, theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape not in [(3,), (5,)] or np.any(theta <= 0) or not np.isfinite(theta).all():
        raise ValueError("Expected three or five finite, positive circuit parameters")
    result = np.asarray(ocv) - theta[0] * np.asarray(current)
    for j in range(1, len(theta), 2):
        result = result - rc_voltage(time, current, theta[j], theta[j + 1])
    return result


class Battery:
    def __init__(self, label="dst80", corrected=True, root=ROOT):
        self.label = label
        self.trace = pd.read_csv(root / "data/processed" / f"{label}.csv")
        curve = pd.read_csv(root / "data/processed/ocv.csv")
        self.manifest = json.loads((root / "data/processed/manifest.json").read_text())
        soc = self.trace.soc.to_numpy()
        if soc.min() < curve.soc.min() or soc.max() > curve.soc.max():
            raise ValueError("OCV extrapolation forbidden")
        self.time = self.trace.time_s.to_numpy()
        self.current = self.trace.current_a.to_numpy()
        self.measured = self.trace.voltage_v.to_numpy()
        self.ocv = np.interp(soc, curve.soc, curve.ocv_v)
        if corrected:
            self.ocv += self.manifest["calibration_rest_offset_v"]

    def prediction(self, theta):
        return predict(self.time, self.current, self.ocv, theta)

    def residual(self, u):
        return self.prediction(physical(u)) - self.measured

    def __call__(self, u):
        return float(np.mean(self.residual(u) ** 2))


class Quadratic:
    def __init__(self, condition=100., dim=5):
        self.weights = np.geomspace(1, condition, dim)
        self.shift = np.full(dim, .25 / np.sqrt(dim))
        self.hessian = 2 * np.diag(self.weights) / self.weights.sum()

    def __call__(self, u):
        return float(np.sum(self.weights * (np.asarray(u) - self.shift) ** 2) / self.weights.sum())


def rotation(seed, dim=5):
    if seed is None:
        return np.eye(dim)
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


@dataclass
class Domain:
    kind: str
    q: np.ndarray

    def __post_init__(self):
        if self.kind not in ("ball", "box") or not np.allclose(self.q.T @ self.q, np.eye(len(self.q)), atol=1e-12):
            raise ValueError("Expected ball/box and an orthogonal matrix")

    def to_base(self, z):
        return np.asarray(z) @ self.q.T

    def from_base(self, u):
        return np.asarray(u) @ self.q

    def feasible(self, z, tolerance=1e-9):
        u = self.to_base(z)
        return bool(np.linalg.norm(u) <= 1 + tolerance if self.kind == "ball" else np.max(np.abs(u)) <= 1 + tolerance)

    def project(self, z):
        u = self.to_base(z)
        u = u / max(1., np.linalg.norm(u)) if self.kind == "ball" else np.clip(u, -1, 1)
        return self.from_base(u)

    def sample(self, rng, n):
        d = len(self.q)
        if self.kind == "box":
            u = rng.uniform(-1, 1, (n, d))
        else:
            u = rng.normal(size=(n, d))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            u *= rng.random((n, 1)) ** (1 / d)
        return self.from_base(u)
