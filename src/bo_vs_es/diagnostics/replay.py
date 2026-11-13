"""Fixed-history scalar-GP replays matching v1 settings without editing v1."""
import time
import warnings

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll_scipy
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.settings import validate_input_scaling
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_vs_es.diagnostics.core import uniform_ball
from bo_vs_es.objectives.models import rotation


def scalar_model(points, losses):
    x = torch.as_tensor(np.asarray(points), dtype=torch.double)
    values = -torch.as_tensor(losses, dtype=torch.double).unsqueeze(-1)
    y = (values - values.mean()) / values.std().clamp_min(1e-12)
    kernel = ScaleKernel(MaternKernel(
        nu=2.5, lengthscale_constraint=Interval(.03, 5.)),
        outputscale_constraint=Interval(.01, 100.)).double()
    kernel.base_kernel.lengthscale = .7
    kernel.outputscale = 1.
    with validate_input_scaling(False):
        return SingleTaskGP(x, y, train_Yvar=torch.full_like(y, 1e-6),
                            covar_module=kernel, outcome_transform=None)


def predictions(model, points):
    query = torch.as_tensor(points, dtype=torch.double)
    model.eval()
    acquisition = LogExpectedImprovement(model, best_f=model.train_targets.max())
    with torch.no_grad():
        posterior = model.posterior(query)
        values = [posterior.mean, posterior.variance, acquisition(query.unsqueeze(1))]
    return [v.detach().numpy() for v in values]


def differences(left, right):
    return {name: float(np.max(np.abs(a - b))) for name, a, b in zip(
        ["posterior_mean_max_abs", "posterior_variance_max_abs", "log_ei_max_abs"], left, right)}


def fit_model(model, settings):
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_gpytorch_mll_scipy(
            mll, options={"maxiter": settings["fit_maxiter"], "ftol": 1e-8})
    with torch.no_grad():
        likelihood = float(mll(model(*model.train_inputs), model.train_targets))
    return {"status": str(fit.status), "message": str(fit.message),
            "warnings": [str(w.message) for w in caught],
            "seconds": time.perf_counter() - start, "marginal_log_likelihood": likelihood,
            "lengthscale": float(model.covar_module.base_kernel.lengthscale.detach().item()),
            "outputscale": float(model.covar_module.outputscale.detach().item()),
            "mean": float(model.mean_module.constant.detach().item())}


def replay_history(points, losses, rotation_seed, config, bo_settings,
                   deadline=float("inf")):
    """No simulator calls. Both orientations receive the same physical history.

    Acquisition starts are freshly declared diagnostic starts, not a claim to
    recover the original run's random-generator state.
    """
    points, losses = np.asarray(points), np.asarray(losses)
    if points.ndim != 2 or points.shape[1] != 5 or len(points) != len(losses):
        raise ValueError("Expected matched five-dimensional observations")
    if not np.isfinite(points).all() or not np.isfinite(losses).all():
        raise ValueError("Nonfinite history")
    if np.any(np.linalg.norm(points, axis=1) > 1 + 1e-9):
        raise ValueError("History is outside the declared ball")
    torch.set_num_threads(1)
    matrices = [np.eye(5), rotation(rotation_seed)]
    queries = uniform_ball(np.random.default_rng(config["query_seed"]), config["query_points"], 5)
    models = [scalar_model(points @ q, losses) for q in matrices]
    frozen = [predictions(m, queries @ q) for m, q in zip(models, matrices)]
    fits = []
    for model in models:
        if time.monotonic() >= deadline:
            raise TimeoutError("Diagnostic wall-time allowance reached")
        fits.append(fit_model(model, bo_settings))
    fitted = [predictions(m, queries @ q) for m, q in zip(models, matrices)]

    # Use exactly the identity-fitted hyperparameters in both orientations. A
    # new model discards prediction caches from the independently fitted model.
    common_state = {k: v.detach().clone() for k, v in models[0].state_dict().items()}
    common = [scalar_model(points @ q, losses) for q in matrices]
    for model in common:
        model.load_state_dict(common_state)
        model.eval()
    pool = uniform_ball(np.random.default_rng(config["acquisition_start_seed"]),
                        bo_settings["raw_samples"], 5)
    acquisitions = [LogExpectedImprovement(m, best_f=m.train_targets.max()) for m in common]
    with torch.no_grad():
        scores = acquisitions[0](torch.tensor(pool, dtype=torch.double).unsqueeze(1))
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite acquisition-start scores")
    indices = scores.topk(bo_settings["num_restarts"]).indices.numpy()
    starts = pool[indices]
    solves = []
    for maxiter in config["acquisition_maxiter"]:
        paired = []
        for q, acquisition in zip(matrices, acquisitions):
            if time.monotonic() >= deadline:
                raise TimeoutError("Diagnostic wall-time allowance reached")
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate, value = optimize_acqf(
                    acquisition, bounds=torch.tensor([[-1.] * 5, [1.] * 5], dtype=torch.double),
                    q=1, num_restarts=len(starts), raw_samples=None,
                    batch_initial_conditions=torch.tensor(starts @ q, dtype=torch.double).unsqueeze(1),
                    nonlinear_inequality_constraints=[(lambda z: 1 - z.square().sum(), True)],
                    options={"maxiter": maxiter, "batch_limit": 1,
                             "ftol": config["acquisition_ftol"]},
                    retry_on_optimization_warning=False)
            z = candidate.detach().numpy()[0]
            u = z @ q.T
            if not np.isfinite(u).all() or np.linalg.norm(u) > 1 + 1e-6:
                raise ValueError("Acquisition solver returned an invalid candidate")
            paired.append({"u": u.tolist(), "norm": float(np.linalg.norm(u)),
                           "log_ei": float(value.detach().item()),
                           "seconds": time.perf_counter() - start,
                           "warnings": [str(w.message) for w in caught]})
        # Cross-score both physical candidates under one acquisition to separate
        # location disagreement from loss in achieved acquisition value.
        candidates = torch.tensor([v["u"] for v in paired], dtype=torch.double)
        with torch.no_grad():
            cross_scores = acquisitions[0](candidates.unsqueeze(1)).numpy()
        solves.append({"maxiter": maxiter, "identity": paired[0], "rotated": paired[1],
                       "candidate_distance": float(np.linalg.norm(candidates[0].numpy() - candidates[1].numpy())),
                       "common_model_log_ei": cross_scores.tolist(),
                       "common_model_log_ei_abs_gap": float(abs(cross_scores[0] - cross_scores[1]))})
    return {"observations": len(points), "rotation_seed": rotation_seed,
            "Q": matrices[1].tolist(), "frozen_control": differences(*frozen),
            "fits": fits, "fitted_control": differences(*fitted),
            "acquisition_starts_u": starts.tolist(), "acquisition": solves,
            "additional_simulator_calls": 0,
            "scope": "fixed-history development replay; no sequential intervention confirmation"}
