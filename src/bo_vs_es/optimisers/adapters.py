import time
import warnings
import numpy as np
import torch
import cma
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll_scipy
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.settings import validate_input_scaling
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


def bo_candidate(z, losses, domain, ard, settings, rng):
    """Fit a GP and maximise LogEI in optimiser coordinates, without renormalising axes."""
    torch.set_num_threads(1)
    x = torch.as_tensor(np.asarray(z), dtype=torch.double)
    values = -torch.as_tensor(losses, dtype=torch.double).unsqueeze(-1)
    y = (values - values.mean()) / values.std().clamp_min(1e-12)
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=x.shape[1] if ard else None,
                                     lengthscale_constraint=Interval(.03, 5.0)),
                         outputscale_constraint=Interval(.01, 100.0)).double()
    kernel.base_kernel.lengthscale = .7
    kernel.outputscale = 1.0
    # Inputs intentionally remain in the rotated coordinates, not an axiswise unit cube.
    with validate_input_scaling(False):
        model = SingleTaskGP(x, y, train_Yvar=torch.full_like(y, 1e-6),
                             covar_module=kernel, outcome_transform=None)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_gpytorch_mll_scipy(mll, options={"maxiter": settings["fit_maxiter"], "ftol": 1e-8})
    fit_s = time.perf_counter() - start
    acquisition = LogExpectedImprovement(model, best_f=y.max())
    feasible = torch.as_tensor(domain.sample(rng, settings["raw_samples"]), dtype=torch.double)
    with torch.no_grad():
        score = acquisition(feasible.unsqueeze(1))
    if not torch.isfinite(score).all():
        raise RuntimeError("Nonfinite acquisition values at feasible starts")
    starts = feasible[score.topk(settings["num_restarts"]).indices].unsqueeze(1)
    d = x.shape[1]
    extent = np.ones(d) if domain.kind == "ball" else np.abs(domain.q).sum(axis=0)
    bounds = torch.tensor(np.stack([-extent, extent]), dtype=torch.double)
    constraints = {}
    if domain.kind == "ball":
        constraints["nonlinear_inequality_constraints"] = [(lambda point: 1 - point.square().sum(), True)]
    else:
        constraints["inequality_constraints"] = [
            (torch.arange(d), torch.tensor(sign * row, dtype=torch.double), -1.)
            for row in domain.q for sign in [-1, 1]
        ]
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as acquired:
        warnings.simplefilter("always")
        candidate, acq_value = optimize_acqf(
            acquisition, bounds, q=1, num_restarts=settings["num_restarts"], raw_samples=None,
            batch_initial_conditions=starts,
            options={"maxiter": settings["acq_maxiter"], "batch_limit": 1, "ftol": 1e-9},
            retry_on_optimization_warning=False, **constraints)
    proposal = candidate.detach().numpy()[0]
    if not domain.feasible(proposal, tolerance=1e-6):
        raise RuntimeError("Acquisition solver returned an infeasible point")
    # Only remove floating point boundary error, never repair an infeasible acquisition result.
    result = domain.project(proposal)
    return result, {
        "fit_s": fit_s, "acquisition_s": time.perf_counter() - start,
        "lengthscales": kernel.base_kernel.lengthscale.detach().numpy().ravel().tolist(),
        "outputscale": float(kernel.outputscale.detach()), "fit_status": str(fit.status),
        "fit_message": str(fit.message), "log_ei": float(acq_value.detach()),
        "warnings": [str(w.message) for w in [*caught, *acquired]],
    }


def make_cma(mean, seed, diagonal, settings):
    d = len(mean)
    # pycma 4.5 adds exp(1e-4*j/d) to full-CMA initial variances.
    # Cancel that numerical perturbation to implement an isotropic start.
    es = cma.CMAEvolutionStrategy(mean, settings["sigma0"], {
        "popsize": settings["population"], "seed": int(seed), "verbose": -9,
        "verb_log": 0, "CMA_diagonal": bool(diagonal), "CMA_active": False,
        "CMA_mirrors": 0, "CMA_diagonal_decoding": 0,
        "CMA_teststds": np.exp(-.5e-4 * np.arange(d) / d).tolist() if not diagonal else None,
        "tolflatfitness": np.inf, "tolfun": 0, "tolfunhist": 0, "tolx": 0,
    })
    # Hold the path and covariance learning rates fixed across representations.
    # These version-specific fields are covered by adapter tests and the lockfile.
    if diagonal:
        es.sp.cc_sep = es.sp.cc
        es.sp.c1_sep = es.sp.c1
        es.sp.cmu_sep = es.sp.cmu
    return es


def covariance(es):
    # Includes global sigma and any sampler/decoding scales, including diagonal CMA.
    d = len(es.mean)
    sampler = np.asarray(es.sm.variances)
    c = np.asarray(es.C)
    if c.ndim != 2:
        c = np.diag(sampler)
    scaling = np.asarray(es.sigma_vec.scaling)
    if scaling.ndim == 0:
        scaling = np.full(d, scaling)
    return es.sigma ** 2 * scaling[:, None] * c * scaling[None, :]
