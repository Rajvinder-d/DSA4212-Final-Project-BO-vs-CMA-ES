"""Numerical controls with no dependency on Torch or downloaded battery data."""
import time

import numpy as np
from scipy.optimize import minimize


class BudgetExhausted(RuntimeError):
    pass


def validate_config(config):
    """Fail before execution if development consumes reserved confirmation cases."""
    for key in ("initial_design_seeds", "rotation_seeds"):
        values = config[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Empty or duplicate {key}")
        if set(values) & set(config["reserved_confirmation"][key]):
            raise ValueError(f"Development overlaps reserved confirmation {key}")
    checkpoints = config["checkpoints"]
    if not checkpoints or checkpoints != sorted(set(checkpoints)):
        raise ValueError("Checkpoints must be increasing and unique")
    if config["n_init"] != 20 or checkpoints[0] < config["n_init"]:
        raise ValueError("Use the v1 initial design of 20 points")
    if config["reference"]["total_calls"] < config["reference"]["starts"]:
        raise ValueError("Each reference start needs an evaluation allowance")
    if config["max_wall_seconds"] <= 0:
        raise ValueError("Wall-time allowance must be positive")


def uniform_ball(rng, count, dimension):
    points = rng.normal(size=(count, dimension))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points * rng.random((count, 1)) ** (1 / dimension)


class CountedResidual:
    """Count every callback, including failures and infeasible solver trials.

    Infeasible trials may inform SLSQP's gradient, but cannot become the returned
    feasible reference. They are not projected or given artificial penalties.
    """
    def __init__(self, residual, budget, deadline=float("inf")):
        self.residual = residual
        self.budget = budget
        self.deadline = deadline
        self.limit = budget
        self.rows = []
        self.best = None

    def __call__(self, point):
        if len(self.rows) >= min(self.limit, self.budget):
            raise BudgetExhausted("Counted simulator-call allocation reached")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Diagnostic wall-time allowance reached")
        point = np.asarray(point, dtype=float)
        if point.ndim != 1 or not np.isfinite(point).all():
            raise ValueError("Invalid reference candidate")
        norm = float(np.linalg.norm(point))
        row = {"evaluation": len(self.rows) + 1, "u": point.tolist(),
               "feasible": norm <= 1.0, "norm": norm, "status": "failed"}
        start = time.perf_counter()
        self.rows.append(row)
        try:
            residual = np.asarray(self.residual(point), dtype=float)
            if residual.size == 0 or not np.isfinite(residual).all():
                raise ValueError("Empty/nonfinite residual")
            loss = float(np.mean(residual ** 2))
            row.update(status="ok", loss=loss)
            if row["feasible"] and (self.best is None or loss < self.best["loss"]):
                self.best = dict(row)
            return loss
        except Exception as exc:
            row["error"] = repr(exc)
            raise
        finally:
            row["objective_s"] = time.perf_counter() - start


def ball_reference(residual, dimension, settings, deadline=float("inf")):
    """A counted, multistart SLSQP reference for exactly the unit-ball domain."""
    starts = np.vstack([np.zeros(dimension), uniform_ball(
        np.random.default_rng(settings["seed"]), settings["starts"] - 1, dimension) * .8])
    evaluator = CountedResidual(residual, settings["total_calls"], deadline)
    terminations = []
    for index, point in enumerate(starts):
        evaluator.limit = (index + 1) * evaluator.budget // len(starts)
        # Reserve one call per block to assess a strictly feasible final candidate.
        block_limit = evaluator.limit
        evaluator.limit -= 1
        try:
            fit = minimize(evaluator, point, method="SLSQP", jac="3-point",
                           bounds=[(-1., 1.)] * dimension,
                           constraints={"type": "ineq", "fun": lambda x: 1 - x @ x,
                                        "jac": lambda x: -2 * x},
                           options={"maxiter": settings["maxiter"],
                                    "ftol": settings["ftol"],
                                    "finite_diff_rel_step": settings["difference_step"]})
            candidate = np.asarray(fit.x)
            norm = np.linalg.norm(candidate)
            if norm > 1:
                candidate = candidate / norm * (1 - 1e-12)
            evaluator.limit = block_limit
            evaluator(candidate)
            terminations.append({"start": index, "success": bool(fit.success),
                                 "message": str(fit.message), "iterations": int(fit.nit),
                                 "solver_final_norm": float(norm), "termination": "solver"})
        except BudgetExhausted as exc:
            terminations.append({"start": index, "success": False, "message": str(exc),
                                 "termination": "budget"})
        except Exception as exc:
            terminations.append({"start": index, "success": False, "message": repr(exc),
                                 "termination": "error"})
            break
    return {"best": evaluator.best, "evaluations": len(evaluator.rows),
            "status": "failed" if any(t["termination"] == "error" for t in terminations) else "complete",
            "budget": evaluator.budget, "starts": starts.tolist(),
            "terminations": terminations, "trace": evaluator.rows,
            "scope": "computed feasible reference; not a certified global optimum"}
