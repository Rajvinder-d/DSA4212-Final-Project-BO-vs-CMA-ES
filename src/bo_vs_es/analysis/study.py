import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from bo_vs_es import ROOT
from bo_vs_es.objectives.models import Battery, LOWER, UPPER, physical
from bo_vs_es.experiments.runner import dump, valid_completed, fingerprint
from bo_vs_es.objectives.data import sha256

COLORS = dict(zip(["bo_iso", "bo_ard", "cma_full", "cma_diag", "random"],
                  ["#0072B2", "#56B4E9", "#D55E00", "#CC79A7", "#777777"]))


def reference_fit(branches=2, corrected=True, budget=300, starts=3):
    """Bounded residual-based reference, counting every numerical-Jacobian simulator call."""
    objective = Battery(corrected=corrected)
    d = 1 + 2 * branches
    mid = (np.log(LOWER[:d]) + np.log(UPPER[:d])) / 2
    scale = (np.log(UPPER[:d]) - np.log(LOWER[:d])) / 2
    records, best = [], {"loss": float("inf")}
    start = time.perf_counter()
    rng = np.random.default_rng(710)
    starting_points = [np.zeros(d), *rng.uniform(-.8, .8, (starts - 1, d))]
    terminations = []
    for number, x0 in enumerate(starting_points):
        limit = (number + 1) * budget // starts

        def residual(u):
            if len(records) >= limit:
                raise StopIteration("Counted simulator-call allocation reached")
            theta = np.exp(mid + scale * u)
            r = objective.prediction(theta) - objective.measured
            loss = float(np.mean(r ** 2))
            records.append({"evaluation": len(records) + 1, "start": number, "loss": loss,
                            "u": u.tolist(), "theta": theta.tolist()})
            if loss < best["loss"]:
                best.update({"loss": loss, "u": u.tolist(), "theta": theta.tolist()})
            return r

        try:
            fit = least_squares(residual, x0, bounds=(-np.ones(d), np.ones(d)),
                                jac="3-point", ftol=1e-10, xtol=1e-10, gtol=1e-10)
            terminations.append(str(fit.message))
        except StopIteration as exc:
            terminations.append(str(exc))
    best.update({"branches": branches, "corrected": corrected, "domain": "full physical box",
                 "source_data_sha256": objective.manifest["output_sha256"],
                 "model_runner_code_sha256": fingerprint(),
                 "analysis_code_sha256": sha256(Path(__file__)),
                 "starts_u": [point.tolist() for point in starting_points],
                 "settings": {"jac": "3-point", "ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-10, "seed": 710},
                 "evaluations": len(records), "budget": budget, "search_s": time.perf_counter() - start,
                 "terminations": terminations, "calibration_rmse_v": np.sqrt(best["loss"]),
                 "assessment_calls": 2})
    for label in ["fuds80", "fuds50"]:
        assessment = Battery(label, corrected=corrected)
        best[label + "_rmse_v"] = float(np.sqrt(np.mean((assessment.prediction(best["theta"]) - assessment.measured) ** 2)))
    folder = ROOT / "artifacts/references"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{branches}rc_{'corrected' if corrected else 'uncorrected'}"
    dump(folder / f"{name}.json", best)
    dump(folder / f"{name}_trace.json", records)
    return best


def curvature(reference, h=1e-4):
    """Local Gauss–Newton curvature in dimensionless log coordinates; separate diagnostic calls."""
    objective = Battery(corrected=reference["corrected"])
    u = np.array(reference["u"])
    if len(u) != 5:
        raise ValueError("Use the two-RC reference for the five-dimensional curvature diagnostic")
    columns = []
    for j in range(5):
        lo, hi = u.copy(), u.copy()
        lo[j] = max(-1., lo[j] - h)
        hi[j] = min(1., hi[j] + h)
        columns.append((objective.residual(hi) - objective.residual(lo)) / (hi[j] - lo[j]))
    jacobian = np.column_stack(columns)
    matrix = 2 * jacobian.T @ jacobian / len(objective.time)
    eigenvalues = np.linalg.eigvalsh(matrix)
    return matrix, eigenvalues, {"extra_simulator_calls": 10, "difference_step": h,
                               "interpretation": "local Gauss-Newton approximation, not the exact global Hessian"}


def save_figure(fig, name):
    path = ROOT / "results/figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def load_results(collection="pilot_v1"):
    folder = ROOT / "artifacts" / collection
    frame = pd.read_csv(folder / "index.csv")
    metadata = json.loads((folder / "collection.json").read_text())
    if frame.run_id.tolist() != metadata["run_ids"]:
        raise ValueError("Collection index is incomplete or does not match its provenance")
    saved_summaries = pd.DataFrame([json.loads((ROOT / row.run_path / "summary.json").read_text())
                                   for row in frame.itertuples()])
    pd.testing.assert_frame_equal(frame, saved_summaries, check_dtype=False, rtol=1e-12, atol=1e-14)
    for _, row in frame[frame.status == "complete"].iterrows():
        run = ROOT / row.run_path
        manifest = json.loads((run / "manifest.json").read_text())
        if not valid_completed(run, manifest["run_key"]):
            raise ValueError(f"Invalid or modified run artifacts: {row.run_id}")
    return frame


def evaluations(row):
    return pd.read_csv(ROOT / row.run_path / "evaluations.csv")


def plot_convergence(results, block="battery_ball"):
    selected = results[(results.block == block) & (results.status == "complete")]
    rotations = selected.rotation_seed.drop_duplicates().sort_values(na_position="first")
    fig, axes = plt.subplots(1, len(rotations), figsize=(6 * len(rotations), 4), squeeze=False)
    for ax, rot in zip(axes[0], rotations):
        subset = selected[selected.rotation_seed.isna() if pd.isna(rot) else selected.rotation_seed.eq(rot)]
        for method, group in subset.groupby("method"):
            curves = []
            for _, row in group.iterrows():
                trace = evaluations(row)
                curve = trace.loss.cummin().to_numpy()
                curves.append(curve)
                ax.plot(trace.evaluation, curve, color=COLORS[method], alpha=.18, lw=.8)
            if curves:
                ax.plot(np.arange(1, len(curves[0]) + 1), np.median(curves, axis=0),
                        color=COLORS[method], label=method, lw=2)
        ax.set(xlabel="Counted objective evaluations", ylabel="Best calibration MSE (V²)", yscale="log",
               title="Identity" if pd.isna(rot) else f"Rotation seed {int(rot)}")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"{block}: individual runs and median; reference box fit excluded")
    fig.tight_layout()
    return fig


def paired_rotation_effects(results, block="battery_ball"):
    selected = results[(results.block == block) & (results.status == "complete")].copy()
    baseline = selected[selected.rotation_seed.isna()]
    rotated = selected[selected.rotation_seed.notna()]
    keys = ["method", "seed", "domain", "handler"]
    pairs = rotated.merge(baseline[keys + ["best_loss", "run_id"]], on=keys, suffixes=("", "_identity"), validate="many_to_one")
    pairs["log10_loss_ratio"] = np.log10(pairs.best_loss / pairs.best_loss_identity)
    return pairs[[*keys, "rotation_seed", "log10_loss_ratio", "run_id", "run_id_identity"]]


def effect_summary(pairs):
    """Average selected rotations within seed, then resample seeds, never timestamp observations."""
    rows = []
    for method, group in pairs.groupby("method"):
        values = group.groupby("seed").log10_loss_ratio.mean().to_numpy()
        rng = np.random.default_rng(129)
        samples = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
        rows.append({"method": method, "seeds": len(values), "mean_log10_ratio": values.mean(),
                     "mean_absolute_log10_ratio": group.groupby("seed").log10_loss_ratio.apply(lambda v: v.abs().mean()).mean(),
                     "bootstrap_low": np.quantile(samples, .025), "bootstrap_high": np.quantile(samples, .975),
                     "interpretation": "descriptive only: fewer than 10 seeds" if len(values) < 10 else
                     "conditional on selected cell, traces and rotations"})
    return pd.DataFrame(rows)


def sensitivity_contrasts(pairs):
    """Pair methods within seed; positive means the coordinate-specific method changes more."""
    frame = pairs.copy()
    frame["absolute_effect"] = frame.log10_loss_ratio.abs()
    wide = frame.groupby(["seed", "method"]).absolute_effect.mean().unstack("method")
    rows = []
    for coordinate, general in [("bo_ard", "bo_iso"), ("cma_diag", "cma_full")]:
        values = (wide[coordinate] - wide[general]).dropna().to_numpy()
        rng = np.random.default_rng(129)
        samples = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
        rows.append({"contrast": f"{coordinate} minus {general}", "seeds": len(values),
                     "mean_absolute_effect_difference": values.mean(),
                     "bootstrap_low": np.quantile(samples, .025), "bootstrap_high": np.quantile(samples, .975),
                     "interpretation": "descriptive pilot contrast" if len(values) < 10 else "conditional seed uncertainty"})
    return pd.DataFrame(rows)


def boundary_interaction(results):
    selected = results[(results.block == "battery_boundary") &
                       results.method.str.startswith("cma") & (results.status == "complete")].copy()
    selected["orientation"] = selected.rotation_seed.fillna(-1)
    selected["log_loss"] = np.log10(selected.best_loss)
    identity = selected[selected.orientation == -1]
    keys = ["method", "seed", "handler"]
    paired = selected[selected.orientation != -1].merge(identity[keys + ["log_loss"]], on=keys,
                                                        suffixes=("", "_identity"), validate="many_to_one")
    paired["rotation_effect"] = paired.log_loss - paired.log_loss_identity
    wide = paired.pivot(index=["method", "seed", "orientation"], columns="handler", values="rotation_effect")
    wide["projection_minus_resampling"] = wide["project"] - wide["resample"]
    return wide


def plot_assessment(results):
    selected = results[(results.block == "battery_ball") & (results.status == "complete")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["fuds80_rmse_v", "fuds50_rmse_v"]):
        for i, method in enumerate(COLORS):
            group = selected[selected.method == method]
            ax.scatter(np.full(len(group), i), group[metric] * 1000, color=COLORS[method], alpha=.7)
        ax.set(xticks=range(5), xticklabels=list(COLORS), ylabel="Held-out voltage RMSE (mV)", title=metric.split("_")[0])
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Final recommendations only; assessment never guides search")
    fig.tight_layout()
    return fig


def boundary_table(results):
    selected = results[(results.block == "battery_boundary") & (results.status == "complete")].copy()
    selected["rejected_fraction"] = selected.rejected / selected.proposals
    selected["repaired_fraction"] = selected.repaired / selected.proposals
    return selected[["run_id", "method", "seed", "rotation_seed", "handler", "best_loss",
                     "rejected_fraction", "repaired_fraction", "search_s"]]


def diagnostic_figure(results):
    selected = results[(results.block == "battery_ball") & (results.seed == 0) &
                       (results.rotation_seed == 101) & (results.status == "complete")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in ["bo_ard", "cma_full"]:
        row = selected[selected.method == method].iloc[0]
        logs = [json.loads(s) for s in (ROOT / row.run_path / "diagnostics.jsonl").read_text().splitlines()]
        if method == "bo_ard":
            values = np.array([v["lengthscales"] for v in logs])
            for j in range(5):
                axes[0].plot([v["after_evaluations"] for v in logs], values[:, j], label=f"z{j+1}")
            axes[0].set(title=f"ARD length scales\nrun {row.run_id}", ylabel="Length scale", yscale="log")
        else:
            values = np.array([v["eigenvalues"] for v in logs])
            for j in range(5):
                axes[1].plot([v["after_evaluations"] for v in logs], values[:, j], label=f"eigenvalue {j+1}")
            axes[1].set(title=f"Full CMA sampling covariance\nrun {row.run_id}", ylabel="Variance", yscale="log")
    for ax in axes:
        ax.set_xlabel("Counted objective evaluations")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def export_results(results):
    folder = ROOT / "results/tables"
    folder.mkdir(parents=True, exist_ok=True)
    results.to_csv(folder / "pilot_runs.csv", index=False)
    pairs = paired_rotation_effects(results)
    pairs.to_csv(folder / "pilot_rotation_pairs.csv", index=False)
    effect_summary(pairs).to_csv(folder / "pilot_rotation_summary.csv", index=False)
    boundary_table(results).to_csv(folder / "pilot_boundary.csv", index=False)
    dump(folder / "provenance.json", {"collection": "artifacts/pilot_v1/collection.json",
                                      "run_ids": results.run_id.tolist(),
                                      "analysis_code_sha256": sha256(Path(__file__)),
                                      "collection_sha256": sha256(ROOT / "artifacts/pilot_v1/collection.json"),
                                      "generator": "notebooks/04_results_and_interpretation.ipynb"})


def optimiser_health(results):
    """Count numerical warnings and statuses; a completed run need not have converged fits."""
    rows = []
    for row in results.itertuples():
        path = ROOT / row.run_path / "diagnostics.jsonl"
        if not row.method.startswith("bo") or not path.exists():
            continue
        logs = [json.loads(line) for line in path.read_text().splitlines()]
        for log in logs:
            rows.append({"run_id": row.run_id, "block": row.block, "method": row.method,
                         "seed": row.seed, "rotation_seed": row.rotation_seed,
                         "after_evaluations": log["after_evaluations"], "fit_status": log["fit_status"],
                         "fit_message": log["fit_message"], "warning_count": len(log["warnings"]),
                         "warnings": " | ".join(log["warnings"]),
                         "lengthscale_min": min(log["lengthscales"]),
                         "lengthscale_max": max(log["lengthscales"])})
    return pd.DataFrame(rows)


def recommendation_parameters(results):
    """Canonicalise RC branches only for presentation; retain original run IDs."""
    rows = []
    for row in results[(results.objective == "battery") & (results.status == "complete")].itertuples():
        rec = json.loads((ROOT / row.run_path / "recommendation.json").read_text())
        r0, r1, t1, r2, t2 = rec["theta"]
        if t1 > t2:
            r1, t1, r2, t2 = r2, t2, r1, t1
        u = np.array(rec["u"])
        rows.append({"run_id": row.run_id, "block": row.block, "method": row.method,
                     "seed": row.seed, "rotation_seed": row.rotation_seed,
                     "R0_ohm": r0, "R_fast_ohm": r1, "tau_fast_s": t1,
                     "R_slow_ohm": r2, "tau_slow_s": t2, "tau_ratio": t2 / t1,
                     "dimensionless_boundary_margin": 1 - (np.linalg.norm(u) if row.domain == "ball" else max(abs(u)))})
    return pd.DataFrame(rows)


def validate_collection(results):
    """Audit saved candidate geometry and matched information, without new simulator calls."""
    from bo_vs_es.objectives.models import Domain, physical
    initial_by_domain_seed = {}
    initial_loss_by_objective = {}
    calls = 0
    for row in results.itertuples():
        if row.status != "complete":
            continue
        folder = ROOT / row.run_path
        manifest = json.loads((folder / "manifest.json").read_text())
        if not valid_completed(folder, manifest["run_key"]):
            raise ValueError(f"Invalid completed run: {row.run_id}")
        initial = json.loads((folder / "initial.json").read_text())
        u = np.array(initial["u"])
        key = (row.domain, row.seed)
        if key in initial_by_domain_seed:
            np.testing.assert_allclose(u, initial_by_domain_seed[key], atol=1e-13, rtol=0)
        initial_by_domain_seed[key] = u
        trace = pd.read_csv(folder / "evaluations.csv")
        domain = Domain(row.domain, np.array(initial["Q"]))
        z_all = np.array([json.loads(v) for v in trace.z])
        u_all = np.array([json.loads(v) for v in trace.u])
        assert all(domain.feasible(z) for z in z_all), row.run_id
        np.testing.assert_allclose(domain.to_base(z_all), u_all, atol=1e-13, rtol=0)
        np.testing.assert_allclose(u_all[:row.n_init], u, atol=1e-13, rtol=0)
        condition = None if pd.isna(row.condition) else row.condition
        loss_key = (row.objective, condition, row.corrected, row.domain, row.seed)
        initial_losses = trace.loss.iloc[:row.n_init].to_numpy()
        if loss_key in initial_loss_by_objective:
            np.testing.assert_allclose(initial_losses, initial_loss_by_objective[loss_key], atol=1e-13, rtol=1e-12)
        initial_loss_by_objective[loss_key] = initial_losses
        if row.objective == "battery":
            theta = np.array([json.loads(v) for v in trace.theta])
            np.testing.assert_allclose(theta, physical(u_all), atol=1e-12, rtol=1e-12)
        rec = json.loads((folder / "recommendation.json").read_text())
        assert np.isclose(trace.loss.iloc[rec["evaluation"] - 1], rec["loss"], rtol=1e-12)
        proposals = [json.loads(line) for line in (folder / "proposals.jsonl").read_text().splitlines()]
        assert sum(p["rejected"] for p in proposals) == row.rejected
        assert sum(p["repaired"] for p in proposals) == row.repaired
        assert len(proposals) == row.proposals
        calls += len(trace)
    return {"verified_runs": int(results.status.eq("complete").sum()), "search_calls": calls,
            "assessment_calls": int(results.assessment_calls.sum()), "additional_simulator_calls": 0,
            "matched_initial_designs": len(initial_by_domain_seed)}


def paired_path_deviations(results):
    """Trace small numerical departures in physically paired candidate paths."""
    selected = results[(results.block == "battery_ball") & (results.status == "complete")]
    rows = []
    for (method, seed), group in selected.groupby(["method", "seed"]):
        baseline = group[group.rotation_seed.isna()].iloc[0]
        trace_i = evaluations(baseline)
        u_i = np.array([json.loads(v) for v in trace_i.u])
        for _, rotated in group[group.rotation_seed.notna()].iterrows():
            trace_q = evaluations(rotated)
            u_q = np.array([json.loads(v) for v in trace_q.u])
            distances = np.linalg.norm(u_i - u_q, axis=1)
            for index, distance in enumerate(distances):
                rows.append({"method": method, "seed": seed, "rotation_seed": rotated.rotation_seed,
                             "run_id": rotated.run_id, "run_id_identity": baseline.run_id,
                             "evaluation": index + 1, "base_coordinate_distance": distance})
    return pd.DataFrame(rows)


def frozen_gp_rotation_check(results):
    """Posterior/LogEI control at matched points with identical fixed GP hyperparameters.

    No objective is called; observations come only from the shared initial DST design.
    This checks GP mathematics, independently of fitted-mean and acquisition-solver paths.
    """
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.models import SingleTaskGP
    from botorch.settings import validate_input_scaling
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from bo_vs_es.objectives.models import Domain, rotation
    selected = results[(results.block == "battery_ball") & (results.method == "bo_iso") &
                       (results.seed == 0) & results.rotation_seed.isna()].iloc[0]
    trace = evaluations(selected).iloc[:selected.n_init]
    u = np.array([json.loads(v) for v in trace.u])
    y = -torch.tensor(trace.loss.to_numpy().copy(), dtype=torch.double).reshape(-1, 1)
    y = (y-y.mean()) / y.std()
    points = Domain('ball', np.eye(5)).sample(np.random.default_rng(930), 100)
    outputs = []
    for q in [np.eye(5), rotation(101)]:
        x = torch.tensor(u @ q, dtype=torch.double)
        kernel = ScaleKernel(MaternKernel(nu=2.5)).double()
        kernel.base_kernel.lengthscale = .7
        kernel.outputscale = 1.
        with validate_input_scaling(False):
            model = SingleTaskGP(x, y, train_Yvar=torch.full_like(y, 1e-6),
                                 covar_module=kernel, outcome_transform=None)
        query = torch.tensor(points @ q, dtype=torch.double)
        with torch.no_grad():
            posterior = model.posterior(query)
            acquisition = LogExpectedImprovement(model,best_f=y.max())(query.unsqueeze(1))
            outputs.append([posterior.mean.numpy(),posterior.variance.numpy(),acquisition.numpy()])
    return {"source_run_id": selected.run_id, "initial_observations": len(trace), "query_points": 100,
            "max_posterior_mean_difference": float(np.max(np.abs(outputs[0][0]-outputs[1][0]))),
            "max_posterior_variance_difference": float(np.max(np.abs(outputs[0][1]-outputs[1][1]))),
            "max_log_ei_difference": float(np.max(np.abs(outputs[0][2]-outputs[1][2]))),
            "extra_simulator_calls": 0,
            "scope": "fixed scalar length scale .7, output scale 1 and common default mean; no refitting"}
