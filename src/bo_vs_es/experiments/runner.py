import csv
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bo_vs_es import ROOT
from bo_vs_es.objectives.data import sha256
from bo_vs_es.objectives.models import Battery, Quadratic, Domain, physical, rotation, LOWER, UPPER
from bo_vs_es.optimisers.adapters import bo_candidate, make_cma, covariance

METHODS = ["bo_iso", "bo_ard", "cma_full", "cma_diag", "random"]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def fingerprint():
    files = [p for folder in ["objectives", "optimisers", "experiments"]
             for p in sorted((ROOT / "src/bo_vs_es" / folder).glob("*.py"))]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


class Evaluator:
    """The sole path to an objective. Failed simulator attempts also consume budget."""
    def __init__(self, objective, domain, budget, folder, battery=False):
        self.objective, self.domain, self.budget = objective, domain, budget
        self.folder, self.battery = Path(folder), battery
        self.rows, self.z, self.losses, self.diagnostics = [], [], [], []
        self.rejected = self.repaired = self.proposals = 0
        self.start = time.perf_counter()

    @property
    def remaining(self):
        return self.budget - len(self.rows)

    def proposal(self, proposed, evaluated=None):
        self.proposals += 1
        rejected = evaluated is None
        repaired = not rejected and not np.allclose(proposed, evaluated, atol=1e-12, rtol=0)
        self.rejected += int(rejected)
        self.repaired += int(repaired)
        with (self.folder / "proposals.jsonl").open("a") as stream:
            stream.write(json.dumps({"proposal": self.proposals, "after_evaluations": len(self.rows),
                                     "proposed_z": np.asarray(proposed).tolist(),
                                     "evaluated_z": None if rejected else np.asarray(evaluated).tolist(),
                                     "rejected": rejected, "repaired": repaired}) + "\n")

    def evaluate(self, z, phase="search", proposed=None):
        if not self.remaining:
            raise RuntimeError("Objective evaluation budget exhausted")
        z = np.asarray(z, dtype=float)
        if z.shape != (5,) or not np.isfinite(z).all() or not self.domain.feasible(z):
            raise ValueError("Malformed/infeasible candidate (no simulator call made)")
        u = self.domain.to_base(z)
        start = time.perf_counter()
        status, error, loss = "ok", "", None
        try:
            loss = float(self.objective(u))
            if not np.isfinite(loss):
                raise ValueError("Nonfinite objective result")
        except Exception as exc:
            status, error, loss = "failed", repr(exc), None
        elapsed = time.perf_counter() - start
        row = {"evaluation": len(self.rows) + 1, "phase": phase, "loss": loss,
               "status": status, "error": error, "objective_s": elapsed,
               "elapsed_s": time.perf_counter() - self.start,
               "z": json.dumps(z.tolist()), "u": json.dumps(u.tolist()),
               "theta": json.dumps(physical(u).tolist()) if self.battery else "",
               "proposed_z": json.dumps(np.asarray(z if proposed is None else proposed).tolist())}
        self.rows.append(row)
        path = self.folder / "evaluations.csv"
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if len(self.rows) == 1:
                writer.writeheader()
            writer.writerow(row)
        if status != "ok":
            raise RuntimeError(f"Objective attempt {len(self.rows)} failed: {error}")
        self.z.append(z.copy())
        self.losses.append(loss)
        return loss

    def diagnostic(self, value):
        value["after_evaluations"] = len(self.rows)
        self.diagnostics.append(value)
        with (self.folder / "diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")


def jobs_for(recipe):
    jobs = []
    for block in recipe["blocks"]:
        for condition, domain, rot, seed, method in itertools.product(
            block.get("conditions", [None]), block["domains"], block["rotations"], block["seeds"], block["methods"]
        ):
            handlers = block.get("cma_handlers", ["resample"]) if method.startswith("cma") else ["none"]
            for handler in handlers:
                jobs.append({"block": block["name"], "objective": block["objective"],
                             "condition": condition, "domain": domain, "rotation_seed": rot,
                             "seed": seed, "method": method, "handler": handler,
                             "budget": block.get("budget", recipe["budget"]),
                             "n_init": recipe["n_init"], "corrected": block.get("corrected", True)})
    return jobs


def valid_completed(folder, key):
    """Only reuse a complete run whose files, evaluation count, and best loss still agree."""
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        if manifest["status"] != "complete" or manifest["run_key"] != key:
            return False
        if not all(sha256(folder / p) == digest for p, digest in manifest["output_sha256"].items()):
            return False
        frame = pd.read_csv(folder / "evaluations.csv")
        return (len(frame) == summary["evaluations"] == summary["budget"] and
                frame.status.eq("ok").all() and np.isclose(frame.loss.min(), summary["best_loss"], rtol=1e-12))
    except (OSError, KeyError, ValueError):
        return False


def execute_job(job, protocol, collection, provenance):
    resolved = {"job": job, "protocol": protocol, "provenance": provenance}
    key = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    folder = collection / key[:16]
    if valid_completed(folder, key):
        result = json.loads((folder / "summary.json").read_text())
        return result, True
    if folder.exists():
        # Never overwrite failed or incomplete runs. A later attempt has an explicit suffix.
        index = 1
        while (collection / f"{key[:16]}-retry{index}").exists():
            candidate = collection / f"{key[:16]}-retry{index}"
            if valid_completed(candidate, key):
                return json.loads((candidate / "summary.json").read_text()), True
            index += 1
        folder = collection / f"{key[:16]}-retry{index}"
    folder.mkdir(parents=True)
    manifest = {"run_key": key, "run_id": folder.name, "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(), **provenance}
    dump(folder / "manifest.json", manifest)
    dump(folder / "config.json", resolved)
    domain = Domain(job["domain"], rotation(job["rotation_seed"]))
    objective = Battery(corrected=job["corrected"]) if job["objective"] == "battery" else Quadratic(job["condition"])
    evaluator = Evaluator(objective, domain, job["budget"], folder, job["objective"] == "battery")
    initial_rng = np.random.default_rng(job["seed"])
    search_rng = np.random.default_rng(np.random.SeedSequence([job["seed"], 41]))
    torch.manual_seed(job["seed"] + 10000)
    initial = domain.sample(initial_rng, job["n_init"])
    dump(folder / "initial.json", {"z": initial.tolist(), "u": domain.to_base(initial).tolist(), "Q": domain.q.tolist()})
    summary = {**job, "run_id": folder.name, "run_path": str(folder.relative_to(ROOT)), "status": "failed"}
    try:
        for point in initial:
            evaluator.evaluate(point, "initial")
        method = job["method"]
        if method.startswith("cma"):
            es = make_cma(evaluator.z[np.argmin(evaluator.losses)], job["seed"] + 10000,
                          method == "cma_diag", protocol["cma"])
            while evaluator.remaining:
                if evaluator.remaining < protocol["cma"]["population"]:
                    raise ValueError("Budget must accommodate whole CMA generations after initialisation")
                points, losses = [], []
                for _ in range(protocol["cma"]["population"]):
                    for attempt in range(protocol["cma"]["max_proposals_per_point"]):
                        proposed = np.asarray(es.ask(1)[0])
                        point = domain.project(proposed) if job["handler"] == "project" else proposed
                        if domain.feasible(point):
                            evaluator.proposal(proposed, point)
                            points.append(point)
                            losses.append(evaluator.evaluate(point, proposed=proposed))
                            break
                        evaluator.proposal(proposed)
                    else:
                        raise RuntimeError("CMA feasible resampling limit reached; no silent repair")
                es.tell(points, losses)
                cov = covariance(es)
                evaluator.diagnostic({"generation": es.countiter, "sigma": float(es.sigma),
                                      "mean_z": es.mean.tolist(), "covariance_z": cov.tolist(),
                                      "covariance_u": (domain.q @ cov @ domain.q.T).tolist(),
                                      "eigenvalues": np.linalg.eigvalsh(cov).tolist()})
        else:
            while evaluator.remaining:
                if method == "random":
                    point = domain.sample(search_rng, 1)[0]
                elif method in ["bo_iso", "bo_ard"]:
                    point, diagnostic = bo_candidate(evaluator.z, evaluator.losses, domain,
                                                     method == "bo_ard", protocol["bo"], search_rng)
                    evaluator.diagnostic(diagnostic)
                else:
                    raise ValueError(f"Unknown method: {method}")
                evaluator.proposal(point, point)
                evaluator.evaluate(point)
        search_s = time.perf_counter() - evaluator.start
        best = int(np.argmin(evaluator.losses))
        best_z = evaluator.z[best]
        best_u = domain.to_base(best_z)
        recommendation = {"rule": "lowest observed deterministic calibration loss", "evaluation": best + 1,
                          "z": best_z.tolist(), "u": best_u.tolist(), "loss": evaluator.losses[best]}
        summary.update({"status": "complete", "best_loss": evaluator.losses[best],
                        "initial_best_loss": min(evaluator.losses[:job["n_init"]]), "search_s": search_s,
                        "objective_s": sum(r["objective_s"] for r in evaluator.rows),
                        "evaluations": len(evaluator.rows), "proposals": evaluator.proposals,
                        "rejected": evaluator.rejected, "repaired": evaluator.repaired,
                        "assessment_calls": 0, "assessment_s": 0.})
        summary["decision_and_logging_s"] = search_s - summary["objective_s"]
        if job["objective"] == "battery":
            recommendation["theta"] = physical(best_u).tolist()
            start = time.perf_counter()
            for label in ["fuds80", "fuds50"]:
                assessment = Battery(label, corrected=job["corrected"])
                summary[label + "_rmse_v"] = float(np.sqrt(assessment(best_u)))
                summary["assessment_calls"] += 1
            summary["assessment_s"] = time.perf_counter() - start
            recommendation["assessment"] = {k: v for k, v in summary.items() if "rmse" in k}
        dump(folder / "recommendation.json", recommendation)
    except Exception as exc:
        summary.update({"status": "failed", "error": repr(exc), "evaluations": len(evaluator.rows),
                        "search_s": time.perf_counter() - evaluator.start})
        (folder / "error.txt").write_text(traceback.format_exc())
    dump(folder / "summary.json", summary)
    manifest.update({"status": summary["status"], "finished_utc": datetime.now(timezone.utc).isoformat(),
                     "output_sha256": {p.name: sha256(p) for p in folder.iterdir() if p.name != "manifest.json"}})
    dump(folder / "manifest.json", manifest)
    return summary, False


def run_collection(recipe_path, progress=True):
    recipe_path = Path(recipe_path)
    if not recipe_path.is_absolute():
        recipe_path = ROOT / recipe_path
    recipe = json.loads(recipe_path.read_text())
    protocol = json.loads((ROOT / recipe["protocol"]).read_text())
    if (protocol["dimension"] != 5 or protocol["capacity_ah"] != 2.0 or protocol["window_s"] != 3600
            or not np.array_equal(protocol["lower"], LOWER) or not np.array_equal(protocol["upper"], UPPER)):
        raise ValueError("Protocol conflicts with the implemented v1 model; revise model and protocol together")
    data_manifest = json.loads((ROOT / "data/processed/manifest.json").read_text())
    for name, digest in data_manifest["output_sha256"].items():
        if sha256(ROOT / "data/processed" / name) != digest:
            raise ValueError(f"Prepared data checksum mismatch: {name}")
    collection = ROOT / "artifacts" / recipe["name"]
    collection.mkdir(parents=True, exist_ok=True)
    provenance = {"code_sha256": fingerprint(), "source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "data_sha256": data_manifest["output_sha256"],
        "preprocessing_sha256": hashlib.sha256(json.dumps(
            {k: v for k, v in data_manifest.items() if k != "prepared_utc"}, sort_keys=True).encode()).hexdigest(),
        "dependency_lock_sha256": sha256(ROOT / "requirements-lock.txt"),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "scipy", "torch", "botorch", "gpytorch", "cma",
                      "openpyxl", "matplotlib", "nbformat", "nbclient", "ipykernel"]},
        "hardware": {"machine": platform.machine(), "platform": platform.platform(),
                     "python": platform.python_version()}, "torch_threads": 1}
    torch.set_num_threads(1)
    results = []
    jobs = jobs_for(recipe)
    for i, job in enumerate(jobs):
        result, cached = execute_job(job, protocol, collection, provenance)
        results.append(result)
        pd.DataFrame(results).to_csv(collection / "index.csv", index=False)
        if progress:
            timing = "(verified cached)" if cached else f"{result['search_s']:.1f}s"
            print(f"[{i+1}/{len(jobs)}] {job['block']} {job['method']} {job['domain']} "
                  f"Q={job['rotation_seed']} seed={job['seed']} {job['handler']}: "
                  f"{result['status']} {timing}", flush=True)
    dump(collection / "collection.json", {"recipe": recipe, "protocol": protocol, "provenance": provenance,
                                          "run_ids": [r["run_id"] for r in results]})
    return pd.DataFrame(results)
