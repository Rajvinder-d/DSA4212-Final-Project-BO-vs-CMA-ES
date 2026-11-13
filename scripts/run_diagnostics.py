"""Check readiness or run bounded development diagnostics, separately from v1."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PACKAGES = ["numpy", "scipy", "pandas", "torch", "botorch", "gpytorch", "cma", "openpyxl"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def readiness(root=ROOT):
    """Read-only; never downloads data, installs packages, or starts experiments."""
    blockers, versions = [], {}
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        blockers.append("Experiment requires Python >=3.11,<3.14")
    for name in PACKAGES:
        if importlib.util.find_spec(name) is None:
            versions[name] = None
            blockers.append(f"Missing package: {name}")
        else:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "unknown"
    path = root / "data/processed/manifest.json"
    if not path.exists():
        blockers.append("Missing prepared CALCE data/processed/manifest.json")
    else:
        try:
            manifest = json.loads(path.read_text())
            for name in ["dst80.csv", "fuds80.csv", "fuds50.csv", "ocv.csv"]:
                candidate = path.parent / name
                if not candidate.exists() or digest(candidate) != manifest["output_sha256"].get(name):
                    blockers.append(f"Missing/modified processed data: {name}")
            audit = json.loads((root / "results/tables/delivery_validation.json").read_text())
            if manifest["output_sha256"] != audit["data_sha256"]:
                blockers.append("Prepared data differ from the recorded v1 inputs")
        except (OSError, KeyError, ValueError) as exc:
            blockers.append(f"Invalid data manifest: {exc}")
    audit_path = root / "results/tables/delivery_validation.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        paths = [p for name in ["objectives", "optimisers", "experiments"]
                 for p in sorted((root / "src/bo_vs_es" / name).glob("*.py"))]
        fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
        if fingerprint != audit["code_sha256"]:
            blockers.append("V1 source fingerprint changed; review diagnostic parity before running")
    return {"ready": not blockers, "python": sys.version.split()[0],
            "packages": versions, "blockers": blockers,
            "experiments_executed": False}


def reconstruct_history(seed, objective, config, protocol, folder, deadline):
    """Generate a short new identity BO history; never claim it is an old run."""
    import numpy as np
    import torch
    from bo_vs_es.objectives.models import Domain
    from bo_vs_es.optimisers.adapters import bo_candidate
    domain = Domain("ball", np.eye(5))
    points = list(domain.sample(np.random.default_rng(seed), config["n_init"]))
    losses, records = [], []
    search_rng = np.random.default_rng(np.random.SeedSequence([seed, 41]))
    torch.manual_seed(seed + 10000)
    for index in range(max(config["checkpoints"])):
        if time.monotonic() >= deadline:
            raise TimeoutError("Diagnostic wall-time allowance reached")
        diagnostic = None
        if index >= config["n_init"]:
            candidate, diagnostic = bo_candidate(points, losses, domain, False,
                                                 protocol["bo"], search_rng)
            points.append(candidate)
        start = time.perf_counter()
        row = {"evaluation": index + 1, "u": points[index].tolist(),
               "status": "failed", "bo_diagnostic": diagnostic}
        try:
            loss = float(objective(points[index]))
            if not np.isfinite(loss):
                raise ValueError("Nonfinite calibration loss")
            losses.append(loss)
            row.update(loss=loss, status="ok")
        except Exception as exc:
            row["error"] = repr(exc)
            raise
        finally:
            row["objective_s"] = time.perf_counter() - start
            records.append(row)
            dump(folder / f"history_seed{seed}.json", {
                "seed": seed, "status": "complete" if len(losses) == max(config["checkpoints"]) else "partial",
                "scope": "new identity-BO history using v1 settings; not the original macOS trajectory",
                "evaluations": records})
    return np.asarray(points), np.asarray(losses)


def run(config_path, stage):
    config = json.loads(config_path.read_text())
    from bo_vs_es.diagnostics.core import validate_config
    validate_config(config)
    check = readiness()
    if not check["ready"]:
        raise RuntimeError("Preflight failed:\n" + "\n".join(check["blockers"]))
    protocol = json.loads((ROOT / config["v1_protocol"]).read_text())
    if protocol["id"] != "rotation-battery-v1":
        raise ValueError("Expected v1 source protocol")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "artifacts/diagnostics_v2" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    code_paths = sorted((ROOT / "src/bo_vs_es").rglob("*.py")) + [Path(__file__)]
    manifest = {"id": config["id"], "stage": stage, "status": "running",
                "environment": check, "started_utc": stamp,
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in code_paths},
                "config_sha256": digest(config_path),
                "v1_protocol_sha256": digest(ROOT / config["v1_protocol"]),
                "data_manifest_sha256": digest(ROOT / "data/processed/manifest.json"),
                "lock_sha256": digest(ROOT / "requirements-lock.txt"),
                "max_wall_seconds": config["max_wall_seconds"],
                "wall_limit_policy": "checked between solver calls; an in-flight solve may exceed the deadline"}
    dump(folder / "manifest.json", manifest)
    dump(folder / "config.json", config)
    dump(folder / "v1_protocol.json", protocol)
    deadline = time.monotonic() + config["max_wall_seconds"]
    try:
        from bo_vs_es.objectives.models import Battery
        objective = Battery(corrected=True)
        if stage in ("replay", "all"):
            from bo_vs_es.diagnostics.replay import replay_history
            rows = []
            for seed in config["initial_design_seeds"]:
                points, losses = reconstruct_history(seed, objective, config, protocol, folder, deadline)
                for checkpoint in config["checkpoints"]:
                    for rot in config["rotation_seeds"]:
                        print(f"Replay: seed={seed}, observations={checkpoint}, rotation={rot}", flush=True)
                        result = replay_history(points[:checkpoint], losses[:checkpoint], rot,
                                                config, protocol["bo"], deadline)
                        result["history_seed"] = seed
                        rows.append(result)
                        dump(folder / "replays.json", rows)
        if stage in ("reference", "all"):
            from bo_vs_es.diagnostics.core import ball_reference
            print("Counted unit-ball reference", flush=True)
            reference = ball_reference(objective.residual, 5, config["reference"], deadline)
            dump(folder / "ball_reference.json", reference)
            if reference["status"] != "complete" or reference["best"] is None:
                raise RuntimeError("Reference failed; inspect recorded callback trace")
            if time.monotonic() >= deadline:
                raise TimeoutError("Wall allowance reached during reference fit")
        manifest["status"] = "complete"
        manifest["interpretation"] = "diagnostic execution completed; not a confirmed mechanism or global optimum"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        (folder / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["output_sha256"] = {p.name: digest(p) for p in folder.iterdir() if p.name != "manifest.json"}
        dump(folder / "manifest.json", manifest)
        print(f"Diagnostic artifacts: {folder.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/diagnostics/development_v2.json")
    parser.add_argument("--stage", choices=["replay", "reference", "all"], default="all")
    parser.add_argument("--run", action="store_true", help="Execute the selected stage; default is readiness only")
    args = parser.parse_args()
    if args.run:
        path = Path(args.config)
        run(path if path.is_absolute() else ROOT / path, args.stage)
    else:
        check = readiness()
        print(json.dumps(check, indent=2))
        raise SystemExit(0 if check["ready"] else 1)
