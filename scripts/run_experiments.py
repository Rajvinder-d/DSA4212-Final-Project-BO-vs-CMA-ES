"""Run an explicitly selected recipe; the main study never runs by default."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from bo_vs_es.experiments.runner import run_collection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/pilot.json")
    args = parser.parse_args()
    results = run_collection(args.config)
    if not results.status.eq("complete").all():
        raise SystemExit("Some runs failed; see preserved error.txt files and the collection index")
