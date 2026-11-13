"""Verify a saved collection and regenerate its numerical pilot summaries."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))
from bo_vs_es.analysis.study import load_results, validate_collection, export_results

if __name__ == "__main__":
    results = load_results()
    print(validate_collection(results))
    export_results(results)
