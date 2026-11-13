"""Execute the four study notebooks in separate fresh kernels, saving real outputs."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/.matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / "artifacts/.ipython"))
os.environ.setdefault("JUPYTER_PATH", str(ROOT / ".venv/share/jupyter"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / "artifacts/.jupyter_runtime"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import nbformat
from nbclient import NotebookClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=4)
    args = parser.parse_args()
    for path in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        if not args.start <= int(path.name[:2]) <= args.end:
            continue
        print(f"Executing {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=3600, kernel_name="bo-vs-es",
                                resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
        print(f"Saved {path.name}", flush=True)
