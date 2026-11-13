"""Verify and summarise saved v5 results without running experiments."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', default='artifacts/boundary-comparison-v5/collection.json')
    parser.add_argument('--plots', action='store_true', help='Save convergence and handler-contrast figures')
    parser.add_argument('--plot-backend', choices=['numpy', 'jax'], default='numpy')
    args = parser.parse_args()
    from bo_vs_es.v5_analysis import export
    os.environ['JAX_PLATFORMS'] = 'cpu'
    output = export(ROOT / args.collection, args.plots, args.plot_backend)
    print(output)


if __name__ == '__main__':
    main()
