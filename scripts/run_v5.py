"""Print the v5 plan by default; --run explicitly starts measured-data work."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(default_config='configs/followup_v5/boundary_comparison.json'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--max-seconds', type=float, help='Optional session limit; finish the current trajectory before pausing')
    parser.add_argument('--backend', choices=['numpy', 'jax'], default='numpy', help='Battery objective backend; both use CPU')
    parser.add_argument('--collection-id', help='Separate collection for another machine or backend')
    parser.add_argument('--execution-label', default='user-launched', help='Attribution, not evidence of understanding')
    args = parser.parse_args()
    from bo_vs_es.v5 import load_config, planned_counts, run
    config = load_config(ROOT / args.config)
    if args.collection_id:
        if Path(args.collection_id).name != args.collection_id or args.collection_id in {'.', '..'}:
            parser.error('--collection-id must be a single directory name')
        config['id'] = args.collection_id
    elif args.backend == 'jax':
        config['id'] += '-jax'
    if not args.run:
        print(json.dumps(dict(config=config, plan=planned_counts(config), backend=args.backend,
                              session_limit_s=args.max_seconds, executed=False), indent=2))
        return 0
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'
    from run_diagnostics import readiness
    check = readiness()
    for name in ['filelock'] + (['jax', 'jaxlib'] if args.backend == 'jax' else []):
        if importlib.util.find_spec(name) is None:
            check['blockers'].append(f'Missing optional package: {name}; install the laptop extra')
    check['ready'] = not check['blockers']
    if not check['ready']:
        print(json.dumps(check, indent=2))
        return 1
    return run(config, check, args.max_seconds, args.execution_label, args.backend)


if __name__ == '__main__':
    raise SystemExit(main())
