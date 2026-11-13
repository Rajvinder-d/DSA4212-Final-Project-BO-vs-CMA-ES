"""Plan, check or execute the recorded studies; no computation without --run."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('prepare', 'tests', 'v1', 'v2', 'v3', 'v4', 'v5-original', 'v5-aligned')
COLLECTIONS = {
    'v1': 'pilot_v1', 'v2': 'diagnostics_v2', 'v3': 'rotation-followup-v3',
    'v4': 'rotation-followup-v4', 'v5-original': 'boundary-comparison-v5-jax',
    'v5-aligned': 'boundary-comparison-v5-v4-aligned',
}


def selected_stages(suite, stage):
    if stage != 'all':
        return [stage]
    return [s for s in STAGES if suite == 'all' or s != 'v5-original']


def commands(stage):
    """Delegate to the existing scientific entry points without changing settings."""
    notebook = ['scripts/execute_notebooks.py']
    return {
        'prepare': [
            ['-m', 'ipykernel', 'install', '--prefix', sys.prefix, '--name', 'bo-vs-es'],
            notebook + ['--start', '1', '--end', '1'],
        ],
        'tests': [['-m', 'pytest', '-q']],
        'v1': [notebook + ['--start', '2', '--end', '4']],
        'v2': [['scripts/run_diagnostics.py', '--stage', 'all', '--run']],
        'v3': [
            ['scripts/run_followup.py', '--stage', 'development', '--run'],
            ['scripts/run_followup.py', '--stage', 'confirmation', '--run'],
            notebook + ['--start', '5', '--end', '5'],
        ],
        'v4': [
            ['scripts/record_v4_baseline.py'],
            ['scripts/run_v4.py', '--stage', 'all', '--run'],
        ],
        'v5-original': [
            ['scripts/run_v5.py', '--backend', 'jax', '--run'],
            ['scripts/analyse_v5.py', '--collection',
             f'artifacts/{COLLECTIONS["v5-original"]}/collection.json', '--plots'],
        ],
        'v5-aligned': [
            ['scripts/run_v5_aligned.py', '--run'],
            ['scripts/analyse_v5.py', '--collection',
             f'artifacts/{COLLECTIONS["v5-aligned"]}/collection.json', '--plots'],
        ],
    }[stage]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_export_manifest(path):
    """Verify committed output bytes; this does not validate absent raw runs."""
    record = json.loads(path.read_text())
    for name, expected in record['outputs'].items():
        candidate = path.parent / name
        if not candidate.is_file() or digest(candidate) != expected:
            raise ValueError(f'Saved export missing or modified: {candidate}')
    return len(record['outputs'])


def compare_saved_values(fresh, saved, path='root'):
    """Allow numerical roundoff only; preserve JSON types and exact counts."""
    if type(fresh) is not type(saved):
        raise ValueError(f'{path}: types differ')
    if isinstance(saved, dict):
        if fresh.keys() != saved.keys():
            raise ValueError(f'{path}: keys differ')
        for key in saved:
            compare_saved_values(fresh[key], saved[key], f'{path}.{key}')
    elif isinstance(saved, list):
        if len(fresh) != len(saved):
            raise ValueError(f'{path}: lengths differ')
        for i, (left, right) in enumerate(zip(fresh, saved)):
            compare_saved_values(left, right, f'{path}[{i}]')
    elif isinstance(saved, float):
        if not (math.isfinite(fresh) and math.isfinite(saved)
                and math.isclose(fresh, saved, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f'{path}: numerical values differ or are non-finite: {fresh!r}, {saved!r}')
    elif fresh != saved:
        raise ValueError(f'{path}: values differ: {fresh!r}, {saved!r}')


def verify_saved_results(root=ROOT):
    """Check saved evidence without the numerical environment or CALCE download."""
    manifests = ['followup_v3/validation.json', 'followup_v4/validation.json',
                 'v5/analysis_validation.json', 'v5-aligned/validation.json',
                 'report_review/validation.json']
    checked = sum(verify_export_manifest(root / 'results' / p) for p in manifests)
    coverage = {}
    for name, path, planned, completed in [
        ('V1', 'tables/pilot_runs.csv', 84, 84),
        ('V3', 'followup_v3/run_status.csv', 885, 885),
        ('V4', 'followup_v4/run_status.csv', 3160, 3160),
        ('original V5', 'v5/run_status.csv', 100, 100),
        ('aligned V5', 'v5-aligned/run_status.csv', 280, 279),
    ]:
        with (root / 'results' / path).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        successful = sum(r['status'] == 'complete' for r in rows)
        if (len(rows), successful, len({r['run_id'] for r in rows})) != (planned, completed, planned):
            raise ValueError(f'{name}: saved coverage/unique run IDs disagree with the submitted study')
        coverage[name] = dict(planned=planned, completed=successful)
    notebook_cells = {}
    for path in sorted((root / 'notebooks').glob('[0-9][0-9]_*.ipynb')):
        cells = [c for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code']
        if not cells or any(c.get('execution_count') is None for c in cells):
            raise ValueError(f'Notebook contains unexecuted cells: {path.name}')
        if any(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', [])):
            raise ValueError(f'Notebook contains an execution error: {path.name}')
        notebook_cells[path.name] = len(cells)
    if len(notebook_cells) != 6:
        raise ValueError('Expected all six executed study notebooks')
    output = root / 'artifacts/report-checks'
    subprocess.run([sys.executable, '-B', str(root / 'scripts/analyse_aligned_v5.py'),
                    '--output-dir', str(output)], cwd=root, check=True)
    fresh = json.loads((output / 'report_checks.json').read_text())
    saved = json.loads((root / 'results/v5-aligned/report_checks.json').read_text())
    compare_saved_values(fresh, saved, 'aligned_v5')
    result = dict(status='saved evidence verified', exported_file_hashes=checked,
                  coverage=coverage, executed_notebook_cells=notebook_cells,
                  aligned_contrasts_recomputed=14, experiments_executed=False,
                  limits='Does not verify absent historical raw runs or execute fresh experiments')
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def data_errors(root, reference):
    path = root / 'data/processed/manifest.json'
    if not path.is_file():
        return ['CALCE data absent: run --stage prepare --run first.']
    try:
        manifest = json.loads(path.read_text())
        errors = []
        for name, expected in reference['data_sha256'].items():
            candidate = path.parent / name
            if not candidate.is_file() or digest(candidate) != expected:
                errors.append(f'Processed data missing or changed: {name}')
        if manifest.get('output_sha256') != reference['data_sha256']:
            errors.append('Processed-data manifest hashes differ from the recorded inputs.')
        for key, value in reference['data_settings'].items():
            if manifest.get(key) != value:
                errors.append(f'Preprocessing setting differs: {key}')
        return errors
    except (OSError, ValueError) as exc:
        return [f'Cannot read prepared-data manifest: {exc}']


def preflight(stages, root=ROOT):
    """Read-only checks; importing this script needs only the standard library."""
    errors = []
    reference = json.loads((root / 'configs/followup_v5/v4_reference.json').read_text())
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        errors.append('Use Python >=3.11,<3.14; the recorded primary version is 3.13.6.')
    if os.name == 'nt' and any(s in stages for s in ('v3', 'v4')):
        errors.append('V3/V4 use POSIX locks: run the full suite on Linux/macOS or WSL2.')
    if not shutil.which('git'):
        errors.append('Git is required for execution provenance.')
    elif subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=root,
                        capture_output=True, text=True).returncode:
        errors.append('Use a Git clone, not an extracted source ZIP.')
    versions = {}
    required = dict(reference['packages'])
    required.update({name: None for name in
                     ('matplotlib', 'nbformat', 'nbclient', 'ipykernel', 'pytest', 'filelock')})
    if 'v5-original' in stages:
        required.update(jax='0.12.0.1', jaxlib='0.12.0.1')
    for name, expected in required.items():
        try:
            versions[name] = importlib.metadata.version(name)
            if expected and versions[name] != expected:
                errors.append(f'{name}=={expected} required; found {versions[name]}.')
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            errors.append(f'Missing dependency: {name}')
    sources = dict(reference['source_sha256'])
    sources['configs/protocols/rotation_battery_v1.json'] = reference['protocol_sha256']
    sources['requirements-lock.txt'] = reference['dependency_lock_sha256']
    for name, expected in sources.items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            errors.append(f'Frozen scientific input missing or changed: {name}')
    data = data_errors(root, reference)
    # Preparation may create missing inputs, but must never overwrite suspect ones.
    if 'prepare' not in stages or (root / 'data/processed/manifest.json').exists():
        errors.extend(data)
    if 'prepare' not in stages and any(s in stages for s in ('v1', 'v3', 'v4')):
        kernel = Path(sys.prefix) / 'share/jupyter/kernels/bo-vs-es/kernel.json'
        try:
            executable = json.loads(kernel.read_text())['argv'][0]
            if Path(executable).resolve() != Path(sys.executable).resolve():
                errors.append('The bo-vs-es notebook kernel points to another Python environment.')
        except (OSError, ValueError, KeyError, IndexError):
            errors.append('Notebook kernel missing/invalid: run --stage prepare --run first.')
    if stages == ['v3']:
        errors.extend(v3_prerequisites(root))
    return dict(ready=not errors, python=sys.version.split()[0], packages=versions,
                blockers=errors, data=data or ['Verified against V4 input hashes/settings.'],
                experiments_executed=False)


def v3_prerequisites(root):
    errors = []
    if not (root / 'artifacts/pilot_v1/index.csv').is_file():
        errors.append('V3 requires the V1 pilot histories; run stage v1 first.')
    candidates = sorted((root / 'artifacts/diagnostics_v2').glob('*/manifest.json'))
    if not candidates:
        errors.append('V3 requires a completed V2 collection; run stage v2 first.')
    else:
        try:
            if json.loads(candidates[-1].read_text()).get('status') != 'complete':
                errors.append('The latest V2 collection is incomplete; inspect its error before V3.')
        except (OSError, ValueError) as exc:
            errors.append(f'Cannot read the latest V2 manifest: {exc}')
    return errors


def normalise_prepared_data(root=ROOT):
    """Accept CRLF-to-LF only when it recovers every recorded input byte exactly."""
    reference = json.loads((root / 'configs/followup_v5/v4_reference.json').read_text())
    folder = root / 'data/processed'
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for key, value in reference['data_settings'].items():
        if manifest.get(key) != value:
            raise ValueError(f'Preprocessing setting differs: {key}; no data were changed.')
    replacements = {}
    for name, expected in reference['data_sha256'].items():
        raw = (folder / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() == expected:
            continue
        normalised = raw.replace(b'\r\n', b'\n')
        if hashlib.sha256(normalised).hexdigest() != expected:
            raise ValueError(f'{name} differs beyond line endings; no data were changed.')
        if manifest.get('output_sha256', {}).get(name) != hashlib.sha256(raw).hexdigest():
            raise ValueError(f'{name} does not match its preparation manifest; no data were changed.')
        replacements[name] = normalised
    if replacements:
        for name, content in replacements.items():
            (folder / name).write_bytes(content)
        manifest['output_sha256'] = reference['data_sha256']
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print('Normalised verified CSV line endings: ' + ', '.join(replacements), flush=True)
    errors = data_errors(root, reference)
    if errors:
        raise ValueError('\n'.join(errors))


def execution_environment(root=ROOT):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               JAX_PLATFORMS='cpu', MPLBACKEND='Agg', PYTHONUNBUFFERED='1',
               JUPYTER_PATH=str(Path(sys.prefix) / 'share/jupyter'),
               MPLCONFIGDIR=str(root / 'artifacts/.matplotlib'),
               IPYTHONDIR=str(root / 'artifacts/.ipython'),
               JUPYTER_RUNTIME_DIR=str(root / 'artifacts/.jupyter_runtime'))
    return env


def run_stage(stage, run_command, root=ROOT):
    if stage == 'v3':
        errors = v3_prerequisites(root)
        if errors:
            raise ValueError('\n'.join(errors))
    recipe = commands(stage)
    if stage == 'prepare' and (root / 'data/processed/manifest.json').exists():
        recipe = recipe[:1]  # Keep prepared data and its provenance timestamp intact.
    for index, command in enumerate(recipe):
        code = run_command(command)
        if code:
            # A failed V5 trajectory does not erase the completed trajectories.
            # Export coverage before returning the failure; never silently retry.
            if stage.startswith('v5-') and index == 0:
                collection = root / 'artifacts' / COLLECTIONS[stage] / 'collection.json'
                if collection.exists():
                    run_command(recipe[1])
            return code
    if stage == 'prepare':
        normalise_prepared_data(root)
    if stage.startswith('v5-'):
        collection = root / 'artifacts' / COLLECTIONS[stage] / 'collection.json'
        metadata = json.loads(collection.read_text())
        rows, jobs = metadata.get('rows', {}), metadata.get('jobs', [])
        if (metadata.get('status') != 'complete' or not jobs or len(rows) != len(jobs)
                or any(row.get('status') != 'complete' for row in rows.values())):
            print(f'{stage} is incomplete; partial exports were retained. Resume its native runner.')
            return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['report', 'all'], default='report',
                        help='all also reruns the historical JAX V5 collection')
    parser.add_argument('--stage', choices=['all', *STAGES], default='all',
                        help='select one stage for an existing reproduction checkout')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='check prerequisites without writing files')
    mode.add_argument('--run', action='store_true', help='execute; full suites can take many hours')
    mode.add_argument('--verify-results', action='store_true',
                      help='audit committed exports using only standard-library Python; no experiments')
    args = parser.parse_args()
    if args.verify_results:
        print(json.dumps(verify_saved_results(), indent=2))
        return 0
    stages = selected_stages(args.suite, args.stage)
    if not (args.check or args.run):
        print('Plan only. Use --check for prerequisites; --run starts computation.')
        for stage in stages:
            print(f'\n{stage}:')
            for command in commands(stage):
                print('  ' + shlex.join([sys.executable, *command]))
        print('\nPreparation reuses existing verified data; full suites require a fresh run directory.')
        return 0
    check = preflight(stages)
    print(json.dumps(check, indent=2), flush=True)
    if args.check or not check['ready']:
        return 0 if check['ready'] else 1
    # A full repeat must not recreate V2 inputs beneath an existing V3/V4 run.
    if args.stage == 'all':
        present = [name for name in COLLECTIONS.values()
                   if (ROOT / 'artifacts' / name).exists()]
        if present:
            print('Existing collections: ' + ', '.join(present) +
                  '. Use the stage/recovery commands in docs/REPRODUCING.md; no run started.')
            return 1
    from filelock import FileLock
    folder = ROOT / 'artifacts/reproduction'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_path = folder / f'{stamp}.log'
    print(f'Execution log: {log_path.relative_to(ROOT)}', flush=True)
    with FileLock(str(folder / 'execution.lock'), timeout=0), log_path.open('w', encoding='utf-8') as log:
        def run_command(command):
            line = shlex.join([sys.executable, *command])
            print('\n' + line, flush=True)
            log.write('\n' + line + '\n')
            log.flush()
            with subprocess.Popen([sys.executable, *command], cwd=ROOT,
                                  env=execution_environment(), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors='replace') as child:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            log.write(f'Exit code: {code}\n')
            return code
        for stage in stages:
            code = run_stage(stage, run_command)
            if code:
                print(f'Stopped at {stage} (exit {code}); preserve its artifacts and inspect the log.')
                return code
    print('Selected stages completed. See docs/REPRODUCING.md for output locations and report checks.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f'Reproduction stopped: {exc}', file=sys.stderr)
        raise SystemExit(1)
