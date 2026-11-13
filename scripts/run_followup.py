"""Readiness by default; explicit bounded development or confirmation execution."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts/.matplotlib'))

from run_diagnostics import readiness


def elapsed(manifest):
    return (datetime.fromisoformat(manifest['finished_utc']) -
            datetime.fromisoformat(manifest['started_utc'])).total_seconds()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/followup_v3.json')
    parser.add_argument('--stage', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--v2-artifacts', help='Completed original v2 diagnostic directory')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    from bo_vs_es.followup import (load_config, jobs_for, provenance, atomic_dump, now,
                                  development, execute_job, check_deadline)
    from bo_vs_es.objectives.data import sha256
    import torch
    config = load_config(ROOT / args.config)
    check = readiness()
    jobs = jobs_for(config)
    if not args.run:
        print(json.dumps(dict(**check, planned_runs=len(jobs),
            planned_search_calls=sum(j['budget'] for j in jobs),
            stage=args.stage, max_compute_seconds=config['max_compute_seconds']), indent=2))
        return 0 if check['ready'] else 1
    if not check['ready']:
        raise RuntimeError(str(check['blockers']))
    torch.set_num_threads(1)
    root = ROOT / 'artifacts' / config['id']
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'execution.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        invocations = root / 'invocations'
        invocations.mkdir(exist_ok=True)
        previous = [json.loads(p.read_text()) for p in invocations.glob('*.json')]
        if any(p['status'] == 'running' for p in previous):
            raise RuntimeError('An interrupted invocation needs investigation before restart; preserve its record')
        prov = provenance(config, check)
        prov_id = hashlib.sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()
        stamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
        invocation_path = invocations / f'{stamp}.json'
        v2_candidates = sorted((ROOT / 'artifacts/diagnostics_v2').glob('*/manifest.json'))
        v2_folder = Path(args.v2_artifacts).resolve() if args.v2_artifacts else v2_candidates[-1].parent
        v2_manifest = json.loads((v2_folder / 'manifest.json').read_text())
        if v2_manifest['status'] != 'complete':
            raise RuntimeError('V2 diagnostics must be complete')
        v2_seconds = elapsed(v2_manifest)
        spent = v2_seconds + sum(p['elapsed_s'] for p in previous)
        allowance = config['max_compute_seconds'] - spent
        if args.stage == 'development':
            allowance = min(allowance, config['development_max_seconds'])
        if allowance <= 0:
            raise RuntimeError('Four-hour study allowance exhausted; no new experiments started')
        started = time.monotonic()
        deadline = started + allowance
        status = dict(stage=args.stage, status='running', started_utc=now(), pid=os.getpid(),
                      provenance_id=prov_id, elapsed_s=0., previous_compute_s=spent,
                      allowance_s=allowance, planned_runs=len(jobs), completed_runs=0,
                      v2_source=str(v2_folder.relative_to(ROOT)))
        atomic_dump(invocation_path, status)
        def persist():
            status['elapsed_s'] = time.monotonic() - started
            status['updated_utc'] = now()
            atomic_dump(invocation_path, status)
            atomic_dump(root / 'status.json', status)
        def stop(signum, frame):
            raise KeyboardInterrupt('Received termination signal; preserving artifacts')
        signal.signal(signal.SIGTERM, stop)
        persist()
        exit_code = 0
        try:
            if args.stage == 'development':
                folder = root / 'development' / stamp
                folder.mkdir(parents=True)
                atomic_dump(folder / 'provenance.json', prov)
                status['result_path'] = str(folder.relative_to(ROOT))
                outcome = development(config, folder, v2_folder, deadline)
                status.update(status='complete', outcome=outcome,
                    output_sha256={p.name: sha256(p) for p in folder.iterdir()})
                atomic_dump(root / 'development_complete.json', status)
            else:
                dev = json.loads((root / 'development_complete.json').read_text())
                if dev['status'] != 'complete' or dev['provenance_id'] != prov_id:
                    raise RuntimeError('Revalidate development after scientific source/configuration changes')
                for name, digest in dev['output_sha256'].items():
                    if sha256(ROOT / dev['result_path'] / name) != digest:
                        raise RuntimeError('Modified development evidence')
                collection = root / 'confirmation'
                collection.mkdir(exist_ok=True)
                collection_path = collection / 'collection.json'
                if collection_path.exists():
                    old = json.loads(collection_path.read_text())
                    if old['provenance_id'] != prov_id:
                        raise RuntimeError('Scientific settings changed; version the study ID before execution')
                metadata = dict(config=config, provenance=prov, provenance_id=prov_id, jobs=jobs,
                                rows=[], development_source=dev['result_path'])
                atomic_dump(collection_path, metadata)
                protocol = json.loads((ROOT / config['v1_protocol']).read_text())
                for index, job in enumerate(jobs):
                    check_deadline(deadline)
                    status['current_job'] = job
                    persist()
                    result, cached = execute_job(job, config, protocol, collection, prov, deadline)
                    metadata['rows'].append(result)
                    atomic_dump(collection_path, metadata)
                    status['completed_runs'] = sum(r['status'] == 'complete' for r in metadata['rows'])
                    status['attempted_jobs'] = len(metadata['rows'])
                    persist()
                    print(f"[{index+1}/{len(jobs)}] {job['block']} {job['method']} design={job['initial_design_seed']} "
                          f"rep={job['search_replicate']} Q={job['rotation_seed']} {job['handler']}: "
                          f"{result['status']} ({'cached' if cached else str(round(result['attempt_s'], 2)) + 's'})", flush=True)
                    if result['status'] == 'budget_limited':
                        raise TimeoutError('Study allowance reached during a run')
                status['status'] = 'complete' if all(r['status'] == 'complete' for r in metadata['rows']) else 'failed'
                exit_code = 0 if status['status'] == 'complete' else 1
        except BaseException as exc:
            status.update(status='budget_limited' if isinstance(exc, TimeoutError) else 'failed',
                          error=repr(exc), traceback=traceback.format_exc())
            exit_code = 2 if isinstance(exc, TimeoutError) else 1
        finally:
            status['finished_utc'] = now()
            status['exit_code'] = exit_code
            persist()
            print(json.dumps(status, indent=2), flush=True)
        return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
