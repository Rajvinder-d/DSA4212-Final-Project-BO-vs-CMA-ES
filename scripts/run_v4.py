"""V4 readiness, charged development/confirmation, and persistent complete pipeline."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts/.matplotlib'))
from run_diagnostics import readiness
from bo_vs_es import followup_v4 as f
from bo_vs_es.objectives.data import sha256


def finish_analysis(config):
    from bo_vs_es.followup_v4_analysis import export_analysis, audit_notebook
    audit = export_analysis()
    subprocess.run([sys.executable, str(ROOT / 'scripts/execute_notebooks.py'), '--start', '6', '--end', '6'],
                   cwd=ROOT, check=True)
    return audit_notebook(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['readiness', 'development', 'confirmation', 'analysis', 'all'], default='readiness')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = f.load_config()
    check = readiness()
    jobs = f.jobs_for(config)
    if args.stage == 'readiness' or not args.run:
        print(json.dumps(dict(**check, planned_runs=len(jobs), search_calls=sum(j['budget'] for j in jobs),
            assessment_calls=sum(2 * len(j['checkpoints']) for j in jobs if j['objective'] == 'battery'),
            allowance_seconds=config['max_compute_seconds']), indent=2))
        return 0 if check['ready'] else 1
    if not check['ready']:
        raise RuntimeError(str(check['blockers']))
    root = ROOT / 'artifacts' / config['id']
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'execution.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'analysis':
            print(json.dumps(finish_analysis(config), indent=2))
            return 0
        f.torch.set_num_threads(1)
        prov = f.provenance(config, check)
        prov_id = hashlib.sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()
        invocations = root / 'invocations'
        invocations.mkdir(exist_ok=True)
        previous = [json.loads(p.read_text()) for p in invocations.glob('*.json')]
        if any(p['status'] == 'running' for p in previous):
            raise RuntimeError('Interrupted invocation requires investigation; reconcile its elapsed time before restarting')
        if any(p['provenance_id'] != prov_id for p in previous):
            raise RuntimeError('Scientific provenance changed; version the study rather than mixing runs')
        spent = sum(p['compute_elapsed_s'] for p in previous)
        if spent >= config['max_compute_seconds']:
            raise RuntimeError('V4 compute allowance exhausted; analysis-only remains available')
        invocation_path = invocations / (f.now().replace(':', '').replace('.', '') + '.json')
        start = time.monotonic()
        deadline = start + config['max_compute_seconds'] - spent
        status = dict(status='running', stage=args.stage, pid=os.getpid(), started_utc=f.now(),
            provenance_id=prov_id, previous_compute_s=spent, compute_elapsed_s=0.,
            allowance_s=config['max_compute_seconds']-spent, planned_confirmation_runs=len(jobs))
        active = True
        def persist():
            if active:
                status['compute_elapsed_s'] = time.monotonic() - start
            status['updated_utc'] = f.now()
            f.atomic_dump(invocation_path, status)
            f.atomic_dump(root / 'status.json', status)
        def interrupted(signum, frame):
            raise SystemExit(f'Interrupted by signal {signum}; records preserved')
        signal.signal(signal.SIGTERM, interrupted)
        persist()
        protocol = json.loads((ROOT / config['protocol']).read_text())

        def collection(stage, stage_jobs, forecast=None):
            folder = root / stage
            folder.mkdir(exist_ok=True)
            path = folder / 'collection.json'
            if path.exists():
                metadata = json.loads(path.read_text())
                if metadata['provenance_id'] != prov_id or metadata['jobs'] != stage_jobs:
                    raise ValueError('Collection identity changed')
            else:
                metadata = dict(provenance=prov, provenance_id=prov_id, jobs=stage_jobs, rows=[])
            old_rows = {json.dumps({k: row[k] for k in stage_jobs[0]}, sort_keys=True): row for row in metadata['rows']}
            selected = []
            launched_seconds = 0.
            stage_start = time.monotonic()
            status.update(stage=stage, stage_planned=len(stage_jobs), stage_completed=0)
            for i, job in enumerate(stage_jobs):
                f.check_deadline(deadline)
                if i == 0 or job['initial_design_seed'] != stage_jobs[i-1]['initial_design_seed'] or job['block'] != stage_jobs[i-1]['block']:
                    f.verify_scientific_inputs(prov)
                status['current_job'] = job
                persist()
                row, cached = f.execute_job(job, config, protocol, folder, prov, deadline)
                selected.append(row)
                if not cached:
                    launched_seconds += row['attempt_s']
                old_rows[json.dumps(job, sort_keys=True)] = row
                metadata['rows'] = list(old_rows.values())  # Never erase later records during a restart.
                metadata['last_updated_utc'] = f.now()
                f.atomic_dump(path, metadata)
                status['stage_completed'] = sum(r['status'] == 'complete' for r in selected)
                status['stage_attempted'] = i + 1
                if forecast:
                    estimates = {tuple(v['key']): v['seconds'] for v in forecast['estimates']}
                    status['estimated_remaining_s'] = sum(estimates[f.timing_key(j)] for j in stage_jobs[i+1:])
                persist()
                print(f'[{stage} {i+1}/{len(stage_jobs)}] {job["block"]} {job["method"]} '
                      f'design={job["initial_design_seed"]} Q={job["rotation_seed"]} upper={job["lengthscale_upper"]}: '
                      f'{row["status"]} ({"cached" if cached else str(round(row["attempt_s"], 2))+"s"})', flush=True)
                if row['status'] == 'budget_limited':
                    raise TimeoutError('V4 allowance reached during a run')
                if stage == 'development' and row['status'] != 'complete':
                    raise RuntimeError('Development failure; confirmation is blocked')
            metadata['runner_overhead_s_per_job'] = max(0., time.monotonic()-stage_start-launched_seconds)/len(stage_jobs)
            f.atomic_dump(path, metadata)
            return metadata

        exit_code = 0
        try:
            if args.stage in ['development', 'all']:
                dev = collection('development', f.jobs_for(config, development=True))
                status['stage'] = 'replays'
                persist()
                replays = f.development_replays(config, dev['rows'], root / 'development/replays.json', deadline)
                f.atomic_dump(root / 'development_complete.json', dict(provenance_id=prov_id,
                    collection_sha256=sha256(root / 'development/collection.json'),
                    replays_sha256=sha256(root / 'development/replays.json'), replay_records=len(replays)))
            if args.stage in ['confirmation', 'all']:
                mark = json.loads((root / 'development_complete.json').read_text())
                if (mark['provenance_id'] != prov_id or mark['collection_sha256'] != sha256(root / 'development/collection.json')
                    or mark['replays_sha256'] != sha256(root / 'development/replays.json')):
                    raise ValueError('Development provenance/hash mismatch')
                dev = json.loads((root / 'development/collection.json').read_text())
                for row in dev['rows']:
                    folder = ROOT / row['run_path']
                    manifest = json.loads((folder/'manifest.json').read_text())
                    if not f.valid_completed(folder, manifest['run_key']):
                        raise ValueError('Invalid development artifact')
                # Cached confirmation runs need no new runtime allocation.
                pending = jobs
                cp = root / 'confirmation/collection.json'
                if cp.exists():
                    prior = json.loads(cp.read_text())
                    if prior['provenance_id'] != prov_id:
                        raise ValueError('Confirmation provenance mismatch')
                    completed = set()
                    for row in prior['rows']:
                        folder = ROOT / row['run_path']
                        manifest = json.loads((folder/'manifest.json').read_text())
                        if f.valid_completed(folder, manifest['run_key']):
                            completed.add(json.dumps({k:row[k] for k in jobs[0]},sort_keys=True))
                    pending = [j for j in jobs if json.dumps(j,sort_keys=True) not in completed]
                forecast = f.timing_forecast(dev['rows'], pending, deadline-time.monotonic(), dev['runner_overhead_s_per_job'])
                f.atomic_dump(root / 'timing_gate.json', forecast)
                print('Timing gate:', json.dumps(forecast), flush=True)
                if not forecast['fits_allowance']:
                    status.update(status='timing_gate_blocked', stage='timing_gate', forecast=forecast,
                        reduced_scope_proposal='Keep 20 primary battery/CMA designs; reduce supporting designs from 10 to 5 in a separately approved protocol revision')
                    exit_code = 2
                else:
                    confirmed = collection('confirmation', jobs, forecast)
                    status['status'] = 'experiments_complete' if all(r['status']=='complete' for r in confirmed['rows']) else 'experiments_failed'
                    exit_code = 0 if status['status']=='experiments_complete' else 1
            else:
                status['status'] = 'development_complete'
        except BaseException as exc:
            status.update(status='budget_limited' if isinstance(exc, TimeoutError) else 'failed', error=repr(exc), traceback=traceback.format_exc())
            exit_code = 1
        finally:
            persist()
            active = False
            status['science_finished_utc'] = f.now()
            status['exit_code'] = exit_code
            persist()
        if args.stage == 'all':
            science_status = status['status']
            try:
                status.update(status='running', stage='analysis', science_status=science_status)
                persist()
                audit = finish_analysis(config)
                status.update(status='complete' if audit['status']=='complete' and exit_code==0 else science_status,
                              stage='finished', analysis_audit=audit)
            except BaseException as exc:
                status.update(status='analysis_failed', science_status=science_status, error=repr(exc), traceback=traceback.format_exc())
                exit_code = 1
        status.update(finished_utc=f.now(), exit_code=exit_code)
        persist()
        print(json.dumps(status, indent=2), flush=True)
        return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
