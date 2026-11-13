"""Prepared v5 design and execution wrapper; no work runs on import.

Original collections use V3 science; the explicitly versioned aligned design
uses frozen V4 science. Neither path alters the shared scientific adapters.
"""
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/followup_v5/boundary_comparison.json'


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config.get('executor') == 'v4-aligned-v1':
        from bo_vs_es.v5_aligned import validate_config
        validate_config(config)
        return config
    if 'executor' in config:
        raise ValueError('Unknown V5 executor; refusing to fall back to original streams')
    seeds = config['initial_design_seeds']
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Initial designs must be nonempty and unique')
    if set(seeds) & set(config['previously_inspected_seeds']):
        raise ValueError('Proposed confirmation seeds overlap previously inspected seeds')
    if config['methods'] != ['bo_iso', 'bo_ard', 'bo_iso_pool', 'random', 'cma_full']:
        raise ValueError('Version the design before changing the declared arms')
    if config['cma_handlers'] != ['resample', 'project'] or config['cma_search_replicates'] != [0, 1, 2]:
        raise ValueError('Both handlers and three CMA replicates are required')
    if (config['n_init'], config['budget'], config['checkpoints']) != (20, 300, [100, 300]):
        raise ValueError('V5 requires 20 initial calls and common checkpoints 100/300')
    if config['targets_rmse_v'] != [.01, .006]:
        raise ValueError('Do not select new targets after inspecting outcomes')
    if config['pool'] != dict(interior=2048, surface=2048, chunk_size=256):
        raise ValueError('Preserve the existing finite-pool intervention')
    if config['bootstrap_resamples'] < 2:
        raise ValueError('Bootstrap requires at least two resamples')
    if Path(config['id']).name != config['id'] or config['id'] in {'.', '..'}:
        raise ValueError('Collection ID must be one directory name')
    return config


def jobs_for(config):
    if config.get('executor') == 'v4-aligned-v1':
        from bo_vs_es.v5_aligned import jobs_for as aligned_jobs
        return aligned_jobs(config)
    if 'executor' in config:
        raise ValueError('Unknown V5 executor')
    jobs = []
    for seed in config['initial_design_seeds']:
        group = []
        for method in config['methods']:
            cma = method == 'cma_full'
            for replicate in config['cma_search_replicates'] if cma else [0]:
                # Independent stream namespace; never conditioned on the handler.
                search_seed = int(identity([config['search_seed_namespace'], seed, replicate])[:8], 16)
                search_seed = search_seed % (2**31 - 2) + 1
                for handler in config['cma_handlers'] if cma else ['none']:
                    group.append(dict(block='battery_ball_v5', objective='battery', domain='ball',
                        condition=None, initial_design_seed=seed, search_replicate=replicate,
                        search_seed=search_seed, rotation_seed=None, method=method, handler=handler,
                        budget=config['budget'], n_init=config['n_init'],
                        checkpoints=config['checkpoints'], corrected=True))
        # Avoid always timing BO before CMA, while completing whole design groups.
        random.Random(config['search_seed_namespace'] + seed).shuffle(group)
        jobs.extend(group)
    return jobs


def planned_counts(config):
    jobs = jobs_for(config)
    return dict(trajectories=len(jobs), search_calls=sum(j['budget'] for j in jobs),
                assessment_calls=sum(2 * len(j['checkpoints']) for j in jobs),
                initial_designs=len(config['initial_design_seeds']),
                note='Planned counts only; no v5 results have been generated')


def interval(values, config):
    """Percentile bootstrap over independent initial-design summaries."""
    values = list(values)
    if not values:
        return dict(design_groups=0, estimate=None, low=None, high=None)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite design-level contrast')
    result = dict(design_groups=len(values), estimate=statistics.mean(values), low=None, high=None)
    if len(values) < 2:
        return result
    rng = random.Random(config['bootstrap_seed'])
    samples = sorted(statistics.mean(rng.choices(values, k=len(values)))
                     for _ in range(config['bootstrap_resamples']))
    def quantile(p):
        index = (len(samples) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(samples) - 1)
        return samples[lo] + (index - lo) * (samples[hi] - samples[lo])
    result.update(low=quantile(.025), high=quantile(.975))
    return result


def contrasts(snapshots, complete_seeds, config):
    """Collapse CMA replicates inside designs, never treating them as designs."""
    rows = []
    for calls in config['checkpoints']:
        values = {}
        for seed in complete_seeds:
            group = [r for r in snapshots if r['initial_design_seed'] == seed and r['calls'] == calls]
            losses = {}
            for row in group:
                loss = row['loss']
                if not math.isfinite(loss) or loss <= 0:
                    raise ValueError('Log-ratio analysis requires finite positive losses')
                losses.setdefault((row['method'], row['handler']), []).append(math.log10(loss))
            expected = {(m, 'none'): 1 for m in config['methods'] if m != 'cma_full'}
            expected.update({('cma_full', h): len(config['cma_search_replicates'])
                             for h in config['cma_handlers']})
            if {k: len(v) for k, v in losses.items()} != expected:
                raise ValueError('Incomplete or duplicate design checkpoint')
            for method, handler in expected:
                actual = [r['search_replicate'] for r in group
                          if (r['method'], r['handler']) == (method, handler)]
                declared = config['cma_search_replicates'] if method == 'cma_full' else [0]
                if sorted(actual) != sorted(declared):
                    raise ValueError('Incomplete or duplicate search replicates')
            means = {k: statistics.mean(v) for k, v in losses.items()}
            values.setdefault('CMA projection / CMA resampling', []).append(
                means['cma_full', 'project'] - means['cma_full', 'resample'])
            for handler in config['cma_handlers']:
                for bo in ['bo_iso', 'bo_ard', 'bo_iso_pool']:
                    values.setdefault(f'CMA {handler} / {bo}', []).append(
                        means['cma_full', handler] - means[bo, 'none'])
        for name, observations in values.items():
            rows.append(dict(contrast=name, calls=calls,
                role='primary' if calls == 100 and name == 'CMA projection / CMA resampling' else 'secondary',
                metric='mean design-level log10 MSE ratio', **interval(observations, config)))
    return rows


def session_limit_reached(started, now, max_seconds):
    return max_seconds is not None and now - started >= max_seconds


def run(config, environment, max_seconds, execution_label, backend='numpy'):
    """Optional per-invocation soft limit; finish a trajectory before pausing."""
    import importlib.metadata
    import platform
    import signal
    import time
    import traceback
    import uuid

    import torch
    from filelock import FileLock
    from bo_vs_es.followup import atomic_dump, execute_job, now, provenance
    from bo_vs_es.objectives.data import sha256

    alignment = None
    if config.get('executor') == 'v4-aligned-v1':
        from bo_vs_es.v5_aligned import verify_alignment
        alignment = verify_alignment(environment)
        from bo_vs_es.followup_v4 import execute_job

    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError('Session limit must be positive, or omitted for no time limit')
    if backend not in {'numpy', 'jax'}:
        raise ValueError('Unknown objective backend')
    protocol = json.loads((ROOT / config['protocol']).read_text())
    if protocol['cma'] != dict(population=8, sigma0=.3, max_proposals_per_point=10000):
        raise ValueError('V5 preserves the v1 CMA settings; version any tuning separately')
    if protocol['bo'] != dict(fit_maxiter=60, acq_maxiter=80, raw_samples=256, num_restarts=8):
        raise ValueError('V5 preserves the v1 continuous BO settings')
    torch.set_num_threads(1)
    jobs = jobs_for(config)
    prov = provenance(config, environment)
    prov['execution'] = execution_label
    prov['python'] = platform.python_version()
    prov['v5_script_sha256'] = sha256(ROOT / 'scripts/run_v5.py')
    prov['protocol_sha256'] = sha256(ROOT / config['protocol'])
    prov['objective_backend'] = backend
    prov['filelock_version'] = importlib.metadata.version('filelock')
    if alignment is not None:
        prov['v4_alignment'] = alignment
        prov['aligned_script_sha256'] = sha256(ROOT / 'scripts/run_v5_aligned.py')
    if backend == 'jax':
        from bo_vs_es.v5_jax import cpu_float64
        device = cpu_float64()
        prov['jax'] = dict(version=importlib.metadata.version('jax'),
                           jaxlib=importlib.metadata.version('jaxlib'),
                           platform=device.platform, device_kind=device.device_kind, float64=True)
    prov_id = identity(prov)
    root = ROOT / 'artifacts' / config['id']
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'execution.lock'), timeout=0):
        collection = root / 'collection.json'
        if collection.exists():
            metadata = json.loads(collection.read_text())
            if metadata['provenance_id'] != prov_id:
                raise ValueError('Collection source/settings/backend/environment changed; use a new --collection-id')
        else:
            metadata = dict(config=config, protocol=protocol, provenance=prov, provenance_id=prov_id,
                            jobs=jobs, rows={}, status='pending')
        attempts = root / 'invocations'
        attempts.mkdir(exist_ok=True)
        previous = [json.loads(p.read_text()) for p in attempts.glob('*.json')]
        if any(p['status'] == 'running' for p in previous):
            raise RuntimeError('An unclosed invocation needs investigation before restart; do not delete its record')
        spent = sum(p['elapsed_s'] for p in previous)
        invocation_path = attempts / f'{uuid.uuid4().hex}.json'
        start = time.monotonic()
        status = dict(status='running', started_utc=now(), elapsed_s=0., session_limit_s=max_seconds,
                      previous_elapsed_s=spent,
                      execution=execution_label, provenance_id=prov_id)
        atomic_dump(invocation_path, status)
        metadata['status'] = 'running'
        atomic_dump(collection, metadata)
        stop_requested = False
        def request_stop(signum, frame):
            nonlocal stop_requested
            if stop_requested:
                raise KeyboardInterrupt('Immediate interruption requested; partial attempt retained')
            stop_requested = True
            print('Stop requested: finishing the current trajectory. A second Ctrl+C interrupts it.', flush=True)
        previous_handler = signal.signal(signal.SIGINT, request_stop)
        try:
            objective = None
            if backend == 'jax':
                from bo_vs_es.v5_jax import prepare_objective
                objective, setup = prepare_objective()
                status['objective_setup'] = setup
                atomic_dump(invocation_path, status)
            runs = root / 'runs'
            runs.mkdir(exist_ok=True)
            for index, job in enumerate(jobs):
                if stop_requested or session_limit_reached(start, time.monotonic(), max_seconds):
                    status['status'] = 'paused'
                    break
                # No in-trajectory deadline: a laptop session limit never throws
                # away an otherwise healthy 300-call trajectory.
                row, cached = execute_job(job, config, protocol, runs, prov, objective=objective)
                metadata['rows'][identity(job)] = row
                atomic_dump(collection, metadata)
                status.update(elapsed_s=time.monotonic() - start, updated_utc=now())
                atomic_dump(invocation_path, status)
                print(f"[{index + 1}/{len(jobs)}] design={job['initial_design_seed']} "
                      f"{job['method']} {job['handler']} rep={job['search_replicate']}: "
                      f"{row['status']} cached={cached}", flush=True)
            else:
                status['status'] = 'complete' if all(r['status'] == 'complete' for r in metadata['rows'].values()) else 'failed'
        except BaseException as exc:
            if hasattr(exc, 'report'):
                status['objective_setup'] = exc.report
            status.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                          error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            status.update(elapsed_s=time.monotonic() - start, finished_utc=now())
            metadata['status'] = status['status']
            metadata['total_elapsed_s'] = spent + status['elapsed_s']
            atomic_dump(invocation_path, status)
            atomic_dump(collection, metadata)
    return 0 if status['status'] in {'complete', 'paused'} else 1
