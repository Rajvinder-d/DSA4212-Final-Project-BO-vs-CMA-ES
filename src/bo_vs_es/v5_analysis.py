"""Read saved v5 results; never evaluate an objective or import an optimiser."""
import csv
import hashlib
import json
import math
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from bo_vs_es.v5 import ROOT, contrasts, identity, jobs_for


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_run(root, row, job, metadata):
    folder = ROOT / row['run_path']
    if not folder.resolve().is_relative_to((root / 'runs').resolve()):
        raise ValueError('Run is outside the selected collection')
    manifest = json.loads((folder / 'manifest.json').read_text())
    resolved = dict(job=job, protocol=metadata['protocol'], provenance=metadata['provenance'])
    if manifest['run_key'] != identity(resolved):
        raise ValueError('Run identity does not match collection provenance')
    if json.loads((folder / 'config.json').read_text()) != resolved:
        raise ValueError('Resolved run settings differ from the collection')
    if manifest['status'] != row['status']:
        raise ValueError('Manifest and summary disagree on run status')
    for name, expected in manifest['output_sha256'].items():
        if digest(folder / name) != expected:
            raise ValueError(f'Modified output: {folder / name}')
    if json.loads((folder / 'summary.json').read_text()) != row:
        raise ValueError('Collection and run summary disagree')
    return folder


def export(collection_path, plots=False, plot_backend='numpy'):
    """Hold the writer's collection lock while reading a consistent snapshot."""
    from filelock import FileLock
    collection_path = Path(collection_path).resolve()
    with FileLock(str(collection_path.parent / 'execution.lock'), timeout=0):
        return _export_locked(collection_path, plots, plot_backend)


def _export_locked(collection_path, plots, plot_backend):
    collection_path = Path(collection_path).resolve()
    root = collection_path.parent
    metadata = json.loads(collection_path.read_text())
    config = metadata['config']
    if identity(metadata['provenance']) != metadata['provenance_id']:
        raise ValueError('Collection provenance digest mismatch')
    jobs = jobs_for(config)
    if metadata['jobs'] != jobs:
        raise ValueError('Collection jobs do not match its declared design')
    expected_keys = {identity(j) for j in jobs}
    if set(metadata['rows']) - expected_keys:
        raise ValueError('Collection contains undeclared jobs')
    status, snapshots, targets, incumbents = [], [], [], []
    successful = defaultdict(int)
    initial_by_seed = {}
    for job in jobs:
        row = metadata['rows'].get(identity(job))
        common = {k: job[k] for k in ['initial_design_seed', 'method', 'handler', 'search_replicate']}
        status.append(dict(**common, status=row['status'] if row else 'not_started',
                           run_id=row['run_id'] if row else None))
        if row is None:
            continue
        folder = verify_run(root, row, job, metadata)
        if row['status'] != 'complete':
            continue
        trace = read_csv(folder / 'evaluations.csv')
        if len(trace) != row['evaluations'] or len(trace) != job['budget']:
            raise ValueError('Completed trajectory has the wrong call count')
        losses, points = [], []
        for index, record in enumerate(trace, 1):
            loss = float(record['loss'])
            point = json.loads(record['u'])
            if (record['status'] != 'ok' or int(record['evaluation']) != index
                    or not math.isfinite(loss) or loss <= 0
                    or len(point) != 5 or not all(math.isfinite(x) for x in point)
                    or math.sqrt(sum(x*x for x in point)) > 1 + 1e-9):
                raise ValueError('Invalid completed trace or infeasible physical point')
            losses.append(loss)
            points.append(point)
        seed = job['initial_design_seed']
        initial = points[:job['n_init']]
        if seed in initial_by_seed and initial_by_seed[seed] != initial:
            raise ValueError('Initial physical designs do not agree across arms')
        if seed not in initial_by_seed and config.get('executor') == 'v4-aligned-v1':
            from bo_vs_es.v5_aligned import initial_points
            if initial != initial_points(config, seed).tolist():
                raise ValueError('Initial physical design does not match V4 construction')
        initial_by_seed[seed] = initial
        best = math.inf
        for calls, (record, loss) in enumerate(zip(trace, losses), 1):
            best = min(best, loss)
            incumbents.append(dict(**common, run_id=row['run_id'], calls=calls,
                loss=loss, best_loss=best, best_rmse_v=math.sqrt(best),
                search_s=float(record['elapsed_s']), objective_s=float(record['objective_s'])))
        if not math.isclose(min(losses), row['best_loss'], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Recorded best loss differs from the trace minimum')
        if [cp['calls'] for cp in row['checkpoints']] != config['checkpoints']:
            raise ValueError('Missing or duplicated checkpoints')
        if row['assessment_calls'] != 2 * len(config['checkpoints']):
            raise ValueError('Incorrect completed-run assessment count')
        successful[seed] += 1
        for cp in row['checkpoints']:
            calls = cp['calls']
            if not math.isclose(cp['loss'], min(losses[:calls]), rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError('Checkpoint loss differs from its trace minimum')
            index = cp['evaluation'] - 1
            if not 0 <= index < calls or not math.isclose(losses[index], cp['loss'], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError('Checkpoint recommendation does not attain its claimed loss')
            if math.dist(cp['u'], points[index]) > 1e-12:
                raise ValueError('Checkpoint recommendation differs from its trace point')
            margins = [1 - math.sqrt(sum(x*x for x in p)) for p in points[:calls]]
            snapshots.append(dict(**common, run_id=row['run_id'], calls=calls, loss=cp['loss'],
                calibration_rmse_v=math.sqrt(cp['loss']), search_s=cp['search_s'],
                objective_s=cp['objective_s'], decision_and_logging_s=cp['search_s'] - cp['objective_s'],
                recommendation_margin=margins[index],
                boundary_fraction=sum(m <= 1e-8 for m in margins) / calls,
                within_001_fraction=sum(m <= .01 for m in margins) / calls,
                within_005_fraction=sum(m <= .05 for m in margins) / calls,
                rejected_fraction=cp['rejected'] / cp['proposals'],
                repaired_fraction=cp['repaired'] / cp['proposals'],
                fuds80_rmse_v=cp['fuds80_rmse_v'], fuds50_rmse_v=cp['fuds50_rmse_v']))
            for target in config['targets_rmse_v']:
                hit = next((i for i, loss in enumerate(losses[:calls]) if loss <= target**2), None)
                targets.append(dict(**common, run_id=row['run_id'], calls=calls, target_rmse_v=target,
                    hit=hit is not None, calls_to_target=hit + 1 if hit is not None else None,
                    seconds_to_target=float(trace[hit]['elapsed_s']) if hit is not None else None,
                    outcome='reached' if hit is not None else 'not reached within completed budget'))
    planned = defaultdict(int)
    for job in jobs:
        planned[job['initial_design_seed']] += 1
    coverage = [dict(initial_design_seed=s, planned=n, complete=successful[s],
                     complete_group=successful[s] == n) for s, n in planned.items()]
    complete_seeds = [r['initial_design_seed'] for r in coverage if r['complete_group']]
    for row in snapshots + targets + incumbents:
        row['complete_group'] = row['initial_design_seed'] in complete_seeds
    comparisons = contrasts(snapshots, complete_seeds, config)
    grouped = defaultdict(list)
    for row in targets:
        if row['complete_group']:
            grouped[row['method'], row['handler'], row['calls'], row['target_rmse_v']].append(row)
    target_summary = []
    for key, group in grouped.items():
        hits = [r for r in group if r['hit']]
        target_summary.append(dict(zip(['method', 'handler', 'calls', 'target_rmse_v'], key),
            complete_designs=len(complete_seeds), runs=len(group), hits=len(hits),
            hit_rate=len(hits) / len(group),
            median_calls_if_reached=statistics.median(r['calls_to_target'] for r in hits) if hits else None,
            median_seconds_if_reached=statistics.median(r['seconds_to_target'] for r in hits) if hits else None))
    # Include failed/retried attempts in resource accounting, even when the latest
    # successful run has replaced them in the collection's job index.
    attempts = []
    for folder in sorted((root / 'runs').iterdir()) if (root / 'runs').exists() else []:
        if not folder.is_dir():
            continue
        summary_path = folder / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        trace_path = folder / 'evaluations.csv'
        attempts.append(dict(run_id=folder.name, status=summary.get('status', 'unclosed'),
            search_calls=len(read_csv(trace_path)) if trace_path.exists() else 0,
            recorded_assessment_calls=summary.get('assessment_calls'), attempt_s=summary.get('attempt_s')))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    output = root / 'analysis' / stamp
    output.mkdir(parents=True, exist_ok=False)
    initial_designs = [dict(initial_design_seed=seed, points=len(points),
        initial_u_sha256=identity(points),
        construction='V4 SeedSequence kind 0' if config.get('executor') == 'v4-aligned-v1' else 'original V5 default_rng(design)',
        verified_against_v4=config.get('executor') == 'v4-aligned-v1')
        for seed, points in sorted(initial_by_seed.items())]
    tables = dict(run_status=status, coverage=coverage, checkpoints=snapshots,
                  contrasts=comparisons, targets=targets, target_summary=target_summary,
                  attempts=attempts, incumbents=incumbents, initial_designs=initial_designs)
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
        with (output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    plot_record = None
    if plots:
        from bo_vs_es.v5_plots import make_plots
        plot_record = make_plots(output, metadata, complete_seeds, comparisons, plot_backend)
        plot_record['source_sha256'] = digest(ROOT / 'src/bo_vs_es/v5_plots.py')
    invocations = [json.loads(p.read_text()) for p in (root / 'invocations').glob('*.json')]
    setup_calls = sum(p.get('objective_setup', {}).get('numpy_calls', 0) +
                      p.get('objective_setup', {}).get('jax_calls', 0) for p in invocations)
    audit = dict(collection_sha256=digest(collection_path), analysis_sha256=digest(Path(__file__)),
        contrast_source_sha256=digest(ROOT / 'src/bo_vs_es/v5.py'),
        executor=config.get('executor', 'original-v5-v3'),
        alignment_source_sha256=(digest(ROOT / 'src/bo_vs_es/v5_aligned.py')
                                 if config.get('executor') == 'v4-aligned-v1' else None),
        planned_runs=len(jobs), completed_runs=sum(successful.values()),
        complete_design_groups=len(complete_seeds), excluded_design_groups=len(coverage) - len(complete_seeds),
        all_recorded_search_calls=sum(r['search_calls'] for r in attempts),
        objective_setup_calls=setup_calls,
        total_recorded_invocation_s=sum(p['elapsed_s'] for p in invocations),
        plots=plot_record,
        recorded_assessment_calls=sum(r['recorded_assessment_calls'] or 0 for r in attempts),
        attempts_without_assessment_accounting=sum(r['recorded_assessment_calls'] is None for r in attempts),
        status='complete' if len(complete_seeds) == len(coverage) else 'partial',
        outputs={p.name: digest(p) for p in output.iterdir()},
        interpretation='Primary contrast is projection/resampling at 100 calls; other contrasts are secondary. '
                       'Intervals are pointwise and resample initial designs. Aligned V5 reuses V4 designs '
                       'and is an exploratory extension, not independent confirmation. Excluded groups may bias comparisons; '
                       'conditional target times must accompany hit rates and coverage.')
    (output / 'validation.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    return output
