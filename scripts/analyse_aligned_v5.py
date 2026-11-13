"""Audit curated V5 exports and derive report quantities; no objective evaluation."""
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/v5-aligned'
OUT = ROOT / 'artifacts/report-checks'


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def bootstrap(values):
    rng = random.Random(5129)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    def quantile(p):
        index = (len(samples) - 1) * p
        lo = int(index)
        return samples[lo] + (index - lo) * (samples[min(lo + 1, len(samples) - 1)] - samples[lo])
    return dict(estimate=statistics.mean(values), low=quantile(.025), high=quantile(.975))


def main(output_dir=OUT):
    audit = json.loads((DATA / 'validation.json').read_text())
    for name, expected in audit['outputs'].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected, name
    coverage = read(DATA / 'coverage.csv')
    seeds = [r['initial_design_seed'] for r in coverage if r['complete_group'] == 'True']
    statuses = read(DATA / 'run_status.csv')
    failures = [r for r in statuses if r['status'] != 'complete']
    assert len(statuses) == audit['planned_runs'] == 280
    assert len(statuses) - len(failures) == audit['completed_runs'] == 279
    assert seeds == [str(n) for n in range(20, 40) if n != 21]
    assert len(seeds) == audit['complete_design_groups'] == 19
    attempts = read(DATA / 'attempts.csv')
    assert sum(int(r['search_calls']) for r in attempts) == audit['all_recorded_search_calls'] == 83866
    assert sum(int(r['recorded_assessment_calls']) for r in attempts) == 1116
    points = read(DATA / 'initial_designs.csv')
    assert len(points) == 20 and all(r['verified_against_v4'] == 'True' and r['points'] == '20' for r in points)
    all_checkpoints = read(DATA / 'checkpoints.csv')
    checkpoints = [r for r in all_checkpoints if r['initial_design_seed'] in seeds]
    assert len(all_checkpoints) == 558 and len(checkpoints) == 532
    traces = defaultdict(list)
    for row in read(DATA / 'incumbents.csv'):
        traces[row['run_id']].append(row)
    assert len(traces) == 279
    for run, rows in traces.items():
        assert [int(r['calls']) for r in rows] == list(range(1, 301)), run
        best = math.inf
        for row in rows:
            loss = float(row['loss'])
            assert math.isfinite(loss) and loss > 0
            best = min(best, loss)
            assert math.isclose(best, float(row['best_loss']), rel_tol=1e-12, abs_tol=1e-14)
        for row in (r for r in all_checkpoints if r['run_id'] == run):
            assert math.isclose(float(rows[int(row['calls']) - 1]['best_loss']),
                                float(row['loss']), rel_tol=1e-12, abs_tol=1e-14)
    def logs(rows, seed, calls, method, handler):
        group = [r for r in rows if (r['initial_design_seed'], int(r['calls']), r['method'], r['handler'])
                 == (seed, calls, method, handler)]
        assert sorted(int(r['search_replicate']) for r in group) == (list(range(5)) if method == 'cma_full' else [0])
        return statistics.mean(math.log10(float(r['loss'])) for r in group)
    comparisons = []
    for contrast in read(DATA / 'contrasts.csv'):
        left, right = contrast['contrast'].split(' / ')
        left_handler = 'project' if left == 'CMA projection' else left.split()[1]
        right_method, right_handler = ('cma_full', 'resample') if right == 'CMA resampling' else (right, 'none')
        calls = int(contrast['calls'])
        effects = [logs(checkpoints, s, calls, 'cma_full', left_handler) -
                   logs(checkpoints, s, calls, right_method, right_handler) for s in seeds]
        estimated = bootstrap(effects)
        assert all(math.isclose(estimated[k], float(contrast[k]), abs_tol=1e-12) for k in estimated)
        comparisons.append(dict(contrast=contrast['contrast'], calls=calls, role=contrast['role'],
            designs=len(seeds), **{k: 10**v for k, v in estimated.items()},
            designs_favouring_numerator=sum(v < 0 for v in effects)))
    sensitivity = []
    for calls in [100, 300]:
        effects = [logs(all_checkpoints, str(s), calls, 'cma_full', 'project') -
                   logs(all_checkpoints, str(s), calls, 'cma_full', 'resample') for s in range(20, 40)]
        sensitivity.append(dict(calls=calls, designs=20, role='post hoc handler-only completeness sensitivity',
                                **{k: 10**v for k, v in bootstrap(effects).items()}))
    summaries = []
    for method, handler in [('bo_iso', 'none'), ('bo_ard', 'none'), ('bo_iso_pool', 'none'),
                            ('cma_full', 'resample'), ('cma_full', 'project'), ('random', 'none')]:
        for calls in [100, 300]:
            rows = [r for r in checkpoints if (r['method'], r['handler'], int(r['calls'])) == (method, handler, calls)]
            fields = ['calibration_rmse_v', 'search_s', 'objective_s', 'boundary_fraction',
                      'recommendation_margin', 'fuds80_rmse_v', 'fuds50_rmse_v']
            summaries.append(dict(method=method, handler=handler, calls=calls, runs=len(rows),
                                  **{k: statistics.median(float(r[k]) for r in rows) for k in fields}))
    # Recompute target summaries from saved incumbents, including failures to hit.
    for row in read(DATA / 'target_summary.csv'):
        candidates = [r for r in checkpoints if (r['method'], r['handler'], r['calls']) ==
                      (row['method'], row['handler'], row['calls'])]
        hits = []
        for candidate in candidates:
            first = next((r for r in traces[candidate['run_id']][:int(row['calls'])]
                          if float(r['best_loss']) <= float(row['target_rmse_v'])**2), None)
            if first:
                hits.append(first)
        assert len(candidates) == int(row['runs']) and len(hits) == int(row['hits'])
        if hits:
            assert math.isclose(statistics.median(float(r['search_s']) for r in hits),
                                float(row['median_seconds_if_reached']), rel_tol=1e-10, abs_tol=1e-10)
            assert statistics.median(int(r['calls']) for r in hits) == float(row['median_calls_if_reached'])
    # Compare shared baseline endpoints across machines, not times or treatment effects.
    v4 = read(ROOT / 'results/followup_v4/checkpoints.csv')
    key = lambda r: (r['method'], r['initial_design_seed'], r['search_replicate'], r['calls'])
    lookup = {key(r): r for r in v4 if r['orientation'] == '-1'
              and r['block'] in {'battery_bo', 'cma_ball'} and r['handler'] in {'none', 'resample'}}
    pairs, grouped = [], defaultdict(list)
    for row in checkpoints:
        if row['handler'] == 'project' or key(row) not in lookup:
            continue
        previous = lookup[key(row)]
        ratio = float(row['loss']) / float(previous['loss'])
        close = math.isclose(float(row['loss']), float(previous['loss']), rel_tol=1e-10, abs_tol=1e-14)
        pair = dict(method=row['method'], calls=int(row['calls']), design=int(row['initial_design_seed']),
                    replicate=int(row['search_replicate']), v4_loss=float(previous['loss']),
                    aligned_v5_loss=float(row['loss']), loss_ratio=ratio, close=close)
        pairs.append(pair)
        grouped[row['method'], row['calls']].append(pair)
    baseline = [dict(method=k[0], calls=int(k[1]), pairs=len(rows), close=sum(r['close'] for r in rows),
                     median_absolute_relative_difference=statistics.median(abs(r['loss_ratio'] - 1) for r in rows),
                     maximum_absolute_relative_difference=max(abs(r['loss_ratio'] - 1) for r in rows))
                for k, rows in grouped.items()]
    output = Path(output_dir) / 'report_checks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(hashes_verified=len(audit['outputs']),
        coverage=dict(planned=280, completed=279, complete_designs=19, excluded_design=21),
        failures=failures, recorded_search_calls=83866, recorded_assessment_calls=1116,
        contrasts=comparisons, handler_sensitivity=sensitivity, medians=summaries,
        baseline_endpoint_comparison=baseline,
        scope='Saved exports only; no raw parameter histories or execution manifest available for local audit. '
              'Initial-design construction is reported verified by the execution-side exporter, not independently '
              'verified against historical V4 raw point arrays. Cross-machine endpoint agreement is assessed '
              'at relative tolerance 1e-10, absolute tolerance 1e-14; it does not establish path agreement.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    with (output.parent / 'v4_aligned_v5_endpoint_pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    print(json.dumps(dict(verified=audit['status'], complete_designs=len(seeds),
                         sensitivity=sensitivity, checks=str(output)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Write derived checks here; inputs always use the committed aligned exports')
    main(parser.parse_args().output_dir)
