"""Prepared checks, not executed during v5 authoring. Fixtures are not study data."""
import copy
import json
import math
import tempfile

import pytest

from bo_vs_es.v5 import ROOT, contrasts, jobs_for, load_config, planned_counts


def test_design_keeps_the_problem_and_budget_fixed():
    config = load_config()
    jobs = jobs_for(config)
    assert planned_counts(config)['search_calls'] == 30000
    assert planned_counts(config)['assessment_calls'] == 400
    assert len(jobs) == 100
    for seed in config['initial_design_seeds']:
        group = [j for j in jobs if j['initial_design_seed'] == seed]
        assert len(group) == 10
        assert all(j['budget'] == 300 and j['domain'] == 'ball' and j['rotation_seed'] is None for j in group)
        for replicate in [0, 1, 2]:
            paired = [j for j in group if j['method'] == 'cma_full' and j['search_replicate'] == replicate]
            assert len(paired) == 2
            assert len({j['search_seed'] for j in paired}) == 1
    assert jobs == jobs_for(config)


def test_seen_seeds_are_rejected(tmp_path):
    config = load_config()
    config['initial_design_seeds'][0] = 10
    path = tmp_path / 'overlap.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='overlap'):
        load_config(path)


def test_contrasts_use_designs_and_have_the_declared_sign():
    config = load_config()
    config['bootstrap_resamples'] = 100
    snapshots = []
    for job in jobs_for(config):
        for calls in config['checkpoints']:
            loss = 1.
            if job['method'] == 'cma_full':
                loss = 2. if job['handler'] == 'project' else 4.
            snapshots.append(dict(initial_design_seed=job['initial_design_seed'], method=job['method'],
                handler=job['handler'], search_replicate=job['search_replicate'], calls=calls, loss=loss))
    rows = contrasts(snapshots, config['initial_design_seeds'], config)
    primary = [r for r in rows if r['role'] == 'primary']
    assert len(primary) == 1
    assert primary[0]['design_groups'] == 10  # Not 30 CMA trajectories.
    assert primary[0]['estimate'] == pytest.approx(math.log10(.5))
    assert primary[0]['low'] == pytest.approx(primary[0]['high'])
    with pytest.raises(ValueError, match='Incomplete'):
        contrasts(snapshots[1:], config['initial_design_seeds'], config)


@pytest.mark.parametrize('handler', ['resample', 'project'])
def test_ball_handler_reuses_executor_and_analysis_preserves_coverage(handler, monkeypatch):
    import numpy as np
    from bo_vs_es import followup
    from bo_vs_es.v5 import identity
    from bo_vs_es.v5_analysis import export

    config = load_config()
    job = copy.deepcopy(next(j for j in jobs_for(config) if j['method'] == 'cma_full' and j['handler'] == handler))
    protocol = json.loads((ROOT / config['protocol']).read_text())
    calls, assessments = [], []
    def objective(u):
        assert np.linalg.norm(u) <= 1 + 1e-9
        calls.append(u.copy())
        return float(.01 + np.sum((u - np.array([1., 0., 0., 0., 0.])) ** 2))
    def battery(label):
        def assess(u):
            assert len(calls) == 300
            assessments.append(label)
            return .01
        return assess
    monkeypatch.setattr(followup, 'Battery', battery)
    parent = ROOT / 'artifacts'
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='v5-test-', dir=parent) as tmp:
        from pathlib import Path
        root = Path(tmp)
        runs = root / 'runs'
        runs.mkdir()
        row, cached = followup.execute_job(job, config, protocol, runs, {}, objective=objective)
        assert row['status'] == 'complete' and not cached
        assert row['evaluations'] == len(calls) == 300
        assert len(assessments) == row['assessment_calls'] == 4
        assert [cp['calls'] for cp in row['checkpoints']] == [100, 300]
        if handler == 'project':
            assert row['repaired'] > 0
            assert row['rejected'] == 0
        else:
            assert row['repaired'] == 0
            assert row['rejected'] > 0
        metadata = dict(config=config, protocol=protocol, provenance={}, provenance_id=identity({}),
                        jobs=jobs_for(config), rows={identity(job): row})
        path = root / 'collection.json'
        path.write_text(json.dumps(metadata))
        output = export(path)
        audit = json.loads((output / 'validation.json').read_text())
        assert audit['completed_runs'] == 1
        assert audit['complete_design_groups'] == 0
        assert audit['status'] == 'partial'
        # Corrupt evidence must be rejected, never silently analysed.
        (ROOT / row['run_path'] / 'recommendation.json').write_text('{}')
        with pytest.raises(ValueError, match='Modified'):
            export(path)


def test_new_session_limit_reuses_completed_work(tmp_path, monkeypatch):
    from bo_vs_es import v5, followup
    from bo_vs_es.objectives import data

    config = load_config()
    protocol = json.loads((ROOT / config['protocol']).read_text())
    path = tmp_path / config['protocol']
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(protocol))
    jobs = jobs_for(config)[:2]
    completed = {}
    new_work = []
    def execute(job, *args, **kwargs):
        key = v5.identity(job)
        if key in completed:
            return completed[key], True
        new_work.append(key)
        completed[key] = dict(status='complete', run_id=key)
        return completed[key], False
    monkeypatch.setattr(v5, 'ROOT', tmp_path)
    monkeypatch.setattr(v5, 'jobs_for', lambda config: jobs)
    monkeypatch.setattr(followup, 'provenance', lambda config, environment: {})
    monkeypatch.setattr(followup, 'execute_job', execute)
    monkeypatch.setattr(data, 'sha256', lambda path: 'fixture-source-hash')
    # Simulate the first session limit being reached after its first whole run.
    monkeypatch.setattr(v5, 'session_limit_reached',
                        lambda start, now, limit: limit == 1 and bool(new_work))
    assert v5.run(config, {}, 1, 'test') == 0
    collection = tmp_path / 'artifacts' / config['id'] / 'collection.json'
    assert json.loads(collection.read_text())['status'] == 'paused'
    assert len(new_work) == 1
    assert v5.run(config, {}, None, 'test') == 0
    assert len(new_work) == 2  # First trajectory was reused, not run again.
    assert json.loads(collection.read_text())['status'] == 'complete'
