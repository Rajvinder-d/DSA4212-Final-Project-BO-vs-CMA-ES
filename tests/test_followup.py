"""Focused checks for the new study; synthetic test fixtures are never study runs."""
import copy
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bo_vs_es import ROOT
from bo_vs_es import followup as f
from bo_vs_es.diagnostics.replay import scalar_model
from bo_vs_es.followup_analysis import group_intervals, target_record, read_collection, paired_budget_changes
from bo_vs_es.objectives.models import Domain, Quadratic, rotation


@pytest.fixture
def collection():
    with tempfile.TemporaryDirectory(prefix='test-followup-', dir=ROOT / 'artifacts') as path:
        yield Path(path)


def test_frozen_grid_and_stream_separation():
    config = f.load_config()
    jobs = f.jobs_for(config)
    assert len(jobs) == 885
    assert sum(j['budget'] for j in jobs) == 196500
    frame = pd.DataFrame(jobs)
    assert frame.groupby(['block', 'initial_design_seed']).size().unique().tolist() == [30, 36, 45]
    assert set(frame.rotation_seed.dropna()) == {202, 303}
    assert frame.groupby(['initial_design_seed', 'search_replicate']).search_seed.nunique().eq(1).all()
    assert len({f.search_seed(config, 10, r) for r in range(3)}) == 3
    assert frame[frame.method.eq('bo_iso_pool')].block.eq('battery_ball').all()
    # A search-stream change does not change the actual initial design.
    a = Domain('ball', np.eye(5)).sample(np.random.default_rng(10), 20)
    b = Domain('ball', rotation(202)).sample(np.random.default_rng(10), 20)
    np.testing.assert_allclose(a, b @ rotation(202).T, atol=1e-14)


def test_reserved_cases_reject_development_overlap(tmp_path):
    config = f.load_config()
    config['confirmation_seeds'][0] = 0
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='reserved'):
        f.load_config(path)


def test_pool_geometry_scoring_and_index_ties():
    settings = dict(interior=64, surface=64, chunk_size=32)
    pool = f.candidate_pool(np.random.default_rng(8), settings)
    np.testing.assert_array_equal(pool, f.candidate_pool(np.random.default_rng(8), settings))
    assert (np.linalg.norm(pool[:64], axis=1) < 1).all()
    np.testing.assert_allclose(np.linalg.norm(pool[64:], axis=1), 1, atol=1e-15)
    points = Domain('ball', np.eye(5)).sample(np.random.default_rng(3), 20)
    losses = [Quadratic(100)(p) for p in points]
    q = rotation(202)
    left = f.acquisition_scores(scalar_model(points, losses), pool, 32)
    right = f.acquisition_scores(scalar_model(points @ q, losses), pool @ q, 32)
    np.testing.assert_allclose(left, right, atol=1e-9, rtol=1e-9)
    assert np.argmax(left) == np.argmax(right)
    duplicate = np.repeat(pool[np.argmax(left)][None, :], 3, axis=0)
    assert np.argmax(f.acquisition_scores(scalar_model(points, losses), duplicate)) == 0


def test_prefixes_counts_assessments_and_verified_reuse(collection, monkeypatch):
    config = f.load_config()
    job = copy.deepcopy(f.jobs_for(config)[0])
    job.update(method='random', budget=300, checkpoints=[100, 300])
    protocol = json.loads((ROOT / config['v1_protocol']).read_text())
    training_calls, assessments = [], []
    def objective(u):
        training_calls.append(u.copy())
        return Quadratic(100)(u)
    def battery(label):
        def evaluate(u):
            assert len(training_calls) == 300  # Held-out checks only after all search.
            assessments.append((label, u.copy()))
            return .01
        return evaluate
    monkeypatch.setattr(f, 'Battery', battery)
    row, cached = f.execute_job(job, config, protocol, collection, {}, objective=objective)
    assert not cached and row['status'] == 'complete'
    assert len(training_calls) == row['evaluations'] == 300
    assert len(assessments) == row['assessment_calls'] == 4
    assert [p['calls'] for p in row['checkpoints']] == [100, 300]
    trace = pd.read_csv(ROOT / row['run_path'] / 'evaluations.csv')
    for checkpoint in row['checkpoints']:
        assert checkpoint['loss'] == pytest.approx(trace.loss.iloc[:checkpoint['calls']].min())
    again, cached = f.execute_job(job, config, protocol, collection, {}, objective=objective)
    assert cached and again == row and len(training_calls) == 300
    (ROOT / row['run_path'] / 'recommendation.json').write_text('{}')
    retried, cached = f.execute_job(job, config, protocol, collection, {}, objective=Quadratic(100))
    assert not cached and '-retry1' in retried['run_id']


@pytest.mark.parametrize('handler', ['resample', 'project'])
def test_cma_boundary_attempts_and_generation_snapshots(collection, handler):
    config = f.load_config()
    job = copy.deepcopy(f.jobs_for(config)[0])
    job.update(method='cma_full', objective='quadratic', condition=100, domain='box',
               rotation_seed=202, handler=handler, budget=36, checkpoints=[28, 36])
    protocol = json.loads((ROOT / config['v1_protocol']).read_text())
    row, _ = f.execute_job(job, config, protocol, collection, {})
    assert row['status'] == 'complete'
    assert [p['calls'] for p in row['checkpoints']] == [28, 36]
    trace = pd.read_csv(ROOT / row['run_path'] / 'evaluations.csv')
    u = np.array([json.loads(x) for x in trace.u])
    assert np.max(np.abs(u)) <= 1 + 1e-9
    assert row['proposals'] - row['rejected'] == 16


def test_failed_simulator_and_deadline_preserve_attempts(collection):
    config = f.load_config()
    job = f.jobs_for(config)[0]
    protocol = json.loads((ROOT / config['v1_protocol']).read_text())
    row, _ = f.execute_job(job, config, protocol, collection, {}, objective=lambda x: float('nan'))
    assert row['status'] == 'failed' and row['evaluations'] == 1
    assert (ROOT / row['run_path'] / 'error.txt').exists()
    calls = []
    def slow(u):
        calls.append(u)
        time.sleep(.02)
        return 1.
    row, _ = f.execute_job(job, config, protocol, collection, {},
                            deadline=time.monotonic() + .1, objective=slow)
    assert row['status'] == 'budget_limited' and row['evaluations'] == len(calls) < 20


def test_group_bootstrap_and_censored_targets():
    frame = pd.DataFrame({'method': ['x'] * 6, 'initial_design_seed': [10]*3 + [11]*3,
                          'effect': [0., 2., 4., 8., 10., 12.]})
    summary = group_intervals(frame, ['method'], 'effect', f.load_config()).iloc[0]
    assert summary.design_groups == 2 and summary.estimate == 6.
    repeated = pd.concat([frame, frame])
    pd.testing.assert_frame_equal(group_intervals(frame, ['method'], 'effect', f.load_config()),
                                  group_intervals(repeated, ['method'], 'effect', f.load_config()))
    trace = pd.DataFrame(dict(evaluation=[1,2,3], elapsed_s=[.1,.2,.3], loss=[.1,.01,.001]))
    missed = target_record(trace, 3, .01)
    assert not missed['hit'] and missed['seconds_to_target'] is None
    assert target_record(trace, 3, .1)['calls_to_target'] == 2


def test_budget_contrast_pairs_trajectories_before_design_bootstrap():
    # Different baseline levels and unequal replicate counts must not weight designs unequally.
    frame = pd.DataFrame([
        (10, 0, 100, 10.), (10, 0, 300, 9.),
        (10, 1, 100, 20.), (10, 1, 300, 17.),
        (11, 0, 100, 100.), (11, 0, 300, 94.),
        (12, 0, 100, 1000.),  # Missing extension: cannot enter the paired contrast.
    ], columns=['initial_design_seed', 'search_replicate', 'calls', 'interaction'])
    changed = paired_budget_changes(frame.sample(frac=1, random_state=4),
        ['initial_design_seed', 'search_replicate'], 'interaction')
    assert changed.change_300_minus_100.tolist() == [-1., -3., -6.]
    changed['method'] = 'cma'
    interval = group_intervals(changed, ['method'], 'change_300_minus_100', f.load_config()).iloc[0]
    assert interval.design_groups == 2
    assert interval.estimate == -4.  # Mean of design means (-2 and -6), not -10/3.
    assert interval.low == -6. and interval.high == -2.
    with pytest.raises(ValueError, match='duplicate'):
        paired_budget_changes(pd.concat([frame, frame.iloc[:1]]),
            ['initial_design_seed', 'search_replicate'], 'interaction')


def test_artifact_validation_exposes_incomplete_groups(collection):
    config = f.load_config()
    config['id'] = collection.name
    job = copy.deepcopy(f.jobs_for(config)[0])
    job.update(method='random', objective='quadratic', condition=100, block='quadratic_control')
    protocol = json.loads((ROOT / config['v1_protocol']).read_text())
    folder = collection / 'confirmation'
    folder.mkdir()
    row, _ = f.execute_job(job, config, protocol, folder, {})
    missing = dict(job, rotation_seed=202)
    f.dump(folder / 'collection.json', dict(jobs=[job, missing], rows=[row]))
    _, _, snapshots, _, _ = read_collection(config)
    assert len(snapshots) == 1 and not snapshots.complete_group.any()
    assert snapshots.loss.iloc[0] == row['best_loss']
