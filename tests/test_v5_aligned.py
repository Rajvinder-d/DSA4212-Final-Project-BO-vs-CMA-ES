"""Alignment checks use synthetic fixtures only; no CALCE evaluation is launched."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bo_vs_es import ROOT
from bo_vs_es import v5, v5_aligned


def frozen_functions(*names):
    """Read V4's pure planning functions without importing its numerical stack."""
    path = ROOT / 'src/bo_vs_es/followup_v4.py'
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'np': np, 'BO_METHODS': ['bo_iso', 'bo_ard', 'bo_iso_pool', 'bo_ard_pool']}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def synthetic_snapshots(config):
    rows = []
    for job in v5.jobs_for(config):
        for calls in config['checkpoints']:
            loss = 1.
            if job['method'] == 'cma_full':
                loss = (job['search_replicate'] + 1.) * (2 if job['handler'] == 'resample' else 1)
            rows.append(dict(initial_design_seed=job['initial_design_seed'],
                method=job['method'], handler=job['handler'],
                search_replicate=job['search_replicate'], calls=calls, loss=loss))
    return rows


class AlignedDesignTests(unittest.TestCase):
    def setUp(self):
        self.config = v5.load_config(v5_aligned.DEFAULT_CONFIG)

    def test_plan_and_paired_handlers(self):
        jobs = v5.jobs_for(self.config)
        counts = v5.planned_counts(self.config)
        self.assertEqual((counts['trajectories'], counts['search_calls'],
                          counts['assessment_calls'], counts['initial_designs']),
                         (280, 84000, 1120, 20))
        self.assertEqual(len({v5.identity(j) for j in jobs}), 280)
        for seed in range(20, 40):
            group = [j for j in jobs if j['initial_design_seed'] == seed]
            self.assertEqual(len(group), 14)
            for replicate in range(5):
                pair = [j for j in group if j['method'] == 'cma_full'
                        and j['search_replicate'] == replicate]
                self.assertEqual({j['handler'] for j in pair}, {'resample', 'project'})
                self.assertEqual(len({j['search_seed'] for j in pair}), 1)
            for job in group:
                self.assertIsNone(job['rotation_seed'])
                self.assertEqual(job['domain'], 'ball')
                self.assertEqual((job['budget'] - job['n_init']) % 8, 0)
                self.assertTrue(all((n - job['n_init']) % 8 == 0 for n in job['checkpoints']))
        self.assertEqual(jobs, v5.jobs_for(self.config))

    def test_baseline_jobs_match_frozen_v4_except_declared_extensions(self):
        functions = frozen_functions('stream', 'search_seed', 'jobs_for')
        v4_config = json.loads((ROOT / 'configs/experiments/followup_v4.json').read_text())
        old = functions['jobs_for'](v4_config)
        key = lambda j: (j['method'], j['initial_design_seed'], j['search_replicate'])
        lookup = {key(j): j for j in old if j['block'] in {'battery_bo', 'cma_ball'}
                  and j['rotation_seed'] is None and j['method'] != 'cma_diag'}
        compared = 0
        for job in v5.jobs_for(self.config):
            if job['handler'] == 'project':
                continue
            original = lookup[key(job)]
            for field in job:
                if field not in {'block', 'stage', 'budget', 'checkpoints'}:
                    self.assertEqual(job[field], original[field], field)
            self.assertEqual(original['budget'], 300 if job['method'] == 'cma_full' else 100)
            compared += 1
        self.assertEqual(compared, 180)

    def test_search_seeds_match_v4_and_differ_from_original_v5(self):
        functions = frozen_functions('stream', 'search_seed')
        for design in range(20, 40):
            for replicate in range(5):
                self.assertEqual(v5_aligned.matching_search_seed(self.config, design, replicate),
                                 functions['search_seed'](self.config, design, replicate))
        self.assertEqual(v5_aligned.matching_search_seed(self.config, 30, 0), 507603206)
        original = next(j for j in v5.jobs_for(v5.load_config())
                        if j['initial_design_seed'] == 30 and j['search_replicate'] == 0)
        self.assertEqual(original['search_seed'], 81147440)

    def test_scientific_sources_match_executed_v4_hashes(self):
        ref = json.loads(v5_aligned.REFERENCE.read_text())
        for name, expected in ref['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)
        protocol = ROOT / self.config['protocol']
        self.assertEqual(hashlib.sha256(protocol.read_bytes()).hexdigest(), ref['protocol_sha256'])
        self.assertEqual(hashlib.sha256((ROOT / 'requirements-lock.txt').read_bytes()).hexdigest(),
                         ref['dependency_lock_sha256'])

    def test_changed_design_is_rejected(self):
        for key, value in [('namespace', 5107), ('initial_design_seeds', list(range(30, 40))),
                           ('cma_search_replicates', [0, 1, 2]), ('budget', 100),
                           ('lengthscale_upper', 20.), ('id', 'boundary-v5-windows-jax')]:
            with self.subTest(key=key):
                config = dict(self.config, **{key: value})
                with self.assertRaises(ValueError):
                    v5_aligned.validate_config(config)

    def test_original_design_is_preserved(self):
        self.assertEqual(v5.planned_counts(v5.load_config())['trajectories'], 100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            path.write_text(json.dumps(dict(self.config, executor='misspelled')))
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                v5.load_config(path)

    def test_five_replicates_are_collapsed_into_twenty_designs(self):
        snapshots = synthetic_snapshots(self.config)
        analysis_config = dict(self.config, bootstrap_resamples=50)
        rows = v5.contrasts(snapshots, self.config['initial_design_seeds'], analysis_config)
        self.assertEqual(len(rows), 14)
        primary = [r for r in rows if r['role'] == 'primary']
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]['design_groups'], 20)
        for field in ['estimate', 'low', 'high']:
            self.assertAlmostEqual(primary[0][field], math.log10(.5))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            v5.contrasts(snapshots[1:], self.config['initial_design_seeds'], analysis_config)
        # Counts alone would miss a duplicated replicate replacing the fifth one.
        duplicate = copy.deepcopy(snapshots)
        next(r for r in duplicate if r['search_replicate'] == 4)['search_replicate'] = 3
        with self.assertRaisesRegex(ValueError, 'replicates'):
            v5.contrasts(duplicate, self.config['initial_design_seeds'], analysis_config)

    def test_default_cli_is_planning_only(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_v5_aligned.py')],
                                capture_output=True, text=True, check=True, cwd=ROOT)
        plan = json.loads(result.stdout)
        self.assertFalse(plan['executed'])
        self.assertEqual(plan['backend'], 'numpy')
        self.assertEqual(plan['plan']['trajectories'], 280)
        self.assertEqual(plan['config']['id'], 'boundary-comparison-v5-v4-aligned')
        self.assertIsNone(plan['session_limit_s'])

    def test_input_and_version_guard_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ['source.py', 'configs/protocols/rotation_battery_v1.json',
                     'requirements-lock.txt', 'data/processed/input.csv']
            digest = hashlib.sha256(b'fixture').hexdigest()
            for name in paths:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            ref = root / 'reference.json'
            manifest = root / 'data/processed/manifest.json'
            manifest.write_text(json.dumps(dict(calibration_rest_offset_v=.01,
                                                output_sha256={'input.csv': digest})))
            ref.write_text(json.dumps(dict(source_sha256={'source.py': digest},
                protocol_sha256=digest, dependency_lock_sha256=digest,
                data_sha256={'input.csv': digest}, packages={'numpy': 'fixture'},
                data_settings={'calibration_rest_offset_v': .01},
                source_record_sha256='fixture-record')))
            with patch.object(v5_aligned, 'ROOT', root), patch.object(v5_aligned, 'REFERENCE', ref):
                environment = dict(packages={'numpy': 'fixture'})
                self.assertEqual(v5_aligned.verify_alignment(environment)['checked_input_files'], 4)
                with self.assertRaisesRegex(ValueError, 'numpy'):
                    v5_aligned.verify_alignment(dict(packages={'numpy': 'wrong'}))
                original = manifest.read_text()
                manifest.write_text(original.replace('0.01', '0.02'))
                with self.assertRaisesRegex(ValueError, 'calibration_rest_offset'):
                    v5_aligned.verify_alignment(environment)
                manifest.write_text(original)
                (root / paths[-1]).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'input.csv'):
                    v5_aligned.verify_alignment(environment)


NUMERICAL_STACK = all(importlib.util.find_spec(name) is not None
                      for name in ['torch', 'botorch', 'gpytorch', 'cma', 'pandas', 'filelock'])


@unittest.skipUnless(NUMERICAL_STACK, 'requires the project numerical environment')
class AlignedExecutionTests(unittest.TestCase):
    """Small synthetic execution checks; never use Battery's real implementation."""
    def setUp(self):
        from bo_vs_es import followup_v4
        self.v4 = followup_v4
        self.config = v5.load_config(v5_aligned.DEFAULT_CONFIG)
        self.protocol = json.loads((ROOT / self.config['protocol']).read_text())
        parent = ROOT / 'artifacts'
        parent.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='v5-aligned-test-', dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    @staticmethod
    def objective(u):
        return float(.01 + np.sum((u - np.array([1., 0., 0., 0., 0.])) ** 2))

    def test_initial_designs_and_indexed_candidate_streams(self):
        from bo_vs_es.followup import candidate_pool
        from bo_vs_es.objectives.models import Domain
        domain = Domain('ball', np.eye(5))
        for design in [20, 30, 39]:
            expected = domain.sample(self.v4.stream(self.config, design), 20)
            np.testing.assert_array_equal(v5_aligned.initial_points(self.config, design), expected)
            for step, kind in [(20, 2), (21, 2), (20, 3)]:
                direct = np.random.default_rng(np.random.SeedSequence([4107, design, 0, step, kind]))
                actual = self.v4.stream(self.config, design, 0, step, kind)
                if kind == 3:
                    np.testing.assert_array_equal(candidate_pool(direct, self.config['pool']),
                                                  candidate_pool(actual, self.config['pool']))
                else:
                    np.testing.assert_array_equal(domain.sample(direct, 256), domain.sample(actual, 256))

    def test_session_resume_uses_v4_executor(self):
        from bo_vs_es import followup
        config = dict(self.config, id=self.root.name)
        jobs = [j for j in v5.jobs_for(config) if j['method'] == 'random'][:2]
        original_execute = self.v4.execute_job
        fresh = []
        def execute(*args, **kwargs):
            row, cached = original_execute(*args, **kwargs)
            if not cached:
                fresh.append(row['run_id'])
            return row, cached
        with patch.object(v5, 'jobs_for', return_value=jobs), \
             patch.object(v5_aligned, 'verify_alignment', return_value={'fixture': True}), \
             patch.object(followup, 'provenance', return_value={}), \
             patch.object(self.v4, 'Battery', side_effect=lambda *a: self.objective), \
             patch.object(self.v4, 'execute_job', side_effect=execute), \
             patch.object(followup, 'execute_job', side_effect=AssertionError('Used V3 executor')), \
             patch.object(v5, 'session_limit_reached', side_effect=lambda start, now, limit: limit == 1 and bool(fresh)):
            self.assertEqual(v5.run(config, {}, 1, 'synthetic-test'), 0)
            path = self.root / 'collection.json'
            self.assertEqual(json.loads(path.read_text())['status'], 'paused')
            self.assertEqual(len(fresh), 1)
            self.assertEqual(v5.run(config, {}, None, 'synthetic-test'), 0)
            metadata = json.loads(path.read_text())
            self.assertEqual(metadata['status'], 'complete')
            self.assertEqual(len(fresh), 2)
        # Baseline replay with V4 job metadata: identical candidates and losses.
        baseline = dict(jobs[0], block='battery_bo', stage='confirmation', budget=100, checkpoints=[100])
        with patch.object(self.v4, 'Battery', side_effect=lambda *a: self.objective):
            row, cached = original_execute(baseline, self.config, self.protocol,
                                           self.root / 'reference', {}, objective=self.objective)
        self.assertEqual(row['status'], 'complete', row.get('error'))
        from bo_vs_es.v5_analysis import read_csv
        actual = read_csv(ROOT / metadata['rows'][v5.identity(jobs[0])]['run_path'] / 'evaluations.csv')
        expected = read_csv(ROOT / row['run_path'] / 'evaluations.csv')
        self.assertEqual([(r['u'], r['loss']) for r in actual[:100]],
                         [(r['u'], r['loss']) for r in expected])

    def test_handlers_checkpoints_and_verified_export(self):
        from bo_vs_es.v5_analysis import export, read_csv
        jobs = v5.jobs_for(self.config)
        rows = {}
        assessed = []
        def assessment(label):
            assessed.append(label)
            return self.objective
        for handler in ['resample', 'project']:
            job = next(j for j in jobs if j['method'] == 'cma_full' and j['handler'] == handler)
            with patch.object(self.v4, 'Battery', side_effect=assessment):
                row, cached = self.v4.execute_job(job, self.config, self.protocol,
                    self.root / 'runs', {}, objective=self.objective)
            self.assertEqual(row['status'], 'complete', row.get('error'))
            self.assertEqual(row['evaluations'], 300)
            self.assertEqual(row['assessment_calls'], 4)
            self.assertEqual([c['calls'] for c in row['checkpoints']], [100, 300])
            self.assertGreater(row['repaired' if handler == 'project' else 'rejected'], 0)
            rows[v5.identity(job)] = row
            # Verified reuse performs no extra objective or assessment calls.
            repeated, cached = self.v4.execute_job(job, self.config, self.protocol,
                self.root / 'runs', {}, objective=lambda u: self.fail('Re-evaluated cached job'))
            self.assertTrue(cached)
        self.assertEqual(len(assessed), 8)
        path = self.root / 'collection.json'
        path.write_text(json.dumps(dict(config=self.config, protocol=self.protocol,
            provenance={}, provenance_id=v5.identity({}), jobs=jobs, rows=rows)))
        output = export(path)
        audit = json.loads((output / 'validation.json').read_text())
        self.assertEqual(audit['completed_runs'], 2)
        self.assertEqual(audit['complete_design_groups'], 0)
        self.assertEqual(audit['status'], 'partial')
        curves = read_csv(output / 'incumbents.csv')
        self.assertEqual(len(curves), 600)
        self.assertTrue(all(r['complete_group'] == 'False' for r in curves))
        designs = read_csv(output / 'initial_designs.csv')
        self.assertEqual(len(designs), 1)
        self.assertEqual(designs[0]['verified_against_v4'], 'True')
        self.assertEqual(designs[0]['initial_u_sha256'],
                         v5.identity(v5_aligned.initial_points(self.config, int(designs[0]['initial_design_seed'])).tolist()))
        (ROOT / row['run_path'] / 'recommendation.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Modified'):
            export(path)

    def test_acquisition_prefix_matches_v4_for_each_bo_variant(self):
        from bo_vs_es.v5_analysis import read_csv
        for method in ['bo_iso', 'bo_ard', 'bo_iso_pool']:
            with self.subTest(method=method):
                job = next(j for j in v5.jobs_for(self.config) if j['method'] == method)
                # Shorten trajectories, not GP fitting or constrained acquisition
                # effort. Two SLSQP iterations can leave every restart infeasible.
                # Compare both post-initial proposals (stream steps 20 and 21).
                job = dict(job, budget=23, checkpoints=[23])
                original = dict(job, block='battery_bo', stage='confirmation',
                                budget=22, checkpoints=[22])
                with patch.object(self.v4, 'Battery', side_effect=lambda *a: self.objective):
                    new, _ = self.v4.execute_job(job, self.config, self.protocol,
                        self.root / 'runs', {}, objective=self.objective)
                    old, _ = self.v4.execute_job(original, self.config, self.protocol,
                        self.root / 'reference', {}, objective=self.objective)
                self.assertEqual(new['status'], 'complete', new.get('error'))
                self.assertEqual(old['status'], 'complete', old.get('error'))
                actual = read_csv(ROOT / new['run_path'] / 'evaluations.csv')
                expected = read_csv(ROOT / old['run_path'] / 'evaluations.csv')
                self.assertEqual(len(actual), 23)
                self.assertEqual(len(expected), 22)
                self.assertEqual([(r['u'], r['loss']) for r in actual[:22]],
                                 [(r['u'], r['loss']) for r in expected])


if __name__ == '__main__':
    unittest.main()
