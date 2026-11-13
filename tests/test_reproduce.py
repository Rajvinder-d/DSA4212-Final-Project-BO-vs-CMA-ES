"""Reproduction orchestration checks; no scientific dependencies or experiments."""
import hashlib
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'reproduce', Path(__file__).resolve().parents[1] / 'scripts/reproduce.py')
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class ReproductionTests(unittest.TestCase):
    def test_saved_comparison_allows_roundoff_only(self):
        saved = {'estimate': [2.543157040798154, 0.0], 'count': 19, 'ok': True, 'id': 'Q404'}
        fresh = dict(saved, estimate=[2.5431570407981536, 1e-16])
        reproduce.compare_saved_values(fresh, saved)
        for changed, location in [
            (dict(saved, estimate=[2.54315704, 0.0]), r'root.estimate\[0\]'),
            (dict(saved, count=20), 'root.count'),
            (dict(saved, count=19.0), 'root.count'),
            (dict(saved, ok=1), 'root.ok'),
            (dict(saved, id='Q505'), 'root.id'),
            (dict(saved, estimate=[]), 'root.estimate'),
            (dict(saved, extra=0), 'root'),
        ]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, location):
                reproduce.compare_saved_values(changed, saved)
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'root'):
                reproduce.compare_saved_values(value, value)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def prepared_fixture(self):
        inputs = {'a.csv': b'x\n1\n', 'b.csv': b'y\n2\n'}
        reference = {'data_sha256': {n: hashlib.sha256(b).hexdigest() for n, b in inputs.items()},
                     'data_settings': {'window_s': 3600}}
        self.write('configs/followup_v5/v4_reference.json', reference)
        manifest = dict(window_s=3600, prepared_utc='unchanged', output_sha256={})
        for name, data in inputs.items():
            windows = data.replace(b'\n', b'\r\n')
            path = self.root / 'data/processed' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(windows)
            manifest['output_sha256'][name] = hashlib.sha256(windows).hexdigest()
        self.write('data/processed/manifest.json', manifest)
        return reference

    def test_default_plan_never_runs_commands_or_writes_files(self):
        with patch.object(reproduce.sys, 'argv', ['reproduce.py']), \
                patch.object(reproduce, 'run_stage') as execute, \
                patch.object(reproduce, 'preflight') as check, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reproduce.main(), 0)
        execute.assert_not_called()
        check.assert_not_called()

    def test_failed_preflight_prevents_execution_even_with_run(self):
        with patch.object(reproduce.sys, 'argv', ['reproduce.py', '--run']), \
                patch.object(reproduce, 'run_stage') as execute, \
                patch.object(reproduce, 'preflight', return_value={'ready': False}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reproduce.main(), 1)
        execute.assert_not_called()

    def test_suite_dependency_order_and_historical_separation(self):
        self.assertEqual(reproduce.selected_stages('report', 'all'),
                         ['prepare', 'tests', 'v1', 'v2', 'v3', 'v4', 'v5-aligned'])
        self.assertIn('v5-original', reproduce.selected_stages('all', 'all'))
        self.assertEqual(reproduce.selected_stages('all', 'v4'), ['v4'])
        flattened = str([reproduce.commands(s) for s in reproduce.STAGES])
        self.assertNotIn('configs/experiments/main.json', flattened)

    def test_failed_command_stops_later_stage_commands(self):
        calls = []
        def run(command):
            calls.append(command)
            return 2
        self.assertEqual(reproduce.run_stage('v4', run, self.root), 2)
        self.assertEqual(calls, [['scripts/record_v4_baseline.py']])

    def test_failed_v5_still_exports_partial_coverage_and_returns_failure(self):
        self.write('artifacts/boundary-comparison-v5-v4-aligned/collection.json', {})
        calls = []
        def run(command):
            calls.append(command)
            return 1 if len(calls) == 1 else 0
        self.assertEqual(reproduce.run_stage('v5-aligned', run, self.root), 1)
        self.assertEqual(calls, reproduce.commands('v5-aligned'))

    def test_failed_v5_before_collection_does_not_attempt_export(self):
        calls = []
        def run(command):
            calls.append(command)
            return 1
        self.assertEqual(reproduce.run_stage('v5-aligned', run, self.root), 1)
        self.assertEqual(len(calls), 1)

    def test_successful_exit_does_not_mislabel_paused_v5_as_complete(self):
        self.write('artifacts/boundary-comparison-v5-v4-aligned/collection.json',
                   {'status': 'paused', 'jobs': [1, 2], 'rows': {'first': {'status': 'complete'}}})
        calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            code = reproduce.run_stage('v5-aligned', lambda c: calls.append(c) or 0, self.root)
        self.assertEqual(code, 2)
        self.assertEqual(calls, reproduce.commands('v5-aligned'))

    def test_v5_success_requires_every_planned_row(self):
        path = self.write('artifacts/boundary-comparison-v5-v4-aligned/collection.json',
                          {'status': 'complete', 'jobs': [1, 2],
                           'rows': {'first': {'status': 'complete'}}})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(reproduce.run_stage('v5-aligned', lambda c: 0, self.root), 2)
        record = json.loads(path.read_text())
        record['rows']['second'] = {'status': 'complete'}
        path.write_text(json.dumps(record))
        self.assertEqual(reproduce.run_stage('v5-aligned', lambda c: 0, self.root), 0)

    def test_export_audit_rejects_modified_or_missing_evidence(self):
        table = self.root / 'values.csv'
        table.write_bytes(b'x\n1\n')
        manifest = self.write('validation.json',
                              {'outputs': {'values.csv': hashlib.sha256(table.read_bytes()).hexdigest()}})
        self.assertEqual(reproduce.verify_export_manifest(manifest), 1)
        table.write_bytes(b'x\n2\n')
        with self.assertRaisesRegex(ValueError, 'missing or modified'):
            reproduce.verify_export_manifest(manifest)
        table.unlink()
        with self.assertRaisesRegex(ValueError, 'missing or modified'):
            reproduce.verify_export_manifest(manifest)

    def test_v3_requires_both_history_sources_and_latest_complete_v2(self):
        self.assertEqual(len(reproduce.v3_prerequisites(self.root)), 2)
        path = self.root / 'artifacts/pilot_v1/index.csv'
        path.parent.mkdir(parents=True)
        path.write_text('fixture')
        self.write('artifacts/diagnostics_v2/001/manifest.json', {'status': 'complete'})
        self.assertEqual(reproduce.v3_prerequisites(self.root), [])
        self.write('artifacts/diagnostics_v2/002/manifest.json', {'status': 'failed'})
        self.assertIn('latest V2', reproduce.v3_prerequisites(self.root)[0])

    def test_preparation_reuses_existing_data_and_timestamp(self):
        self.prepared_fixture()
        reproduce.normalise_prepared_data(self.root)
        path = self.root / 'data/processed/manifest.json'
        before = path.read_bytes()
        calls = []
        self.assertEqual(reproduce.run_stage('prepare', lambda c: calls.append(c) or 0, self.root), 0)
        self.assertEqual(calls, reproduce.commands('prepare')[:1])
        self.assertEqual(path.read_bytes(), before)

    def test_windows_normalisation_requires_exact_recorded_bytes(self):
        reference = self.prepared_fixture()
        reproduce.normalise_prepared_data(self.root)
        self.assertEqual(reproduce.data_errors(self.root, reference), [])
        manifest = json.loads((self.root / 'data/processed/manifest.json').read_text())
        self.assertEqual(manifest['prepared_utc'], 'unchanged')

    def test_mismatched_data_never_partially_normalised(self):
        self.prepared_fixture()
        first = self.root / 'data/processed/a.csv'
        before = first.read_bytes()
        (self.root / 'data/processed/b.csv').write_bytes(b'y\r\n99\r\n')
        with self.assertRaisesRegex(ValueError, 'beyond line endings'):
            reproduce.normalise_prepared_data(self.root)
        self.assertEqual(first.read_bytes(), before)

    def test_preflight_reports_missing_packages_without_importing_them(self):
        reference = self.prepared_fixture()
        reference.update(source_sha256={}, protocol_sha256='missing', dependency_lock_sha256='missing',
                         packages={'torch': 'test-version'})
        self.write('configs/followup_v5/v4_reference.json', reference)
        with patch.object(reproduce.importlib.metadata, 'version',
                          side_effect=reproduce.importlib.metadata.PackageNotFoundError):
            result = reproduce.preflight(['v5-aligned'], self.root)
        self.assertFalse(result['ready'])
        self.assertIn('Missing dependency: torch', result['blockers'])
        self.assertFalse(result['experiments_executed'])


if __name__ == '__main__':
    unittest.main()
