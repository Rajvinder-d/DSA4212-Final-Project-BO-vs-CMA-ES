"""Saved report display coverage; no optimisation runs."""
import csv
import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('evidence', Path(__file__).resolve().parents[1] / 'scripts/prepare_report_evidence.py')
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def test_report_displays_match_saved_evidence():
    table, conditioning = evidence.display_data()
    assert len(table.splitlines()) == 8
    assert len(conditioning) == 12
    assert table in (evidence.ROOT / 'report/report.md').read_text()


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'nan', 'reversed', 'groups'])
def test_interval_validation_rejects_corruption(tmp_path, corruption):
    row = dict(method='bo_iso', design_groups='5', low='-1', estimate='0', high='1')
    rows = [row]
    if corruption == 'missing': rows = []
    elif corruption == 'duplicate': rows *= 2
    elif corruption == 'nan': row['estimate'] = 'nan'
    elif corruption == 'reversed': row['low'] = '2'
    elif corruption == 'groups': row['design_groups'] = '4'
    path = tmp_path / 'intervals.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError):
        evidence.read_intervals(path, ['method'], {('bo_iso',)}, 5)
