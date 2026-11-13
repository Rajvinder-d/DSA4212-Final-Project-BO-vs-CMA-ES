"""Report-stage inference must preserve pairing and the independent design unit."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    'report_contrasts', Path(__file__).resolve().parents[1] / 'scripts/analyse_rotation_contrasts.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.fixture
def sample():
    rows = []
    for _, block, left, right, calls, replicates in analysis.COMPARISONS:
        for method in [left, right]:
            for seed in analysis.DESIGNS:
                for rep in range(replicates):
                    for orientation in analysis.ORIENTATIONS:
                        effect = ((seed - 20) / 100 + rep / 10) if method == left else 0.
                        rows.append(dict(block=block, method=method, calls=calls, initial_design_seed=seed,
                            search_replicate=rep, orientation=orientation, loss=2 * 10**effect,
                            loss_identity=2., effect=effect, complete_group=True, objective='battery',
                            domain='ball', handler='none' if block == 'battery_bo' else 'resample'))
    return pd.DataFrame(rows)


def test_pairing_bootstrap_and_design_aggregation(sample):
    pairs, intervals = analysis.analyse(sample)
    assert len(pairs) == 240 and len(intervals) == 12
    assert set(intervals.design_groups) == {20}
    cma = pairs[pairs.contrast.eq('Diagonal minus full CMA')]
    np.testing.assert_allclose(cma.effect, (cma.initial_design_seed - 20) / 100 + .2)
    row = intervals[intervals.contrast.eq('Diagonal minus full CMA')].iloc[0]
    values = np.arange(20) / 100 + .2
    draws = np.random.default_rng(4129).choice(values, (5000, 20), replace=True).mean(axis=1)
    np.testing.assert_allclose([row.estimate, row.low, row.high],
                              [values.mean(), *np.quantile(draws, [.025, .975])])
    shuffled_pairs, shuffled_intervals = analysis.analyse(sample.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(pairs, shuffled_pairs)
    pd.testing.assert_frame_equal(intervals, shuffled_intervals)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'zero', 'nan', 'effect', 'handler', 'incomplete', 'identity'])
def test_reject_invalid_evidence(sample, fault):
    if fault == 'missing':
        sample = sample.iloc[1:]
    elif fault == 'duplicate':
        sample = pd.concat([sample, sample.iloc[[0]]])
    else:
        field, value = {'zero': ('loss', 0), 'nan': ('loss', np.nan), 'effect': ('effect', .7),
                        'handler': ('handler', 'project'), 'incomplete': ('complete_group', False),
                        'identity': ('loss_identity', 3)}[fault]
        sample.loc[0, field] = value
    with pytest.raises(ValueError):
        analysis.analyse(sample)


def test_saved_export_coverage():
    pairs, intervals = analysis.analyse(pd.read_csv(analysis.SOURCE, keep_default_na=False))
    assert len(pairs) == 240 and len(intervals) == 12
    assert set(pairs.initial_design_seed) == set(range(20, 40))
