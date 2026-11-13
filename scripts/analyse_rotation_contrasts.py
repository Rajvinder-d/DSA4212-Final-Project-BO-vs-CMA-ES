"""Post hoc V4 battery method contrasts from committed exports; no optimisation."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bo_vs_es.followup_analysis import group_intervals

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/followup_v4/rotation_pairs.csv'
OUT = ROOT / 'artifacts/report-checks/rotation'
DESIGNS = range(20, 40)
ORIENTATIONS = [404, 505, 606]
COMPARISONS = [
    ('ARD minus scalar, continuous', 'battery_bo', 'bo_ard', 'bo_iso', 100, 1),
    ('ARD minus scalar, pool', 'battery_bo', 'bo_ard_pool', 'bo_iso_pool', 100, 1),
    ('Diagonal minus full CMA', 'cma_ball', 'cma_diag', 'cma_full', 100, 5),
    ('Diagonal minus full CMA', 'cma_ball', 'cma_diag', 'cma_full', 300, 5),
]


def analyse(frame):
    """Reject incomplete/malformed selected arms; pair before design averaging."""
    selected = frame[
        (frame.block.eq('battery_bo') & frame.method.isin(
            ['bo_ard', 'bo_iso', 'bo_ard_pool', 'bo_iso_pool'])) |
        (frame.block.eq('cma_ball') & frame.method.isin(['cma_diag', 'cma_full']))
    ].copy()
    keys = ['block', 'method', 'calls', 'initial_design_seed', 'search_replicate', 'orientation']
    expected = set()
    for _, block, left, right, calls, replicates in COMPARISONS:
        expected.update(itertools.product([block], [left, right], [calls], DESIGNS,
                                          range(replicates), ORIENTATIONS))
    if selected.duplicated(keys).any() or set(selected[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Expected exactly the complete 20-design, three-rotation arm grid')
    handlers = np.where(selected.block.eq('battery_bo'), 'none', 'resample')
    if not (selected.complete_group.eq(True).all() and selected.objective.eq('battery').all()
            and selected.domain.eq('ball').all() and (selected.handler == handlers).all()):
        raise ValueError('Expected complete battery groups in the ball with prescribed handlers')
    losses = selected[['loss', 'loss_identity']].to_numpy(dtype=float)
    if not np.isfinite(losses).all() or (losses <= 0).any():
        raise ValueError('Log ratios require finite, positive losses')
    identity_keys = keys[:-1]
    if selected.groupby(identity_keys).loss_identity.nunique().ne(1).any():
        raise ValueError('Identity loss must agree across rotations of the same run')
    selected['computed_effect'] = np.log10(selected.loss / selected.loss_identity)
    if not np.allclose(selected.computed_effect, selected.effect, rtol=1e-12, atol=1e-14):
        raise ValueError('Stored rotation effect disagrees with exported losses')
    results = []
    index = ['initial_design_seed', 'search_replicate', 'orientation']
    for label, block, left, right, calls, replicates in COMPARISONS:
        subset = selected[selected.block.eq(block) & selected.calls.eq(calls)
                          & selected.method.isin([left, right])]
        wide = subset.pivot(index=index, columns='method', values='computed_effect')
        paired = (wide[left] - wide[right]).rename('effect').reset_index()
        design = paired.groupby(['initial_design_seed', 'orientation'], as_index=False).effect.mean()
        design['contrast'], design['calls'], design['replicates'] = label, calls, replicates
        results.append(design)
    pairs = pd.concat(results, ignore_index=True)
    intervals = group_intervals(pairs, ['contrast', 'calls', 'orientation'], 'effect',
                               {'bootstrap_seed': 4129, 'bootstrap_resamples': 5000})
    intervals['interpretation'] = ('post hoc; signed rotation-effect difference; pointwise 95% '
                                   'design-bootstrap interval; conditional on selected rotations')
    return pairs, intervals


def main(output_dir=OUT):
    output_dir = Path(output_dir)
    pairs, intervals = analyse(pd.read_csv(SOURCE, keep_default_na=False))
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'rotation_method_pairs.csv', index=False)
    intervals.to_csv(output_dir / 'rotation_method_intervals.csv', index=False)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = [SOURCE, Path(__file__), ROOT / 'src/bo_vs_es/followup_analysis.py',
              ROOT / 'configs/experiments/followup_v4.json']
    audit = dict(
        role='post hoc report-stage analysis; no optimisation or historical export changes',
        estimand='mean_design[R_left - R_right], R=log10(rotated loss / identity loss)',
        design_seeds=list(DESIGNS), orientations=ORIENTATIONS, contrasts=12, design_pairs=len(pairs),
        bootstrap=dict(resamples=5000, seed=4129, unit='initial design', interval='percentile 95%, pointwise'),
        cma_aggregation='pair replicate effects, then average five within each design',
        versions=dict(numpy=np.__version__, pandas=pd.__version__),
        inputs={str(p.relative_to(ROOT)): digest(p) for p in inputs},
        outputs={p.name: digest(p) for p in sorted(output_dir.glob('rotation_method_*.csv'))})
    (output_dir / 'validation.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(intervals.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Write recomputed tables to a separate directory; preserve submitted exports')
    main(parser.parse_args().output_dir)
