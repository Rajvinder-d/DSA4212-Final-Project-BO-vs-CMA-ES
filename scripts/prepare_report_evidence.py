"""Validate saved boundary/conditioning displays and render Figure 3; no experiments."""
import argparse
import csv
import itertools
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_intervals(path, keys, expected, groups):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = [r for r in rows if tuple(r[k] for k in keys) in expected]
    observed = [tuple(r[k] for k in keys) for r in selected]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(f'{path}: missing or duplicate interval rows')
    for row in selected:
        low, estimate, high = (float(row[k]) for k in ['low', 'estimate', 'high'])
        if (int(row['design_groups']) != groups or not all(map(math.isfinite, [low, estimate, high]))
                or not low <= estimate <= high):
            raise ValueError(f'{path}: invalid interval or design count')
    return selected


def display_data():
    v4 = ROOT / 'results/followup_v4'
    keys = ['method', 'calls', 'orientation']
    expected = set(itertools.product(['cma_full', 'cma_diag'], ['100', '300'], ['404', '505', '606']))
    boundary = read_intervals(v4 / 'boundary_intervals.csv', keys, expected, 20)
    changes = read_intervals(v4 / 'boundary_budget_intervals.csv', ['method', 'orientation'],
                             set(itertools.product(['cma_full', 'cma_diag'], ['404', '505', '606'])), 20)
    if not all(float(r['low']) <= 0 <= float(r['high']) for r in boundary + changes):
        raise ValueError('The report’s unresolved boundary finding no longer matches the exports')
    lookup = {tuple(r[k] for k in keys): r for r in boundary}
    table = ['| CMA variant | Rotation | 100 calls | 300 calls |', '| --- | --- | --- | --- |']
    for method, label in [('cma_full', 'Full'), ('cma_diag', 'Diagonal')]:
        for orientation in ['404', '505', '606']:
            values = []
            for calls in ['100', '300']:
                row = lookup[method, calls, orientation]
                values.append(f"{float(row['estimate']):.3f} [{float(row['low']):.3f}, {float(row['high']):.3f}]")
            table.append(f'| {label} | Q{orientation} | ' + ' | '.join(values) + ' |')
    expected = set(itertools.product(['quadratic_control'], ['1.0', '100.0', '10000.0'],
                                    ['bo_iso', 'bo_ard'], ['none'], ['100'], ['202', '303']))
    conditioning = read_intervals(ROOT / 'results/followup_v3/rotation_effects_by_orientation.csv',
        ['block', 'condition', 'method', 'handler', 'calls', 'orientation'], expected, 5)
    return '\n'.join(table), conditioning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check source table and exports without rendering')
    args = parser.parse_args()
    table, conditioning = display_data()
    if table not in (ROOT / 'report/report.md').read_text():
        raise ValueError('Boundary table in report.md differs from saved exports:\n' + table)
    if not args.check:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True, layout='constrained')
        for ax, orientation in zip(axes, ['202', '303']):
            for method, label, colour, marker in [('bo_iso', 'Scalar BO', '#0072B2', 'o'),
                                                  ('bo_ard', 'ARD BO', '#D55E00', 's')]:
                rows = sorted((r for r in conditioning if r['method'] == method and r['orientation'] == orientation),
                              key=lambda r: float(r['condition']))
                x = [float(r['condition']) for r in rows]
                y = [float(r['estimate']) for r in rows]
                errors = [[v-float(r['low']) for r,v in zip(rows,y)], [float(r['high'])-v for r,v in zip(rows,y)]]
                ax.errorbar(x, y, yerr=errors, fmt=marker+'-', color=colour, label=label, capsize=3, linewidth=1.3)
            ax.set(xscale='log', xticks=[1,100,10000], xticklabels=['1','100','10,000'],
                   xlabel='Quadratic condition number κ', title=f'Q{orientation}', ylim=(-.7,2.35))
            ax.axhline(0, color='.5', linestyle='--', linewidth=.8)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('Mean log10(rotated / identity loss)')
        axes[0].legend(frameon=False)
        for extension in ['png', 'pdf']:
            fig.savefig(ROOT / f'report/figures/figure3.{extension}', dpi=300)
        plt.close(fig)
    print('Verified 12 boundary intervals, six budget changes and 12 conditioning intervals; report table matches.')


if __name__ == '__main__':
    main()
