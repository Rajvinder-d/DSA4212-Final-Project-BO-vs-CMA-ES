"""Plot verified saved trajectories; optional JAX computation, Matplotlib output."""
import csv
import importlib.metadata
from collections import defaultdict

import numpy as np


LABELS = {
    ('bo_iso', 'none'): 'Scalar GP-BO',
    ('bo_ard', 'none'): 'ARD GP-BO',
    ('bo_iso_pool', 'none'): 'Pool GP-BO',
    ('random', 'none'): 'Random',
    ('cma_full', 'resample'): 'CMA resampling',
    ('cma_full', 'project'): 'CMA projection',
}


def saved_curves(losses, backend):
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 2 or np.any(losses <= 0) or not np.isfinite(losses).all():
        raise ValueError('Expected a finite positive run-by-evaluation loss matrix')
    expected = 1000 * np.sqrt(np.minimum.accumulate(losses, axis=1))
    if backend == 'numpy':
        return expected
    if backend != 'jax':
        raise ValueError('Unknown plotting computation backend')
    import jax
    from bo_vs_es.v5_jax import cpu_float64, incumbent_rmse_mv
    result = np.asarray(incumbent_rmse_mv(
        jax.device_put(losses, cpu_float64())).block_until_ready())
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    return result


def make_plots(output, metadata, complete_seeds, comparisons, backend='numpy'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from bo_vs_es.v5 import ROOT

    settings = dict(computation_backend=backend, matplotlib=matplotlib.__version__,
                    design_groups=len(complete_seeds), files=[])
    if backend == 'jax':
        settings.update(jax=importlib.metadata.version('jax'),
                        jaxlib=importlib.metadata.version('jaxlib'))
    selected = [r for r in metadata['rows'].values()
                if r['status'] == 'complete' and r['initial_design_seed'] in complete_seeds]
    if not selected:
        settings['status'] = 'no complete design groups to plot'
        return settings
    traces = []
    for row in selected:
        with (ROOT / row['run_path'] / 'evaluations.csv').open(newline='') as stream:
            traces.append(list(csv.DictReader(stream)))
    curves = saved_curves([[float(r['loss']) for r in trace] for trace in traces], backend)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for color, (arm, label) in zip(plt.get_cmap('tab10').colors, LABELS.items()):
        designs = defaultdict(list)
        for row, trace, curve in zip(selected, traces, curves):
            if (row['method'], row['handler']) != arm:
                continue
            designs[row['initial_design_seed']].append(curve)
            axes[1].plot([float(r['elapsed_s']) for r in trace], curve, color=color, alpha=.2, lw=.7)
        if not designs:
            continue
        means = np.stack([np.mean(v, axis=0) for v in designs.values()])
        axes[0].plot(np.arange(1, means.shape[1] + 1), np.median(means, axis=0), color=color, label=label)
        axes[1].plot([], [], color=color, label=label)
    axes[0].set(xlabel='Counted objective evaluations', ylabel='Best calibration RMSE (mV)', yscale='log')
    axes[0].axvline(100, ls='--', color='black', lw=.7)
    axes[1].set(xlabel='Elapsed search time (s)', ylabel='Best calibration RMSE (mV)', xscale='log', yscale='log')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    fig.suptitle(f'Battery ball: {len(complete_seeds)} complete initial designs; 300 calls each')
    fig.tight_layout()
    path = output / 'quality_and_time.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    settings['files'].append(path.name)

    handler = [r for r in comparisons if r['contrast'] == 'CMA projection / CMA resampling']
    if handler:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for index, row in enumerate(handler):
            ax.scatter(row['estimate'], index, color='black')
            if row['low'] is not None:
                ax.hlines(index, row['low'], row['high'], color='black')
        ax.set_yticks(range(len(handler)), [f"{r['calls']} calls ({r['role']})" for r in handler])
        ax.axvline(0, ls='--', color='grey', lw=.8)
        ax.set(xlabel='Mean design-level log10 MSE ratio: projection / resampling',
               title='Negative favours projection; 95% design-bootstrap intervals')
        fig.tight_layout()
        path = output / 'handler_contrast.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        settings['files'].append(path.name)
    (output / 'figure_notes.md').write_text(
        '# V5 figure definitions\n\n'
        f'{len(complete_seeds)} complete initial-design groups; identity orientation, '
        'five-dimensional ball, deterministic DST calibration loss. Every arm has '
        '20 paid initial observations and 300 total calls. CMA has '
        f"{len(metadata['config']['cma_search_replicates'])} search "
        'replicates per design and handler; BO/random have one.\n\n'
        '`quality_and_time.png`: the calls panel takes the mean RMSE curve of '
        'replicates within each design, then the median across designs. The time '
        'panel displays individual trajectories without an aggregate time ranking. '
        'Search time excludes compilation/setup and post-search FUDS assessment.\n\n'
        '`handler_contrast.png`: mean log10 loss ratios after averaging replicates '
        'within designs; percentile intervals resample whole designs. The 100-call '
        'comparison is primary, 300 calls secondary. Read coverage and target '
        'attainment before interpreting partial exports. Figure creation does '
        'not establish support for the hypothesis.\n', encoding='utf-8')
    settings['status'] = 'written'
    return settings
