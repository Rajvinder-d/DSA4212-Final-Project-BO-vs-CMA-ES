"""Read-only artifact checks and explicitly generated v3 tables and figures."""
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bo_vs_es import ROOT
from bo_vs_es.experiments.runner import valid_completed
from bo_vs_es.objectives.data import sha256
from bo_vs_es.objectives.models import Domain, physical
from bo_vs_es.followup import atomic_dump, load_config, now

COLORS = dict(bo_iso='#0072B2', bo_ard='#56B4E9', bo_iso_pool='#009E73',
              cma_full='#D55E00', cma_diag='#CC79A7', random='#777777')


def group_intervals(frame, keys, value, config):
    """Average associated replicates/orientations first, resample whole designs."""
    rows = []
    for group, subset in frame.groupby(keys, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        values = subset.groupby('initial_design_seed')[value].mean().to_numpy()
        rng = np.random.default_rng(config['bootstrap_seed'])
        samples = rng.choice(values, (config['bootstrap_resamples'], len(values)), replace=True).mean(axis=1)
        rows.append(dict(zip(keys, group), design_groups=len(values), estimate=float(values.mean()),
                         low=float(np.quantile(samples, .025)), high=float(np.quantile(samples, .975)),
                         interpretation='conditional on selected objective and orientations; exploratory interval with few design groups'))
    return pd.DataFrame(rows)


def target_record(trace, calls, threshold):
    selected = trace.iloc[:calls]
    hits = selected[selected.loss <= threshold ** 2]
    if hits.empty:
        return dict(hit=False, calls_to_target=None, seconds_to_target=None,
                    censored_at_calls=calls, outcome='not reached within budget')
    row = hits.iloc[0]
    return dict(hit=True, calls_to_target=int(row.evaluation), seconds_to_target=float(row.elapsed_s),
                censored_at_calls=calls, outcome='reached')


def paired_budget_changes(frame, keys, value):
    """Pair checkpoints before aggregation; never compare two marginal intervals."""
    paired = frame.pivot(index=keys, columns='calls', values=value).reindex(columns=[100, 300])
    paired = paired.dropna(subset=[100, 300])
    paired['change_300_minus_100'] = paired[300] - paired[100]
    return paired.rename(columns={100: 'value_100', 300: 'value_300'}).reset_index()


def read_collection(config):
    root = ROOT / 'artifacts' / config['id']
    path = root / 'confirmation/collection.json'
    if not path.exists():
        return root, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    metadata = json.loads(path.read_text())
    expected = pd.DataFrame(metadata['jobs']).groupby(['block', 'initial_design_seed']).size()
    actual = pd.DataFrame(metadata['rows'])
    completed = actual[actual.status.eq('complete')].groupby(['block', 'initial_design_seed']).size() if len(actual) else pd.Series(dtype=int)
    coverage = expected.rename('planned').to_frame().join(completed.rename('complete')).fillna(0).reset_index()
    coverage['complete_group'] = coverage.planned.eq(coverage.complete)
    complete_groups = {(r.block, r.initial_design_seed) for r in coverage.itertuples() if r.complete_group}
    snapshots, targets, health = [], [], []
    initial_by_design = {}
    for row in metadata['rows']:
        folder = ROOT / row['run_path']
        manifest = json.loads((folder / 'manifest.json').read_text())
        if any(sha256(folder / name) != digest for name, digest in manifest['output_sha256'].items()):
            raise ValueError(f"Modified artifact: {row['run_id']}")
        if json.loads((folder / 'summary.json').read_text()) != row:
            raise ValueError('Collection/summary mismatch')
        if row['status'] != 'complete':
            continue
        if not valid_completed(folder, manifest['run_key']):
            raise ValueError(f"Invalid completion: {row['run_id']}")
        trace = pd.read_csv(folder / 'evaluations.csv')
        initial = json.loads((folder / 'initial.json').read_text())
        domain = Domain(row['domain'], np.array(initial['Q']))
        z = np.array([json.loads(x) for x in trace.z])
        u = np.array([json.loads(x) for x in trace.u])
        if not all(domain.feasible(x) for x in z):
            raise ValueError('Infeasible evaluated candidate')
        np.testing.assert_allclose(domain.to_base(z), u, atol=1e-13, rtol=0)
        np.testing.assert_allclose(u[:row['n_init']], initial['u'], atol=1e-13, rtol=0)
        key = (row['domain'], row['initial_design_seed'])
        if key in initial_by_design:
            np.testing.assert_allclose(initial_by_design[key], initial['u'], atol=1e-13, rtol=0)
        initial_by_design[key] = initial['u']
        if row['objective'] == 'battery':
            np.testing.assert_allclose(np.array([json.loads(x) for x in trace.theta]), physical(u), rtol=1e-12)
            if row['assessment_calls'] != 2 * len(row['checkpoints']):
                raise ValueError('Assessment accounting mismatch')
        proposals = [json.loads(s) for s in (folder / 'proposals.jsonl').read_text().splitlines()]
        if (len(proposals) != row['proposals'] or sum(p['rejected'] for p in proposals) != row['rejected']
                or sum(p['repaired'] for p in proposals) != row['repaired']):
            raise ValueError('Proposal accounting mismatch')
        identity = {k: row[k] for k in ['block', 'objective', 'condition', 'method', 'domain', 'handler',
                    'initial_design_seed', 'search_replicate', 'search_seed', 'rotation_seed', 'run_id', 'run_path']}
        identity['orientation'] = -1 if row['rotation_seed'] is None else row['rotation_seed']
        identity['complete_group'] = (row['block'], row['initial_design_seed']) in complete_groups
        for cp in row['checkpoints']:
            prefix = trace.iloc[:cp['calls']]
            if not np.isclose(cp['loss'], prefix.loss.min(), atol=1e-14, rtol=1e-12):
                raise ValueError('Checkpoint is not its prefix minimum')
            np.testing.assert_allclose(cp['u'], u[cp['evaluation'] - 1], atol=1e-13, rtol=0)
            margin = 1 - (np.linalg.norm(cp['u']) if row['domain'] == 'ball' else np.max(np.abs(cp['u'])))
            snapshots.append(dict(**identity, **cp, calibration_rmse_v=float(np.sqrt(cp['loss'])),
                boundary_margin=float(margin), evaluated_boundary_fraction=float(np.mean(
                    (np.linalg.norm(u[:cp['calls']], axis=1) if row['domain'] == 'ball' else np.max(np.abs(u[:cp['calls']]), axis=1)) >= 1 - 1e-8)),
                rejected_fraction=cp['rejected'] / cp['proposals'], repaired_fraction=cp['repaired'] / cp['proposals']))
            if row['objective'] == 'battery':
                for target in config['battery_targets_rmse_v']:
                    targets.append(dict(**identity, calls=cp['calls'], target_rmse_v=target,
                                        **target_record(trace, cp['calls'], target)))
        diagnostic_path = folder / 'diagnostics.jsonl'
        if row['method'].startswith('bo'):
            for line in diagnostic_path.read_text().splitlines():
                d = json.loads(line)
                scales = np.array(d['lengthscales'])
                health.append(dict(**identity, after_evaluations=d['after_evaluations'], fit_status=d['fit_status'],
                    warnings=len(d['warnings']), near_upper_bound=bool(np.any(scales >= 4.95)),
                    near_lower_bound=bool(np.any(scales <= .0303)),
                    lengthscale_min=float(scales.min()), lengthscale_max=float(scales.max())))
    return root, metadata, pd.DataFrame(snapshots), pd.DataFrame(targets), pd.DataFrame(health)


def export_analysis(config_path=None):
    config = load_config(config_path)
    root, metadata, snapshots, targets, health = read_collection(config)
    output = ROOT / 'results/followup_v3'
    output.mkdir(parents=True, exist_ok=True)
    written, figures = [], []
    def table(name, frame):
        frame.to_csv(output / f'{name}.csv', index=False)
        written.append(name + '.csv')
    def figure(name, fig):
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures.append(name + '.png')
    dev_path = root / 'development_complete.json'
    if dev_path.exists():
        dev = json.loads(dev_path.read_text())
        folder = ROOT / dev['result_path']
        for name, digest in dev['output_sha256'].items():
            if sha256(folder / name) != digest:
                raise ValueError('Development evidence checksum mismatch')
        replays = json.loads((folder / 'original_history_replays.json').read_text())
        replay_rows = []
        for r in replays:
            common = dict(source_run_id=r['source_run_id'], history_seed=r['history_seed'], observations=r['observations'],
                          fitted_prediction_gap=r['fitted_control']['posterior_mean_max_abs'],
                          fitted_logei_gap=r['fitted_control']['log_ei_max_abs'])
            for solve in r['acquisition']:
                replay_rows.append(dict(**common, component=f"continuous_{solve['maxiter']}",
                    candidate_distance=solve['candidate_distance'], acquisition_gap=solve['common_model_log_ei_abs_gap']))
            for mode, check in r['finite_pool'].items():
                replay_rows.append(dict(**common, component='pool_' + mode,
                    candidate_distance=check['candidate_distance'],
                    acquisition_gap=abs(check['common_model_log_ei'][0] - check['common_model_log_ei'][1])))
        table('development_replays', pd.DataFrame(replay_rows))
        reference = json.loads((folder / 'ball_reference_diagnostics.json').read_text())
        atomic_dump(output / 'ball_reference.json', reference)
        written.append('ball_reference.json')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        replay_frame = pd.DataFrame(replay_rows)
        for name, rows in replay_frame.groupby('component'):
            axes[0].scatter(rows.observations, np.maximum(rows.candidate_distance, 1e-16), label=name, alpha=.7)
            axes[1].scatter(rows.observations, np.maximum(rows.acquisition_gap, 1e-16), label=name, alpha=.7)
        for ax, label in zip(axes, ['Mapped candidate distance', 'Common-model absolute LogEI gap']):
            ax.set(xlabel='History length', ylabel=label, yscale='log')
            ax.legend(fontsize=7)
        figure('development_mechanism', fig)
    if metadata is not None:
        rows = pd.DataFrame(metadata['rows'])
        table('run_status', rows.drop(columns=['checkpoints'], errors='ignore'))
        planned = pd.DataFrame(metadata['jobs']).groupby(['block', 'initial_design_seed']).size().rename('planned')
        done = rows[rows.status.eq('complete')].groupby(['block', 'initial_design_seed']).size().rename('complete') if len(rows) else pd.Series(dtype=int, name='complete')
        coverage = planned.to_frame().join(done).fillna(0).reset_index()
        coverage['complete_group'] = coverage.planned.eq(coverage.complete)
        table('coverage', coverage)
        table('checkpoints', snapshots)
        table('targets', targets)
        table('optimiser_health', health)
    if len(snapshots):
        usable = snapshots[snapshots.complete_group].copy()
        keys = ['block', 'condition', 'method', 'handler', 'initial_design_seed', 'search_replicate', 'calls']
        baseline = usable[usable.orientation.eq(-1)]
        pairs = usable[usable.orientation.ne(-1)].merge(baseline[keys + ['loss', 'run_id']], on=keys,
            suffixes=('', '_identity'), validate='many_to_one')
        pairs['log10_loss_ratio'] = np.log10(pairs.loss / pairs.loss_identity)
        pairs['absolute_log10_ratio'] = pairs.log10_loss_ratio.abs()
        table('rotation_pairs', pairs)
        boundary = pd.DataFrame()
        if len(pairs):
            group = ['block', 'condition', 'method', 'handler', 'calls']
            table('rotation_effects_by_orientation', group_intervals(pairs, group + ['orientation'], 'log10_loss_ratio', config))
            table('rotation_effects_pooled', group_intervals(pairs, group, 'log10_loss_ratio', config))
            contrasts = []
            comparison_keys = ['block', 'condition', 'handler', 'initial_design_seed', 'search_replicate', 'calls', 'orientation']
            for variant, reference_method in [('bo_iso_pool', 'bo_iso'), ('bo_ard', 'bo_iso'), ('cma_diag', 'cma_full')]:
                compared = pairs[pairs.method.eq(variant)].merge(
                    pairs[pairs.method.eq(reference_method)][comparison_keys + ['absolute_log10_ratio']],
                    on=comparison_keys, suffixes=('', '_reference'), validate='one_to_one')
                if len(compared):
                    compared['contrast'] = variant + ' minus ' + reference_method
                    compared['difference'] = compared.absolute_log10_ratio - compared.absolute_log10_ratio_reference
                    contrasts.append(compared)
            if contrasts:
                contrasts = pd.concat(contrasts, ignore_index=True)
                table('sensitivity_contrasts', group_intervals(contrasts,
                    ['block', 'condition', 'handler', 'calls', 'contrast'], 'difference', config))
            boundary = pairs[pairs.block.eq('battery_boundary')].pivot(index=[
                'initial_design_seed', 'search_replicate', 'orientation', 'method', 'calls'],
                columns='handler', values='log10_loss_ratio')
            if len(boundary):
                boundary['interaction'] = boundary['project'] - boundary['resample']
                boundary = boundary.reset_index()
                table('boundary_interactions', boundary)
                table('boundary_interaction_intervals', group_intervals(boundary, ['method', 'calls', 'orientation'], 'interaction', config))
                fig, ax = plt.subplots(figsize=(8, 4))
                for (method, calls), values in boundary.groupby(['method', 'calls']):
                    means = values.groupby('initial_design_seed').interaction.mean()
                    ax.scatter(means.index, means.values, label=f'{method}, {calls} calls', alpha=.7)
                ax.axhline(0, color='black', lw=1)
                ax.set(xlabel='Initial-design seed', ylabel='Projection minus resampling rotation effect',
                       title='CMA boundary interaction; averages of replicates/orientations within design')
                ax.legend(fontsize=7)
                figure('boundary_interactions', fig)
            q_pairs = pairs[pairs.block.eq('quadratic_control')]
            if len(q_pairs):
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
                for ax, orientation in zip(axes, config['rotations'][1:]):
                    for method, group_q in q_pairs[q_pairs.orientation.eq(orientation)].groupby('method'):
                        estimates = group_q.groupby('condition').log10_loss_ratio.mean()
                        ax.plot(estimates.index, estimates.values, 'o-', color=COLORS[method], label=method)
                    ax.axhline(0, color='black', lw=1)
                    ax.set(xscale='log', xlabel='Condition number', ylabel='Mean log10 rotated / identity loss',
                           title=f'Q{orientation}; within-condition rotation contrasts')
                    ax.legend(fontsize=7)
                figure('quadratic_rotation_contrasts', fig)
        stochastic = []
        identity = baseline[baseline.method.str.startswith('cma')]
        for key, group in identity.groupby(['block', 'condition', 'method', 'handler', 'initial_design_seed', 'calls'], dropna=False):
            for a, b in itertools.combinations(group.itertuples(), 2):
                stochastic.append(dict(zip(['block', 'condition', 'method', 'handler', 'initial_design_seed', 'calls'], key),
                    replicate_a=a.search_replicate, replicate_b=b.search_replicate,
                    log10_loss_ratio=float(np.log10(b.loss / a.loss)),
                    interpretation='identity vs identity sampling variability; overlapping pairs are dependent'))
        table('cma_identity_variability', pd.DataFrame(stochastic))
        cma_state = []
        complete_ids = set(usable.run_id)
        for row in metadata['rows']:
            if row['run_id'] not in complete_ids or not row['method'].startswith('cma'):
                continue
            for line in (ROOT / row['run_path'] / 'diagnostics.jsonl').read_text().splitlines():
                d = json.loads(line)
                eigen = d['eigenvalues']
                cma_state.append(dict(run_id=row['run_id'], block=row['block'], method=row['method'],
                    initial_design_seed=row['initial_design_seed'], orientation=row['rotation_seed'],
                    search_replicate=row['search_replicate'], handler=row['handler'],
                    calls=d['after_evaluations'], sigma=d['sigma'], covariance_spectral_ratio=max(eigen)/min(eigen)))
        table('cma_adaptation', pd.DataFrame(cma_state))
        if cma_state:
            state = pd.DataFrame(cma_state)
            state = state[state.block.eq('battery_ball')]
            if len(state):
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for method, rows_state in state.groupby('method'):
                    grouped = rows_state.groupby(['initial_design_seed', 'calls'])[['sigma', 'covariance_spectral_ratio']].mean()
                    medians = grouped.groupby('calls').median()
                    for ax, metric in zip(axes, ['sigma', 'covariance_spectral_ratio']):
                        ax.plot(medians.index, medians[metric], label=method, color=COLORS[method])
                        ax.axvline(100, ls='--', color='black', lw=1)
                        ax.set(xlabel='Counted calls', ylabel=metric, yscale='log')
                        ax.legend()
                fig.suptitle('CMA adaptation extension; median of within-design averages')
                figure('cma_adaptation', fig)
        if len(targets):
            target_summary = targets[targets.complete_group].groupby(
                ['block', 'method', 'handler', 'calls', 'orientation', 'target_rmse_v'], dropna=False).agg(
                runs=('hit', 'size'), hits=('hit', 'sum'), hit_rate=('hit', 'mean'),
                median_calls_if_reached=('calls_to_target', 'median'),
                median_seconds_if_reached=('seconds_to_target', 'median')).reset_index()
            table('target_summary', target_summary)
        primary = usable[(usable.block == 'battery_ball') & (usable.calls == 100)]
        if len(primary):
            reference = primary[primary.method.eq('bo_iso')]
            comparison = primary[primary.method.eq('bo_iso_pool')].merge(
                reference[['initial_design_seed', 'orientation', 'loss', 'search_s']],
                on=['initial_design_seed', 'orientation'], suffixes=('', '_reference'), validate='one_to_one')
            comparison['log10_loss_ratio'] = np.log10(comparison.loss / comparison.loss_reference)
            comparison['extra_search_s'] = comparison.search_s - comparison.search_s_reference
            table('intervention_performance', comparison)
            table('intervention_loss_intervals', group_intervals(comparison, ['orientation'], 'log10_loss_ratio', config))
            # Collapse search replicates/orientations within each independent design.
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for method, group in primary.groupby('method'):
                for ax, time_axis in zip(axes, [False, True]):
                    curves = []
                    for row in group.itertuples():
                        trace = pd.read_csv(ROOT / row.run_path / 'evaluations.csv').iloc[:100]
                        x = trace.elapsed_s.to_numpy() if time_axis else trace.evaluation.to_numpy()
                        y = np.sqrt(trace.loss.cummin().to_numpy()) * 1000
                        ax.plot(x, y, color=COLORS[method], alpha=.1, lw=.6)
                        if not time_axis:
                            curves.append((row.initial_design_seed, y))
                    if time_axis:
                        ax.plot([], [], color=COLORS[method], label=method)
                    else:
                        designs = sorted({seed for seed, _ in curves})
                        mean_by_design = [np.mean([v for s, v in curves if s == seed], axis=0) for seed in designs]
                        ax.plot(np.arange(1, 101), np.median(mean_by_design, axis=0), color=COLORS[method], label=method)
                    ax.set(yscale='log', xlabel='Elapsed search time (s)' if time_axis else 'Counted calls',
                           ylabel='Best calibration RMSE (mV)')
                    if time_axis:
                        ax.set_xscale('log')
                    ax.legend(fontsize=7)
            fig.suptitle('Battery ball, first 100 calls; faint individual paths; call-panel median of design averages')
            figure('confirmation_quality_and_time', fig)
            paths = []
            for method in ['bo_iso', 'bo_iso_pool', 'bo_ard', 'random']:
                for seed, group in primary[primary.method.eq(method)].groupby('initial_design_seed'):
                    base = group[group.orientation.eq(-1)].iloc[0]
                    u0 = np.array([json.loads(x) for x in pd.read_csv(ROOT / base.run_path / 'evaluations.csv').u.iloc[:100]])
                    for row in group[group.orientation.ne(-1)].itertuples():
                        u1 = np.array([json.loads(x) for x in pd.read_csv(ROOT / row.run_path / 'evaluations.csv').u.iloc[:100]])
                        for i, distance in enumerate(np.linalg.norm(u0 - u1, axis=1)):
                            paths.append(dict(method=method, initial_design_seed=seed, orientation=row.orientation,
                                              evaluation=i+1, distance=float(distance), run_id=row.run_id, identity_run_id=base.run_id))
            table('candidate_path_distances', pd.DataFrame(paths))
            paths = pd.DataFrame(paths)
            table('final_path_summary', paths[paths.evaluation.eq(100)].groupby('method').agg(
                paired_paths=('distance', 'size'), median_distance=('distance', 'median'),
                max_distance=('distance', 'max')).reset_index())
            fig, ax = plt.subplots(figsize=(9, 4))
            for method, values in paths.groupby('method'):
                # Give each design equal weight after averaging its two rotations.
                grouped = values.groupby(['initial_design_seed', 'evaluation']).distance.mean()
                median = grouped.groupby('evaluation').median()
                ax.plot(median.index, np.maximum(median.values, 1e-16), label=method, color=COLORS[method])
            ax.axvline(20, color='black', ls='--', lw=1)
            ax.set(yscale='log', xlabel='Counted calls', ylabel='Mapped candidate distance',
                   title='Battery ball: median of within-design rotation averages; floor 1e-16')
            ax.legend()
            figure('sequential_path_matching', fig)
            timing = primary.copy()
            timing['overhead_s'] = timing.search_s - timing.objective_s
            table('primary_quality_and_timing', timing.groupby('method').agg(
                runs=('run_id', 'size'), design_groups=('initial_design_seed', 'nunique'),
                median_rmse_v=('calibration_rmse_v', 'median'), median_search_s=('search_s', 'median'),
                median_objective_s=('objective_s', 'median'), median_overhead_s=('overhead_s', 'median')).reset_index())

        # Post-confirmation review: additional summaries of the frozen v3 runs.
        # Pair budgets within trajectories, then bootstrap independent designs.
        cma = usable[usable.method.str.startswith('cma') & usable.objective.eq('battery')].copy()
        if len(cma):
            cma['log10_loss'] = np.log10(cma.loss)
            changes = paired_budget_changes(cma, ['block', 'method', 'handler', 'orientation',
                'initial_design_seed', 'search_replicate', 'run_id'], 'log10_loss')
            table('cma_budget_changes', changes)
            table('cma_budget_change_intervals', group_intervals(changes,
                ['block', 'method', 'handler', 'orientation'], 'change_300_minus_100', config))
            table('cma_checkpoint_quality', cma.groupby(
                ['block', 'method', 'handler', 'orientation', 'calls']).agg(
                runs=('run_id', 'size'), median_rmse_v=('calibration_rmse_v', 'median')).reset_index())
        if len(boundary):
            changes = paired_budget_changes(boundary,
                ['method', 'orientation', 'initial_design_seed', 'search_replicate'], 'interaction')
            table('boundary_budget_changes', changes)
            intervals = group_intervals(changes, ['method', 'orientation'], 'change_300_minus_100', config)
            intervals['analysis_status'] = 'post-confirmation exploratory budget contrast'
            table('boundary_budget_change_intervals', intervals)
        if stochastic:
            variability = pd.DataFrame(stochastic)
            variability['absolute_log10_ratio'] = variability.log10_loss_ratio.abs()
            descriptive = []
            summary_keys = ['block', 'method', 'handler', 'calls', 'initial_design_seed']
            identity_spread = variability.groupby(summary_keys).absolute_log10_ratio.mean().reset_index()
            identity_spread['comparison'] = 'I vs I (independent searches)'
            descriptive.append(identity_spread)
            rotations = pairs[pairs.method.str.startswith('cma') & pairs.objective.eq('battery')]
            for orientation, values in rotations.groupby('orientation'):
                spread = values.groupby(summary_keys).absolute_log10_ratio.mean().reset_index()
                spread['comparison'] = f'Q{int(orientation)} vs I'
                descriptive.append(spread)
            descriptive = pd.concat(descriptive, ignore_index=True)
            table('cma_variability_by_design', descriptive)
            table('cma_variability_summary', descriptive.groupby(
                ['block', 'method', 'handler', 'calls', 'comparison']).agg(
                design_groups=('initial_design_seed', 'nunique'),
                mean_absolute_log10_ratio=('absolute_log10_ratio', 'mean')).reset_index())
            ball = descriptive[descriptive.block.eq('battery_ball')]
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
            labels = ['I vs I (independent searches)', *[f'Q{q} vs I' for q in config['rotations'][1:]]]
            for row, method in enumerate(['cma_full', 'cma_diag']):
                for col, calls in enumerate([100, 300]):
                    ax = axes[row, col]
                    selected = ball[ball.method.eq(method) & ball.calls.eq(calls)]
                    for position, label in enumerate(labels):
                        values = selected[selected.comparison.eq(label)].absolute_log10_ratio
                        ax.scatter(np.full(len(values), position), values, color=COLORS[method], alpha=.55)
                        ax.plot([position-.18, position+.18], [values.mean()]*2, color='black')
                    ax.set(xticks=range(len(labels)), xticklabels=['I vs I', *[f'Q{q} vs I' for q in config['rotations'][1:]]],
                           ylabel='Within-design mean absolute log10 loss ratio', title=f'{method}, {calls} calls')
            fig.suptitle('Battery ball: dots are 10 designs; bars are means\nDescriptive variability comparison, not an equivalence test')
            figure('cma_rotation_and_variability', fig)
    audit = dict(generated_utc=now(), execution='agent',
        analysis_revision='v3-review-2026-11-11',
        added_analysis_scope='Post-confirmation paired budget contrasts and descriptive summaries; no new optimisation runs',
        development_complete=dev_path.exists(), confirmation_started=metadata is not None,
        planned_runs=len(metadata['jobs']) if metadata else 885,
        complete_runs=sum(r['status'] == 'complete' for r in metadata['rows']) if metadata else 0,
        completed_run_search_calls=sum(r['evaluations'] for r in metadata['rows'] if r['status'] == 'complete') if metadata else 0,
        all_recorded_search_attempts=sum(r['evaluations'] for r in metadata['rows']) if metadata else 0,
        assessment_calls=sum(r['assessment_calls'] for r in metadata['rows']) if metadata else 0,
        status='complete' if metadata and len(metadata['rows']) == len(metadata['jobs']) and all(r['status'] == 'complete' for r in metadata['rows']) else 'partial',
        collection_sha256=sha256(root / 'confirmation/collection.json') if metadata else None,
        analysis_sha256=sha256(Path(__file__)),
        outputs={name: sha256(output / name) for name in written + figures},
        interpretation='Only complete planned design groups enter contrasts; partial/failed groups remain in coverage and run-status tables. No student understanding inferred.')
    atomic_dump(output / 'validation.json', audit)
    return audit
