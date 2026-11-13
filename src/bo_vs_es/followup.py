"""Versioned follow-up; the fingerprinted v1 implementation remains unchanged."""
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.analytic import LogExpectedImprovement

from bo_vs_es import ROOT
from bo_vs_es.diagnostics.core import uniform_ball, validate_config
from bo_vs_es.diagnostics.replay import scalar_model, fit_model, replay_history
from bo_vs_es.experiments.runner import Evaluator, dump, valid_completed
from bo_vs_es.objectives.data import sha256
from bo_vs_es.objectives.models import Battery, Quadratic, Domain, rotation, physical
from bo_vs_es.optimisers.adapters import bo_candidate, make_cma, covariance


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    dump(temporary, value)
    temporary.replace(path)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Study compute allowance reached; partial attempt retained')


def load_config(path=None):
    config = json.loads(Path(path or ROOT / 'configs/experiments/followup_v3.json').read_text())
    development = json.loads((ROOT / config['development_protocol']).read_text())
    validate_config(development)
    if (set(config['confirmation_seeds']) != set(development['reserved_confirmation']['initial_design_seeds'])
            or set(config['rotations']) != {None, *development['reserved_confirmation']['rotation_seeds']}):
        raise ValueError('Confirmation must use the reserved seeds and orientations')
    if not set(config['quadratic_seeds']) <= set(config['confirmation_seeds']):
        raise ValueError('Quadratic cases must be reserved confirmation designs')
    for key in ['confirmation_seeds', 'rotations', 'cma_search_replicates', 'quadratic_seeds', 'conditions']:
        if not config[key] or len(config[key]) != len(set(config[key])):
            raise ValueError(f'Empty or duplicate {key}')
    if config['n_init'] != 20 or config['primary_calls'] != 100 or config['cma_extension_calls'] != 300:
        raise ValueError('V3 freezes 20 initial observations and checkpoints 100/300')
    if any(config['pool'][k] <= 0 for k in ['interior', 'surface', 'chunk_size']):
        raise ValueError('Pool sizes must be positive')
    return config


def search_seed(config, initial_seed, replicate):
    # Streams depend on design/replicate, never on orientation, method or handler.
    return int(np.random.SeedSequence([config['search_seed_namespace'], initial_seed, replicate])
               .generate_state(1)[0] % (2**31 - 2)) + 1


def jobs_for(config):
    jobs = []
    for block in ['battery_ball', 'battery_boundary', 'quadratic_control']:
        seeds = config['quadratic_seeds'] if block == 'quadratic_control' else config['confirmation_seeds']
        for seed in seeds:  # Finish every orientation/method in a design group first.
            for condition in config['conditions'] if block == 'quadratic_control' else [None]:
                for rot in config['rotations']:
                    methods = ['cma_full', 'cma_diag'] if block == 'battery_boundary' else [
                        'bo_iso', 'bo_ard', 'cma_full', 'cma_diag', 'random']
                    if block == 'battery_ball':
                        methods = [*methods, 'bo_iso_pool']
                    for method in methods:
                        cma = method.startswith('cma')
                        replicates = config['cma_search_replicates'] if cma and block != 'quadratic_control' else [0]
                        for replicate in replicates:
                            for handler in ['resample', 'project'] if block == 'battery_boundary' else ['resample' if cma else 'none']:
                                budget = config['cma_extension_calls'] if cma and block != 'quadratic_control' else config['primary_calls']
                                jobs.append(dict(block=block, objective='quadratic' if condition else 'battery',
                                    domain='box' if block == 'battery_boundary' else 'ball', condition=condition,
                                    initial_design_seed=seed, search_replicate=replicate,
                                    search_seed=search_seed(config, seed, replicate), rotation_seed=rot,
                                    method=method, handler=handler, budget=budget, n_init=config['n_init'],
                                    checkpoints=sorted({config['primary_calls'], budget}), corrected=True))
    return jobs


def candidate_pool(rng, settings):
    inside = uniform_ball(rng, settings['interior'], 5)
    surface = rng.normal(size=(settings['surface'], 5))
    surface /= np.linalg.norm(surface, axis=1, keepdims=True)
    return np.vstack([inside, surface])


def acquisition_scores(model, points, chunk_size=256):
    model.eval()
    acq = LogExpectedImprovement(model, best_f=model.train_targets.max())
    with torch.no_grad():
        chunks = [acq(torch.as_tensor(points[i:i + chunk_size], dtype=torch.double).unsqueeze(1))
                  .detach().numpy() for i in range(0, len(points), chunk_size)]
    scores = np.concatenate(chunks)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite finite-pool acquisition values')
    return scores


def pool_candidate(points, losses, domain, settings, rng, pool_settings):
    torch.set_num_threads(1)
    model = scalar_model(points, losses)
    fit = fit_model(model, settings)
    start = time.perf_counter()
    pool = candidate_pool(rng, pool_settings)
    scores = acquisition_scores(model, domain.from_base(pool), pool_settings['chunk_size'])
    index = int(np.argmax(scores))  # First index breaks exact ties.
    top = np.sort(scores)[-2:]
    return domain.project(domain.from_base(pool[index])), dict(
        fit_s=fit['seconds'], acquisition_s=time.perf_counter() - start,
        fit_status=fit['status'], fit_message=fit['message'], warnings=fit['warnings'],
        lengthscales=[fit['lengthscale']], outputscale=fit['outputscale'],
        log_ei=float(scores[index]), pool_index=index, pool_top_gap=float(top[-1] - top[-2]),
        pool_sha256=hashlib.sha256(pool.tobytes()).hexdigest())


def provenance(config, environment):
    paths = sorted((ROOT / 'src/bo_vs_es').rglob('*.py')) + [ROOT / 'scripts/run_followup.py']
    return dict(config=config, environment=environment,
                source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
                data_manifest=json.loads((ROOT / 'data/processed/manifest.json').read_text()),
                dependency_lock_sha256=sha256(ROOT / 'requirements-lock.txt'),
                hardware=dict(machine=platform.machine(), platform=platform.platform()),
                execution='agent', torch_threads=1)


def execute_job(job, config, protocol, collection, provenance_record, deadline=float('inf'), objective=None):
    resolved = dict(job=job, protocol=protocol, provenance=provenance_record)
    key = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    folder = collection / key[:16]
    retry = 0
    while folder.exists():
        if valid_completed(folder, key):
            return json.loads((folder / 'summary.json').read_text()), True
        retry += 1
        folder = collection / f'{key[:16]}-retry{retry}'
    check_deadline(deadline)
    folder.mkdir(parents=True)
    manifest = dict(run_key=key, run_id=folder.name, status='running', started_utc=now(),
                    source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
                    **provenance_record)
    dump(folder / 'manifest.json', manifest)
    dump(folder / 'config.json', resolved)
    domain = Domain(job['domain'], rotation(job['rotation_seed']))
    summary = dict(job, run_id=folder.name, run_path=str(folder.relative_to(ROOT)), status='failed',
                   assessment_calls=0, assessment_s=0., checkpoints=[])
    evaluator = None
    start = time.perf_counter()
    try:
        objective = objective or (Quadratic(job['condition']) if job['objective'] == 'quadratic' else Battery())
        evaluator = Evaluator(objective, domain, job['budget'], folder, job['objective'] == 'battery')
        initial = domain.sample(np.random.default_rng(job['initial_design_seed']), job['n_init'])
        dump(folder / 'initial.json', dict(z=initial.tolist(), u=domain.to_base(initial).tolist(), Q=domain.q.tolist()))
        rng = np.random.default_rng(job['search_seed'])
        torch.manual_seed(job['search_seed'])
        for point in initial:
            check_deadline(deadline)
            evaluator.evaluate(point, 'initial')
        cma = job['method'].startswith('cma')
        es = make_cma(evaluator.z[np.argmin(evaluator.losses)], job['search_seed'],
                      job['method'] == 'cma_diag', protocol['cma']) if cma else None
        while evaluator.remaining:
            check_deadline(deadline)
            if cma:
                population = protocol['cma']['population']
                if evaluator.remaining < population:
                    raise ValueError('Incomplete CMA generation in declared budget')
                points, losses = [], []
                for _ in range(population):
                    for _ in range(protocol['cma']['max_proposals_per_point']):
                        check_deadline(deadline)
                        proposed = np.asarray(es.ask(1)[0])
                        point = domain.project(proposed) if job['handler'] == 'project' else proposed
                        if domain.feasible(point):
                            evaluator.proposal(proposed, point)
                            points.append(point)
                            losses.append(evaluator.evaluate(point, proposed=proposed))
                            break
                        evaluator.proposal(proposed)
                    else:
                        raise RuntimeError('CMA feasible resampling limit reached')
                es.tell(points, losses)
                cov = covariance(es)
                evaluator.diagnostic(dict(generation=es.countiter, sigma=float(es.sigma),
                    mean_z=es.mean.tolist(), covariance_z=cov.tolist(),
                    covariance_u=(domain.q @ cov @ domain.q.T).tolist(), eigenvalues=np.linalg.eigvalsh(cov).tolist()))
            else:
                if job['method'] == 'random':
                    point = domain.sample(rng, 1)[0]
                else:
                    if job['method'] == 'bo_iso_pool':
                        point, diagnostic = pool_candidate(evaluator.z, evaluator.losses, domain,
                            protocol['bo'], rng, config['pool'])
                    else:
                        point, diagnostic = bo_candidate(evaluator.z, evaluator.losses, domain,
                            job['method'] == 'bo_ard', protocol['bo'], rng)
                    evaluator.diagnostic(diagnostic)
                check_deadline(deadline)
                evaluator.proposal(point, point)
                evaluator.evaluate(point)
            count = len(evaluator.rows)
            if count in job['checkpoints']:
                best = int(np.argmin(evaluator.losses))
                u = domain.to_base(evaluator.z[best])
                checkpoint = dict(calls=count, evaluation=best + 1, loss=evaluator.losses[best],
                    u=u.tolist(), z=evaluator.z[best].tolist(), search_s=time.perf_counter() - evaluator.start,
                    objective_s=sum(r['objective_s'] for r in evaluator.rows),
                    proposals=evaluator.proposals, rejected=evaluator.rejected, repaired=evaluator.repaired)
                summary['checkpoints'].append(checkpoint)
                dump(folder / 'checkpoints.json', summary['checkpoints'])
        summary.update(status='complete', best_loss=min(evaluator.losses), evaluations=len(evaluator.rows),
                       search_s=time.perf_counter() - evaluator.start, proposals=evaluator.proposals,
                       rejected=evaluator.rejected, repaired=evaluator.repaired,
                       objective_s=sum(r['objective_s'] for r in evaluator.rows))
        # Assessment is after all search, including the extension, and never feeds back.
        assessment_start = time.perf_counter()
        for checkpoint in summary['checkpoints']:
            if job['objective'] == 'battery':
                checkpoint['theta'] = physical(checkpoint['u']).tolist()
                for label in ['fuds80', 'fuds50']:
                    check_deadline(deadline)
                    summary['assessment_calls'] += 1
                    checkpoint[label + '_rmse_v'] = float(np.sqrt(Battery(label)(checkpoint['u'])))
        summary['assessment_s'] = time.perf_counter() - assessment_start
        dump(folder / 'checkpoints.json', summary['checkpoints'])
        dump(folder / 'recommendation.json', dict(rule='lowest observed deterministic calibration loss',
                                                **summary['checkpoints'][-1]))
    except Exception as exc:
        summary.update(status='budget_limited' if isinstance(exc, TimeoutError) else 'failed', error=repr(exc))
        (folder / 'error.txt').write_text(traceback.format_exc())
    finally:
        summary['evaluations'] = len(evaluator.rows) if evaluator else 0
        summary['attempt_s'] = time.perf_counter() - start
        dump(folder / 'summary.json', summary)
        manifest.update(status=summary['status'], finished_utc=now(),
                        output_sha256={p.name: sha256(p) for p in folder.iterdir() if p.name != 'manifest.json'})
        dump(folder / 'manifest.json', manifest)
    return summary, False


def development(config, folder, v2_folder, deadline):
    """Replay original pilot histories, plus a reference diagnostic. No v1 edits."""
    from bo_vs_es.analysis.study import load_results
    config_v2 = json.loads((ROOT / config['development_protocol']).read_text())
    protocol = json.loads((ROOT / config['v1_protocol']).read_text())
    v2_manifest = json.loads((v2_folder / 'manifest.json').read_text())
    if v2_manifest['status'] != 'complete':
        raise ValueError('Complete v2 diagnostics before original-history follow-up')
    for name, digest in v2_manifest['output_sha256'].items():
        if sha256(v2_folder / name) != digest:
            raise ValueError('Modified v2 diagnostic artifact')
    results = load_results()
    rows = []
    for seed in config_v2['initial_design_seeds']:
        source = results[(results.block == 'battery_ball') & (results.method == 'bo_iso') &
                         results.seed.eq(seed) & results.rotation_seed.isna()].iloc[0]
        path = ROOT / source.run_path / 'evaluations.csv'
        trace = pd.read_csv(path)
        for n in config_v2['checkpoints']:
            check_deadline(deadline)
            points = np.array([json.loads(x) for x in trace.u.iloc[:n]])
            losses = trace.loss.iloc[:n].to_numpy()
            replay = replay_history(points, losses, 101, config_v2, protocol['bo'], deadline)
            matrices = [np.eye(5), rotation(101)]
            models = [scalar_model(points @ q, losses) for q in matrices]
            for model in models:
                check_deadline(deadline)
                fit_model(model, protocol['bo'])
            pool = candidate_pool(np.random.default_rng(config_v2['acquisition_start_seed']), config['pool'])
            checks = {}
            for mode in ['independent_fits', 'common_fit']:
                if mode == 'common_fit':
                    state = {k: v.detach().clone() for k, v in models[0].state_dict().items()}
                    models = [scalar_model(points @ q, losses) for q in matrices]
                    for model in models:
                        model.load_state_dict(state)
                scores = [acquisition_scores(m, pool @ q, config['pool']['chunk_size']) for m, q in zip(models, matrices)]
                indices = [int(np.argmax(s)) for s in scores]
                checks[mode] = dict(indices=indices, candidate_distance=float(np.linalg.norm(pool[indices[0]] - pool[indices[1]])),
                    max_score_difference=float(np.max(np.abs(scores[0] - scores[1]))),
                    common_model_log_ei=[float(scores[0][i]) for i in indices],
                    top_gaps=[float(np.sort(s)[-1] - np.sort(s)[-2]) for s in scores])
            replay.update(source_run_id=source.run_id, source_trace_sha256=sha256(path),
                          history_seed=seed, history_origin='original saved v1 pilot', finite_pool=checks)
            rows.append(replay)
            dump(folder / 'original_history_replays.json', rows)
            print(f'Original-history replay: seed {seed}, observations {n}', flush=True)
    reference = json.loads((v2_folder / 'ball_reference.json').read_text())
    u = np.asarray(reference['best']['u'])
    if np.linalg.norm(u) > 1:
        raise ValueError('Reference must be feasible')
    objective = Battery()
    columns, callbacks = [], []
    for j in range(5):
        lo, hi = u.copy(), u.copy()
        lo[j], hi[j] = max(-1., lo[j] - 1e-4), min(1., hi[j] + 1e-4)
        values = []
        for point in [lo, hi]:
            check_deadline(deadline)
            record = dict(u=point.tolist(), feasible_ball=bool(np.linalg.norm(point) <= 1), status='attempted')
            callbacks.append(record)
            dump(folder / 'reference_callbacks.json', callbacks)
            values.append(objective.residual(point))
            record['status'] = 'ok'
        columns.append((values[1] - values[0]) / (hi[j] - lo[j]))
    jacobian = np.column_stack(columns)
    hessian = 2 * jacobian.T @ jacobian / len(objective.time)
    assessments = {}
    for label in ['fuds80', 'fuds50']:
        check_deadline(deadline)
        callbacks.append(dict(assessment=label, u=u.tolist(), status='attempted'))
        dump(folder / 'reference_callbacks.json', callbacks)
        assessments[label + '_rmse_v'] = float(np.sqrt(Battery(label)(u)))
        callbacks[-1]['status'] = 'ok'
    dump(folder / 'reference_callbacks.json', callbacks)
    dump(folder / 'ball_reference_diagnostics.json', dict(
        source=str(v2_folder.relative_to(ROOT)), source_sha256=sha256(v2_folder / 'ball_reference.json'),
        reference=reference['best'], terminations=reference['terminations'], assessments=assessments,
        hessian=hessian.tolist(), eigenvalues=np.linalg.eigvalsh(hessian).tolist(),
        extra_simulator_calls=len(callbacks), difference_step=1e-4,
        interpretation='Ambient local Gauss-Newton curvature at a feasible ball point; derivative trials may leave the ball; not global or constrained tangent curvature'))
    return dict(replays=len(rows), extra_simulator_calls=len(callbacks), v2_source=str(v2_folder.relative_to(ROOT)))
