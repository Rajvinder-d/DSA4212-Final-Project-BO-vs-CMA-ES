"""Frozen V4 science; V1–V3 modules are imported, never modified."""
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll_scipy
from botorch.models import SingleTaskGP
from botorch.settings import validate_input_scaling
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
import warnings

from bo_vs_es import ROOT
from bo_vs_es.experiments.runner import Evaluator, dump, valid_completed
from bo_vs_es.followup import atomic_dump, now, check_deadline, candidate_pool, acquisition_scores
from bo_vs_es.objectives.data import sha256
from bo_vs_es.objectives.models import Battery, Quadratic, Domain, rotation, physical
from bo_vs_es.optimisers.adapters import bo_candidate, make_cma, covariance
from bo_vs_es.diagnostics.replay import predictions, differences

BO_METHODS = ['bo_iso', 'bo_ard', 'bo_iso_pool', 'bo_ard_pool']


def load_config(path=None):
    path = Path(path or ROOT / 'configs/experiments/followup_v4.json')
    config = json.loads(path.read_text())
    # V4 is a frozen experiment, not a generic configurable benchmark.
    expected = dict(id='rotation-followup-v4', confirmation_seeds=list(range(20, 40)),
        supporting_seeds=list(range(20, 30)), development_seeds=[0, 1],
        rotations=[None, 404, 505, 606], development_rotation=101, namespace=4107,
        n_init=20, primary_calls=100, cma_calls=300, cma_search_replicates=list(range(5)),
        condition=10000, lengthscale_lower=.03, lengthscale_upper=5., alternate_upper=20.,
        pool=dict(interior=2048, surface=2048, chunk_size=256), replay_checkpoints=[20, 40, 80],
        replay_query_seed=4930, replay_pool_seed=4941, bootstrap_resamples=5000,
        bootstrap_seed=4129, battery_targets_rmse_v=[.006, .01], max_compute_seconds=14400,
        timing_headroom=1.25, protocol='configs/protocols/rotation_battery_v1.json')
    if config != expected:
        raise ValueError('V4 settings changed: version the protocol and study ID')
    return config


def stream(config, design, replicate=0, step=0, kind=0):
    """0 initial, 1 CMA/Torch, 2 starts, 3 pool, 4 random search; no treatment IDs."""
    return np.random.default_rng(np.random.SeedSequence([config['namespace'], design, replicate, step, kind]))


def search_seed(config, design, replicate):
    return int(stream(config, design, replicate, kind=1).integers(1, 2**31 - 1))


def jobs_for(config, development=False):
    jobs = []
    # Complete each block/design group before moving to the next design.
    blocks = ['battery_bo', 'cma_ball', 'cma_box', 'quadratic_bo', 'battery_bound', 'quadratic_bound']
    for block in blocks:
        cma = block.startswith('cma')
        supporting = block.startswith('quadratic') or block.endswith('bound')
        seeds = config['development_seeds'] if development else config['supporting_seeds' if supporting else 'confirmation_seeds']
        methods = ['cma_full', 'cma_diag'] if cma else (
            ['bo_iso_pool', 'bo_ard_pool'] if block.endswith('bound') else BO_METHODS + ([] if development else ['random']))
        for seed in seeds:
            for orientation in [None] if development else config['rotations']:
                for method in methods:
                    for replicate in ([0] if development or not cma else config['cma_search_replicates']):
                        for handler in (['resample', 'project'] if block == 'cma_box' else ['resample' if cma else 'none']):
                            budget = config['cma_calls'] if cma else config['primary_calls']
                            jobs.append(dict(block=block, objective='quadratic' if block.startswith('quadratic') else 'battery',
                                condition=config['condition'] if block.startswith('quadratic') else None,
                                domain='box' if block == 'cma_box' else 'ball', method=method, handler=handler,
                                initial_design_seed=seed, search_replicate=replicate,
                                search_seed=search_seed(config, seed, replicate), rotation_seed=orientation,
                                budget=budget, n_init=config['n_init'], checkpoints=sorted({100, budget}),
                                lengthscale_upper=config['alternate_upper'] if block.endswith('bound') else config['lengthscale_upper'],
                                corrected=True, stage='development' if development else 'confirmation'))
    return jobs


def model_for(points, losses, ard=False, upper=5.):
    x = torch.as_tensor(np.asarray(points), dtype=torch.double)
    values = -torch.as_tensor(losses, dtype=torch.double).unsqueeze(-1)
    y = (values - values.mean()) / values.std().clamp_min(1e-12)
    kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=x.shape[1] if ard else None,
        lengthscale_constraint=Interval(.03, upper)), outputscale_constraint=Interval(.01, 100.)).double()
    kernel.base_kernel.lengthscale = .7
    kernel.outputscale = 1.
    with validate_input_scaling(False):
        return SingleTaskGP(x, y, train_Yvar=torch.full_like(y, 1e-6), covar_module=kernel, outcome_transform=None)


def fit_model(model, settings):
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = fit_gpytorch_mll_scipy(mll, options={'maxiter': settings['fit_maxiter'], 'ftol': 1e-8})
    return dict(fit_s=time.perf_counter() - start, fit_status=str(fit.status), fit_message=str(fit.message),
        warnings=[str(w.message) for w in caught],
        lengthscales=model.covar_module.base_kernel.lengthscale.detach().numpy().ravel().tolist(),
        outputscale=float(model.covar_module.outputscale.detach()), mean=float(model.mean_module.constant.detach()))


def propose(points, losses, domain, method, upper, settings, rng, pool_settings):
    torch.set_num_threads(1)
    ard = method in ('bo_ard', 'bo_ard_pool')
    if not method.endswith('_pool'):
        if upper != 5.:
            raise ValueError('V4 continuous acquisition uses only the original upper bound')
        # Direct reuse guarantees baseline parity, including warnings/constraints.
        return bo_candidate(points, losses, domain, ard, settings, rng)
    model = model_for(points, losses, ard, upper)
    diagnostic = fit_model(model, settings)
    start = time.perf_counter()
    pool = candidate_pool(rng, pool_settings)
    scores = acquisition_scores(model, domain.from_base(pool), pool_settings['chunk_size'])
    index = int(np.argmax(scores))
    top = np.sort(scores)[-2:]
    diagnostic.update(acquisition_s=time.perf_counter() - start, log_ei=float(scores[index]),
        pool_index=index, pool_top_gap=float(top[-1] - top[-2]),
        pool_sha256=hashlib.sha256(pool.tobytes()).hexdigest())
    return domain.project(domain.from_base(pool[index])), diagnostic


def provenance(config, environment):
    folders = ['objectives', 'optimisers', 'experiments', 'diagnostics']
    paths = [ROOT / 'src/bo_vs_es/__init__.py', ROOT / 'src/bo_vs_es/followup.py',
             Path(__file__), ROOT / 'scripts/run_v4.py']
    paths += [p for folder in folders for p in (ROOT / 'src/bo_vs_es' / folder).glob('*.py')]
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    manifest.pop('generated_utc', None)
    return dict(config=config, environment=environment,
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)},
        protocol_sha256=sha256(ROOT / config['protocol']), data_manifest=manifest,
        dependency_lock_sha256=sha256(ROOT / 'requirements-lock.txt'),
        hardware=dict(machine=platform.machine(), platform=platform.platform()), execution='agent', torch_threads=1)


def verify_scientific_inputs(prov):
    for name, digest in prov['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Active scientific source changed: {name}')
    for name, digest in prov['data_manifest']['output_sha256'].items():
        if sha256(ROOT / 'data/processed' / name) != digest:
            raise ValueError(f'Prepared data changed: {name}')


def timing_key(job):
    # Random shares objective/domain with a pool run for a conservative forecast.
    method = 'bo_iso_pool' if job['method'] == 'random' else job['method']
    return (job['objective'], job['domain'], method, job['lengthscale_upper'], job['handler'])


def timing_forecast(development_rows, jobs, remaining, overhead_s=0.):
    measurements = {}
    for row in development_rows:
        if row['status'] != 'complete':
            raise ValueError('Development has incomplete/failed jobs; no confirmation gate')
        measurements.setdefault(timing_key(row), []).append(row['attempt_s'])
    estimates = {key: float(np.mean(values)) + overhead_s for key, values in measurements.items()}
    raw = sum(estimates[timing_key(j)] for j in jobs)
    return dict(raw_confirmation_s=raw, projected_confirmation_s=1.25 * raw,
        remaining_allowance_s=remaining, fits_allowance=1.25 * raw <= remaining,
        headroom=1.25, collection_overhead_per_job_s=overhead_s,
        estimates=[dict(key=list(k), seconds=v) for k, v in estimates.items()],
        policy='Full fixed grid only; gate failure starts no confirmation jobs')


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
        initial = domain.sample(stream(config, job['initial_design_seed'], kind=0), job['n_init'])
        dump(folder / 'initial.json', dict(z=initial.tolist(), u=domain.to_base(initial).tolist(), Q=domain.q.tolist()))
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
                step = len(evaluator.rows)
                kind = 4 if job['method'] == 'random' else (3 if job['method'].endswith('_pool') else 2)
                rng = stream(config, job['initial_design_seed'], job['search_replicate'], step, kind)
                if job['method'] == 'random':
                    point = domain.sample(rng, 1)[0]
                else:
                    point, diagnostic = propose(evaluator.z, evaluator.losses, domain, job['method'],
                        job['lengthscale_upper'], protocol['bo'], rng, config['pool'])
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
    except (Exception, KeyboardInterrupt) as exc:
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


def development_replays(config, rows, output, deadline):
    protocol = json.loads((ROOT / config['protocol']).read_text())
    records = []
    q = rotation(config['development_rotation'])
    matrices = [np.eye(5), q]
    queries = Domain('ball', np.eye(5)).sample(np.random.default_rng(config['replay_query_seed']), 100)
    pool = candidate_pool(np.random.default_rng(config['replay_pool_seed']), config['pool'])
    for row in rows:
        if row['method'] != 'bo_iso' or row['lengthscale_upper'] != 5.:
            continue
        source = ROOT / row['run_path'] / 'evaluations.csv'
        trace = pd.read_csv(source)
        for n in config['replay_checkpoints']:
            points = np.array([json.loads(x) for x in trace.u.iloc[:n]])
            losses = trace.loss.iloc[:n].to_numpy()
            for ard in [False, True]:
                for upper in [5., 20.]:
                    models = [model_for(points @ matrix, losses, ard, upper) for matrix in matrices]
                    fits = []
                    for model in models:
                        check_deadline(deadline)
                        fits.append(fit_model(model, protocol['bo']))
                    predicted = [predictions(m, queries @ matrix) for m, matrix in zip(models, matrices)]
                    modes = ['independent_fits'] if ard else ['independent_fits', 'common_fit']
                    state = {k: v.detach().clone() for k, v in models[0].state_dict().items()}
                    for mode in modes:
                        check_deadline(deadline)
                        if mode == 'common_fit':
                            models = [model_for(points @ matrix, losses, False, upper) for matrix in matrices]
                            for model in models:
                                model.load_state_dict(state)
                        scores = [acquisition_scores(m, pool @ matrix, config['pool']['chunk_size'])
                                  for m, matrix in zip(models, matrices)]
                        indices = [int(np.argmax(s)) for s in scores]
                        control = differences(*[predictions(m, queries @ matrix) for m, matrix in zip(models, matrices)])
                        records.append(dict(objective=row['objective'], design=row['initial_design_seed'],
                            source_run_id=row['run_id'], source_trace_sha256=sha256(source), observations=n,
                            kernel='ard' if ard else 'iso', upper=upper, mode=mode, fits=fits,
                            independent_prediction_differences=differences(*predicted), **control,
                            max_pool_score_difference=float(np.max(np.abs(scores[0] - scores[1]))),
                            pool_indices=indices, candidate_distance=float(np.linalg.norm(pool[indices[0]] - pool[indices[1]])),
                            common_identity_score_gap=float(abs(scores[0][indices[0]] - scores[0][indices[1]])),
                            extra_simulator_calls=0))
                        atomic_dump(output, records)
    return records
