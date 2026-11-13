import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from gpytorch.kernels import MaternKernel
from scipy.optimize import least_squares

from bo_vs_es import ROOT
from bo_vs_es.objectives.data import parse_workbook, dynamic_trace, ocv_curve, COLUMNS, sha256
from bo_vs_es.objectives.models import Domain, rotation, physical, dimensionless, predict, rc_voltage, Quadratic, Battery
from bo_vs_es.experiments.runner import Evaluator, valid_completed, jobs_for
from bo_vs_es.optimisers.adapters import make_cma, covariance


def test_cma_covariance_and_seeded_reproducibility():
    settings = {"population": 8, "sigma0": .3}
    q = rotation(101)
    for diagonal in [False, True]:
        es = make_cma(np.zeros(5), 123, diagonal, settings)
        initial = covariance(es)
        assert np.allclose(initial, .09 * np.eye(5), atol=1e-15, rtol=0)
        if diagonal:
            assert es.sp.c1_sep == es.sp.c1 and es.sp.cmu_sep == es.sp.cmu
            assert es.sp.cc_sep == es.sp.cc
        first = np.array(es.ask())
        es.tell(first, np.sum(first ** 2 * np.arange(1, 6), axis=1))
        cov = covariance(es)
        assert np.linalg.eigvalsh(cov).min() > 0
        assert np.allclose(np.linalg.eigvalsh(q @ cov @ q.T), np.linalg.eigvalsh(cov))
        if diagonal:
            assert np.allclose(cov, np.diag(np.diag(cov)))
        repeat = make_cma(np.zeros(5), 123, diagonal, settings)
        assert np.array_equal(first, np.array(repeat.ask()))


def test_rotations_domains_and_curvature():
    q = rotation(101)
    assert np.allclose(q.T @ q, np.eye(5), atol=1e-14)
    assert np.isclose(np.linalg.det(q), 1)
    f = Quadratic(10000)
    assert np.allclose(np.linalg.eigvalsh(q.T @ f.hessian @ q), np.linalg.eigvalsh(f.hessian))
    for kind in ['ball', 'box']:
        domain = Domain(kind, q)
        base = Domain(kind, np.eye(5)).sample(np.random.default_rng(12), 20)
        rotated = domain.sample(np.random.default_rng(12), 20)
        assert np.allclose(base, domain.to_base(rotated), atol=1e-14)
        for u, z in zip(base, rotated):
            assert domain.feasible(z)
            assert np.allclose(u, dimensionless(physical(u)))
            assert np.isclose(f(u), f(domain.to_base(z)), atol=1e-14)
            assert domain.feasible(domain.project(z * 10))
        # Projection is covariant, not clipping in the rotated coordinate box.
        z = np.array([3, -2, 1, -1, 2.])
        expected = Domain(kind, np.eye(5)).project(domain.to_base(z))
        assert np.allclose(domain.to_base(domain.project(z)), expected)


def test_kernel_rotation_control():
    rng = np.random.default_rng(8)
    x = torch.tensor(rng.normal(size=(12, 5)), dtype=torch.double)
    xq = x @ torch.tensor(rotation(101), dtype=torch.double)
    iso = MaternKernel(nu=2.5).double()
    assert torch.allclose(iso(x).to_dense(), iso(xq).to_dense(), atol=1e-12)
    ard = MaternKernel(nu=2.5, ard_num_dims=5).double()
    ard.lengthscale = torch.tensor([.2, .4, .8, 1.6, 3.2])
    assert not torch.allclose(ard(x).to_dense(), ard(xq).to_dense())


def test_rc_analytic_and_identifiable_recovery():
    t = np.r_[0., np.cumsum(np.linspace(.2, 1.2, 200))]
    current = np.ones(len(t)) * 2
    expected = .06 * (1 - np.exp(-t / 12))
    assert np.allclose(rc_voltage(t, current, .03, 12), expected, atol=1e-13)
    # Distinct time constants and a persistently changing input avoid branch ambiguity locally.
    t = np.arange(600.)
    current = np.random.default_rng(8).choice([-1., 0., 1., 2.], size=30).repeat(20)
    theta = np.array([.04, .02, 5., .03, 70.])
    measured = predict(t, current, np.full(len(t), 3.8), theta)
    result = least_squares(lambda logtheta: predict(t, current, np.full(len(t), 3.8), np.exp(logtheta)) - measured,
                           np.log(theta * [1.1, .9, 1.15, 1.1, .9]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    assert np.allclose(np.exp(result.x), theta, rtol=1e-4)


def test_workbook_schema_overlap_and_conflicts():
    rows = pd.DataFrame({name: np.zeros(3) for name in COLUMNS})
    rows['Data_Point'] = [1, 2, 3]
    rows['Test_Time(s)'] = [0., 1., 2.]
    rows['Voltage(V)'] = [4., 3.9, 3.8]

    def workbook(conflict=False):
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            rows.iloc[:2].to_excel(writer, sheet_name='Channel_1', index=False)
            second = rows.iloc[1:].copy()
            if conflict:
                second.iloc[0, second.columns.get_loc('Voltage(V)')] = 5.
            second.to_excel(writer, sheet_name='Channel_2', index=False)
        return stream.getvalue()

    frame, audit = parse_workbook(workbook())
    assert len(frame) == 3 and audit['overlapping_records_removed'] == 1
    with pytest.raises(ValueError, match='Conflicting'):
        parse_workbook(workbook(True))
    with pytest.raises(ValueError, match='OOXML'):
        parse_workbook(b'not an Excel workbook')


def test_dynamic_selection_and_sign():
    frame = pd.DataFrame({'time_s': [0., 10., 20., 21., 22., 23.], 'step': [2, 6, 6, 7, 8, 7],
                          'current_charge_a': [1., 0., 0., -2., 0., -1.], 'voltage_v': [4.] * 6})
    trace = dynamic_trace(frame, .8, duration_s=3)
    assert trace.step.tolist() == [6, 7, 8, 7]
    assert trace.current_a.tolist() == [0., 2., 0., 1.]
    assert trace.soc.iloc[2] == pytest.approx(.8 - 2 / 7200)


def test_ocv_branch_mapping():
    n = 101
    d = pd.DataFrame({'step': 6, 'current_charge_a': -.1,
                      'discharge_ah': np.linspace(0, 2, n), 'charge_ah': 0.,
                      'voltage_v': np.linspace(4.1, 3.1, n)})
    c = pd.DataFrame({'step': 8, 'current_charge_a': .1,
                      'charge_ah': np.linspace(0, 2, n), 'discharge_ah': 2.,
                      'voltage_v': np.linspace(3.3, 4.3, n)})
    curve, spans = ocv_curve(pd.concat([d, c]))
    assert np.allclose(curve.ocv_v, 3.2 + curve.soc)
    assert spans['charge_ah'] == 2.


def test_budget_failed_call_and_invalid_proposal(tmp_path):
    domain = Domain('ball', np.eye(5))
    evaluator = Evaluator(Quadratic(), domain, 2, tmp_path)
    with pytest.raises(ValueError):
        evaluator.evaluate(np.ones(5))
    assert evaluator.remaining == 2
    evaluator.evaluate(np.zeros(5))
    evaluator.objective = lambda u: np.nan
    with pytest.raises(RuntimeError, match='failed'):
        evaluator.evaluate(np.zeros(5))
    assert evaluator.remaining == 0 and evaluator.rows[-1]['status'] == 'failed'
    with pytest.raises(RuntimeError, match='exhausted'):
        evaluator.evaluate(np.zeros(5))


def test_completion_hash_guard(tmp_path):
    pd.DataFrame({'loss': [2., 1.], 'status': ['ok', 'ok']}).to_csv(tmp_path/'evaluations.csv', index=False)
    (tmp_path/'summary.json').write_text(json.dumps({'evaluations': 2, 'budget': 2, 'best_loss': 1.}))
    hashes = {p.name: sha256(p) for p in tmp_path.iterdir()}
    (tmp_path/'manifest.json').write_text(json.dumps({'status': 'complete', 'run_key': 'test', 'output_sha256': hashes}))
    assert valid_completed(tmp_path, 'test')
    (tmp_path/'evaluations.csv').write_text('changed')
    assert not valid_completed(tmp_path, 'test')


def test_configs_use_complete_populations():
    for path in (ROOT/'configs/experiments').glob('*.json'):
        recipe = json.loads(path.read_text())
        if recipe.get('id') == 'rotation-followup-v4':
            from bo_vs_es.followup_v4 import jobs_for as followup_jobs
            jobs = followup_jobs(recipe)
        elif recipe.get('id') == 'rotation-followup-v3':
            from bo_vs_es.followup import jobs_for as followup_jobs
            jobs = followup_jobs(recipe)
        else:
            jobs = jobs_for(recipe)
        for job in jobs:
            assert (job['budget'] - job['n_init']) % 8 == 0
    assert len(jobs_for(json.loads((ROOT/'configs/experiments/pilot.json').read_text()))) == 84


def test_measured_data_and_training_only_offset():
    if not (ROOT/'data/processed/manifest.json').exists():
        pytest.skip('Download and prepare CALCE data for the measured-data integration check')
    train = Battery('dst80')
    test = Battery('fuds80')
    assert train.time[0] == 0 and train.time[-1] <= 3600
    assert train.trace.soc.between(0, 1).all()
    assert train.manifest['calibration_rest_offset_v'] == test.manifest['calibration_rest_offset_v']
    point = np.zeros(5)
    before = train(point)
    test.measured = np.full_like(test.measured, 1e6)
    assert train(point) == before


def test_bo_seeded_candidate_and_exact_box_constraints():
    from bo_vs_es.optimisers.adapters import bo_candidate
    domain = Domain('box', rotation(101))
    initial = domain.sample(np.random.default_rng(2), 20)
    losses = [Quadratic(100)(domain.to_base(z)) for z in initial]
    settings = dict(fit_maxiter=10, acq_maxiter=15, raw_samples=24, num_restarts=2)
    torch.manual_seed(321)
    a, da = bo_candidate(initial, losses, domain, False, settings, np.random.default_rng(321))
    torch.manual_seed(321)
    b, db = bo_candidate(initial, losses, domain, False, settings, np.random.default_rng(321))
    assert domain.feasible(a) and domain.feasible(b)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    assert len(da['lengthscales']) == 1
    assert da['fit_status'] == db['fit_status']


def test_identical_branches_and_invalid_soc():
    from bo_vs_es.objectives.models import predict
    t = np.arange(50.)
    current = np.sin(t)
    ocv = np.full(len(t), 3.8)
    theta = np.array([.05, .02, 10., .03, 40.])
    np.testing.assert_allclose(predict(t,current,ocv,theta),predict(t,current,ocv,theta[[0,3,4,1,2]]),atol=1e-14)
    # When time constants coincide, only the sum of branch resistances is observed.
    a = [.05, .02, 10., .03, 10.]
    b = [.05, .01, 10., .04, 10.]
    np.testing.assert_allclose(predict(t,current,ocv,a),predict(t,current,ocv,b),atol=1e-14)
    raw = pd.DataFrame({'time_s':[0.,1.,3600.], 'step':[6,7,7],
                        'current_charge_a':[0.,-10.,-10.], 'voltage_v':[4.,3.8,3.5]})
    with pytest.raises(ValueError,match='SOC leaves'):
        dynamic_trace(raw,.8)


def test_runner_assesses_only_after_search_and_verified_resume(tmp_path, monkeypatch):
    import bo_vs_es.experiments.runner as runner
    labels = []
    class RecordedObjective:
        def __init__(self,label='dst80',corrected=True):
            self.label=label
        def __call__(self,u):
            labels.append(self.label)
            return float(np.sum(np.asarray(u)**2)+.01)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'Battery',RecordedObjective)
    job=dict(block='test',objective='battery',condition=None,domain='ball',rotation_seed=101,
             seed=2,method='random',handler='none',budget=28,n_init=20,corrected=True)
    result,cached=runner.execute_job(job,{},tmp_path/'runs',{})
    assert not cached and result['status']=='complete'
    assert labels == ['dst80']*28+['fuds80','fuds50']
    repeated,cached=runner.execute_job(job,{},tmp_path/'runs',{})
    assert cached and repeated['run_id']==result['run_id']
    assert len(labels)==30


def test_uncertainty_uses_seeds_not_rotation_rows():
    from bo_vs_es.analysis.study import effect_summary,sensitivity_contrasts
    pairs=pd.DataFrame([dict(method=method,seed=seed,rotation_seed=rot,log10_loss_ratio=value)
                        for method,value in [('bo_iso',.2),('bo_ard',-.1),('cma_full',.1),('cma_diag',.3)]
                        for seed in [0,1] for rot in [101,202,303]])
    summary=effect_summary(pairs)
    assert summary.seeds.eq(2).all()
    contrasts=sensitivity_contrasts(pairs)
    assert contrasts.seeds.eq(2).all()
    np.testing.assert_allclose(contrasts.mean_absolute_effect_difference,[-.1,.2],atol=1e-14)
