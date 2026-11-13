"""V4 checks: synthetic fixtures only, never counted as scientific repetitions."""
import copy
import json
import time

import numpy as np
import pandas as pd
import pytest

from bo_vs_es import ROOT
from bo_vs_es import followup_v4 as f
from bo_vs_es.followup import pool_candidate
from bo_vs_es.followup_analysis import group_intervals, paired_budget_changes
from bo_vs_es.objectives.models import Domain, Quadratic, rotation
from bo_vs_es.optimisers.adapters import bo_candidate


def protocol():
    return json.loads((ROOT / f.load_config()['protocol']).read_text())


def test_v4_grid_accounting_and_separation():
    cfg = f.load_config()
    for dev, count, search, assess in [(False, 3160, 796000, 10560), (True, 36, 6000, 72)]:
        jobs = f.jobs_for(cfg, dev)
        assert len(jobs) == count
        assert sum(j['budget'] for j in jobs) == search
        assert sum(2*len(j['checkpoints']) for j in jobs if j['objective']=='battery') == assess
        assert len({json.dumps(j, sort_keys=True) for j in jobs}) == count
        assert {j['initial_design_seed'] for j in jobs} == set(cfg['development_seeds'] if dev else cfg['confirmation_seeds'])
    with pytest.raises(ValueError):
        path = ROOT / 'configs/experiments/followup_v3.json'
        f.load_config(path)


def test_v4_streams_match_physical_candidates_without_treatment_ids():
    cfg = f.load_config()
    a = Domain('ball', np.eye(5))
    b = Domain('ball', rotation(404))
    left = a.sample(f.stream(cfg, 20), 20)
    right = b.to_base(b.sample(f.stream(cfg, 20), 20))
    np.testing.assert_allclose(left, right, atol=1e-14)
    pool = f.candidate_pool(f.stream(cfg,20,0,20,3), cfg['pool'])
    np.testing.assert_array_equal(pool, f.candidate_pool(f.stream(cfg,20,0,20,3), cfg['pool']))
    assert not np.array_equal(pool, f.candidate_pool(f.stream(cfg,20,0,21,3), cfg['pool']))
    assert not np.array_equal(f.stream(cfg,20,0,20,2).normal(size=5), f.stream(cfg,20,0,20,3).normal(size=5))
    assert len({f.search_seed(cfg,20,r) for r in range(5)}) == 5


@pytest.mark.parametrize('ard', [False, True])
def test_v4_continuous_baseline_parity(ard):
    domain=Domain('ball', np.eye(5))
    points=domain.sample(np.random.default_rng(4),20)
    losses=[Quadratic(100)(p) for p in points]
    method='bo_ard' if ard else 'bo_iso'
    old, old_d = bo_candidate(points,losses,domain,ard,protocol()['bo'],np.random.default_rng(9))
    new, new_d = f.propose(points,losses,domain,method,5.,protocol()['bo'],np.random.default_rng(9),f.load_config()['pool'])
    np.testing.assert_allclose(old,new,atol=1e-12,rtol=0)
    np.testing.assert_allclose(old_d['lengthscales'],new_d['lengthscales'],atol=1e-12)


def test_v4_pool_baseline_parity_bounds_and_kernel_geometry():
    domain=Domain('ball',np.eye(5)); points=domain.sample(np.random.default_rng(3),20)
    losses=[Quadratic(100)(p) for p in points]; settings=dict(interior=64,surface=64,chunk_size=32)
    old, d0=pool_candidate(points,losses,domain,protocol()['bo'],np.random.default_rng(8),settings)
    new, d1=f.propose(points,losses,domain,'bo_iso_pool',5.,protocol()['bo'],np.random.default_rng(8),settings)
    np.testing.assert_allclose(old,new,atol=1e-12)
    assert d0['pool_index']==d1['pool_index'] and d0['pool_sha256']==d1['pool_sha256']
    q=rotation(404); pool=f.candidate_pool(np.random.default_rng(8),settings)
    scores=f.acquisition_scores(f.model_for(points,losses),pool)
    rotated=f.acquisition_scores(f.model_for(points@q,losses),pool@q)
    np.testing.assert_allclose(scores,rotated,atol=1e-9,rtol=1e-9)
    assert np.argmax(scores)==np.argmax(rotated)
    duplicate=np.repeat(pool[:1],3,axis=0)
    assert np.argmax(f.acquisition_scores(f.model_for(points,losses),duplicate))==0
    ard=f.model_for(points,losses,True,20.)
    assert ard.covar_module.base_kernel.lengthscale.numel()==5
    assert float(ard.covar_module.base_kernel.raw_lengthscale_constraint.upper_bound)==20.
    ard.covar_module.base_kernel.lengthscale=[.1,.3,.7,1.4,3.]
    rotated_ard=f.model_for(points@q,losses,True,20.)
    rotated_ard.load_state_dict(ard.state_dict())
    assert np.max(np.abs(f.acquisition_scores(ard,pool)-f.acquisition_scores(rotated_ard,pool@q)))>1e-3


@pytest.fixture
def folder():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts',prefix='test-v4-') as p:
        yield Path(p)


def test_v4_accounting_checkpoints_assessment_reuse_and_retries(folder,monkeypatch):
    cfg=f.load_config(); job=copy.deepcopy(f.jobs_for(cfg)[0])
    job.update(method='random',budget=300,checkpoints=[100,300])
    calls=[]; assessments=[]
    def objective(u):
        calls.append(u.copy()); return Quadratic(100)(u)
    def battery(label):
        def assess(u):
            assert len(calls)==300
            assessments.append(label); return .01
        return assess
    monkeypatch.setattr(f,'Battery',battery)
    row,cached=f.execute_job(job,cfg,protocol(),folder,{},objective=objective)
    assert row['status']=='complete' and not cached
    assert len(calls)==300 and len(assessments)==row['assessment_calls']==4
    trace=pd.read_csv(ROOT/row['run_path']/'evaluations.csv')
    for cp in row['checkpoints']:
        assert cp['loss']==pytest.approx(trace.loss.iloc[:cp['calls']].min())
    _,cached=f.execute_job(job,cfg,protocol(),folder,{},objective=objective)
    assert cached and len(calls)==300
    (ROOT/row['run_path']/'recommendation.json').write_text('{}')
    failed,_=f.execute_job(job,cfg,protocol(),folder,{},objective=lambda u:float('nan'))
    assert failed['status']=='failed' and failed['evaluations']==1 and '-retry1' in failed['run_id']


@pytest.mark.parametrize('handler',['project','resample'])
def test_v4_cma_repair_budget_and_deadline(folder,handler,monkeypatch):
    cfg=f.load_config();job=copy.deepcopy(f.jobs_for(cfg)[0])
    job.update(method='cma_diag',objective='quadratic',condition=100,domain='box',rotation_seed=404,
               handler=handler,budget=36,checkpoints=[28,36])
    row,_=f.execute_job(job,cfg,protocol(),folder,{})
    assert row['status']=='complete' and [p['calls'] for p in row['checkpoints']]==[28,36]
    assert row['proposals']-row['rejected']==16
    clock = [0.]
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    with pytest.raises(TimeoutError):
        f.execute_job(dict(job,search_seed=99),cfg,protocol(),folder,{},deadline=-1)
    def advancing_objective(u):
        clock[0] += 1.
        return 1.
    row,_=f.execute_job(dict(job,search_seed=100),cfg,protocol(),folder,{},deadline=3.5,objective=advancing_objective)
    assert row['status']=='budget_limited' and row['evaluations']==4
    original_domain = f.Domain
    def setup_expiry(*args):
        clock[0] = 10.
        return original_domain(*args)
    monkeypatch.setattr(f, 'Domain', setup_expiry)
    row,_=f.execute_job(dict(job,search_seed=101),cfg,protocol(),folder,{},deadline=5.,objective=lambda u: 1.)
    assert row['status']=='budget_limited' and row['evaluations']==0


def test_v4_timing_gate_includes_overhead_and_rejects_missing_or_failed_development():
    cfg=f.load_config(); dev=[dict(j,status='complete',attempt_s=1.) for j in f.jobs_for(cfg,True)]
    result=f.timing_forecast(dev,f.jobs_for(cfg),5000,overhead_s=.1)
    assert result['projected_confirmation_s']==pytest.approx(3160*1.1*1.25)
    assert result['fits_allowance']
    assert not f.timing_forecast(dev,f.jobs_for(cfg),100)['fits_allowance']
    dev[0]['status']='failed'
    with pytest.raises(ValueError,match='Development'):
        f.timing_forecast(dev,f.jobs_for(cfg),5000)


def test_v4_interactions_and_cross_block_coverage():
    from bo_vs_es.followup_v4_analysis import paired_contrasts
    rows=[]
    effects={'bo_iso':1.,'bo_iso_pool':.25,'bo_ard':2.,'bo_ard_pool':.75}
    for design in [20,21]:
        for block,values in [('battery_bo',effects),('battery_bound',{'bo_iso_pool':.1,'bo_ard_pool':.5})]:
            for method,effect in values.items():
                for orientation in [-1,404]:
                    rows.append(dict(block=block,objective='battery',method=method,handler='none',
                        initial_design_seed=design,search_replicate=0,calls=100,orientation=orientation,
                        loss=10**(effect if orientation==404 else 0),search_s=1.,
                        run_id=f'{block}-{design}-{method}-{orientation}',complete_group=True))
    frame=pd.DataFrame(rows)
    tables=paired_contrasts(frame)
    np.testing.assert_allclose(tables['bo_interactions'].kernel_by_search,-.5)
    np.testing.assert_allclose(tables['bound_rotation_pairs'].sort_values('method').bound_rotation_change,[-.25,-.25,-.15,-.15])
    interval=group_intervals(tables['bo_interactions'],['block'],'kernel_by_search',f.load_config()).iloc[0]
    assert interval.design_groups==2 and interval.estimate==pytest.approx(-.5)
    frame.loc[frame.block.eq('battery_bo')&frame.initial_design_seed.eq(21),'complete_group']=False
    paired=paired_contrasts(frame)['bound_rotation_pairs']
    assert set(paired.initial_design_seed)=={20}


def test_v4_boundary_difference_of_differences():
    from bo_vs_es.followup_v4_analysis import paired_contrasts
    rows=[]
    for design in [20,21]:
        for rep in [0,1]:
            for calls in [100,300]:
                for handler in ['resample','project']:
                    effect=.2 if handler=='resample' else (.6 if calls==100 else .1)
                    for orientation in [-1,404]:
                        rows.append(dict(block='cma_box',objective='battery',method='cma_diag',handler=handler,
                            initial_design_seed=design,search_replicate=rep,calls=calls,orientation=orientation,
                            loss=10**((1 if calls==100 else 0)+(effect if orientation==404 else 0)),
                            run_id=f'{design}-{rep}-{handler}-{orientation}',complete_group=True))
    tables=paired_contrasts(pd.DataFrame(rows))
    np.testing.assert_allclose(tables['boundary_budget_pairs'].change_300_minus_100,-.5,atol=1e-14)
    assert len(tables['cma_identity_pairs'])==8
