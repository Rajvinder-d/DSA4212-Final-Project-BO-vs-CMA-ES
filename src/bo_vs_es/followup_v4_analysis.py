"""V4 analysis only: validate recorded artifacts; never launch optimisation."""
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bo_vs_es import ROOT
from bo_vs_es.followup_v4 import load_config, jobs_for, atomic_dump, now
from bo_vs_es.followup_analysis import read_collection, group_intervals, paired_budget_changes
from bo_vs_es.objectives.data import sha256

COLORS = dict(bo_iso='#0072B2', bo_ard='#56B4E9', bo_iso_pool='#009E73',
              bo_ard_pool='#CC9900', cma_full='#D55E00', cma_diag='#CC79A7', random='#777777')


def paired_contrasts(snapshots):
    """Match treatments before any averaging; only complete block/design groups."""
    usable = snapshots[snapshots.complete_group].copy()
    if not np.isfinite(usable.loss).all() or (usable.loss <= 0).any():
        raise ValueError('Zero/nonfinite loss: log-ratio estimands require explicit investigation')
    keys = ['block','method','handler','initial_design_seed','search_replicate','calls']
    baseline = usable[usable.orientation.eq(-1)]
    pairs = usable[usable.orientation.ne(-1)].merge(baseline[keys+['loss','run_id']],on=keys,
        suffixes=('','_identity'),validate='many_to_one')
    pairs['effect'] = np.log10(pairs.loss/pairs.loss_identity)
    pairs['absolute_effect'] = pairs.effect.abs()
    tables = {'rotation_pairs':pairs}
    bo = pairs[pairs.block.isin(['battery_bo','quadratic_bo']) & pairs.method.str.startswith('bo')]
    index = ['block','objective','initial_design_seed','search_replicate','calls','orientation']
    if len(bo):
        wide = bo.pivot(index=index,columns='method',values='effect').reindex(columns=['bo_iso','bo_ard','bo_iso_pool','bo_ard_pool']).dropna()
        wide['iso_pool_minus_continuous'] = wide.bo_iso_pool-wide.bo_iso
        wide['ard_pool_minus_continuous'] = wide.bo_ard_pool-wide.bo_ard
        wide['kernel_by_search'] = wide.ard_pool_minus_continuous-wide.iso_pool_minus_continuous
        tables['bo_interactions'] = wide.reset_index()
        quality_keys=['block','objective','initial_design_seed','search_replicate','calls','orientation']
        primary=usable[usable.block.isin(['battery_bo','quadratic_bo'])]
        comparisons=[]
        for kernel in ['iso','ard']:
            joined=primary[primary.method.eq('bo_'+kernel+'_pool')].merge(
                primary[primary.method.eq('bo_'+kernel)][quality_keys+['loss','search_s','run_id']],
                on=quality_keys,suffixes=('','_continuous'),validate='one_to_one')
            joined['kernel']=kernel
            joined['quality_effect']=np.log10(joined.loss/joined.loss_continuous)
            joined['search_seconds_difference']=joined.search_s-joined.search_s_continuous
            comparisons.append(joined)
        tables['pool_quality_pairs']=pd.concat(comparisons,ignore_index=True)
    bound_keys=['objective','method','initial_design_seed','search_replicate','calls','orientation']
    bound=usable[usable.block.isin(['battery_bound','quadratic_bound'])]
    base=usable[usable.block.isin(['battery_bo','quadratic_bo']) & usable.method.str.endswith('_pool')]
    if len(bound):
        matched=bound.merge(base[bound_keys+['loss','run_id']],on=bound_keys,suffixes=('','_bound5'),validate='one_to_one')
        matched['quality_effect']=np.log10(matched.loss/matched.loss_bound5)
        tables['bound_quality_pairs']=matched
        br=pairs[pairs.block.isin(['battery_bound','quadratic_bound'])]
        b0=pairs[pairs.block.isin(['battery_bo','quadratic_bo']) & pairs.method.str.endswith('_pool')]
        matched=br.merge(b0[bound_keys+['effect','run_id']],on=bound_keys,suffixes=('','_bound5'),validate='one_to_one')
        matched['bound_rotation_change']=matched.effect-matched.effect_bound5
        tables['bound_rotation_pairs']=matched
    boundary=pairs[pairs.block.eq('cma_box')]
    if len(boundary):
        idx=['method','initial_design_seed','search_replicate','calls','orientation']
        wide=boundary.pivot(index=idx,columns='handler',values='effect').reindex(columns=['project','resample']).dropna()
        wide['interaction']=wide['project']-wide['resample']
        tables['boundary_pairs']=wide.reset_index()
        tables['boundary_budget_pairs']=paired_budget_changes(wide.reset_index(),
            ['method','initial_design_seed','search_replicate','orientation'],'interaction')
    cma=usable[usable.method.str.startswith('cma')].copy()
    if len(cma):
        cma['log10_loss']=np.log10(cma.loss)
        tables['cma_budget_pairs']=paired_budget_changes(cma,
            ['block','method','handler','initial_design_seed','search_replicate','orientation','run_id'],'log10_loss')
        identity=cma[cma.orientation.eq(-1)]
        records=[]
        keys=['block','method','handler','initial_design_seed','calls']
        for group, values in identity.groupby(keys):
            for a,b in itertools.combinations(values.itertuples(),2):
                records.append(dict(zip(keys,group),replicate_a=a.search_replicate,replicate_b=b.search_replicate,
                    run_id_a=a.run_id,run_id_b=b.run_id,effect=float(np.log10(b.loss/a.loss))))
        tables['cma_identity_pairs']=pd.DataFrame(records)
    return tables


def export_analysis():
    cfg=load_config(); root,meta,snap,targets,health=read_collection(cfg)
    out=ROOT/'results/followup_v4';out.mkdir(parents=True,exist_ok=True)
    written=[]
    def table(name,frame):
        frame.to_csv(out/(name+'.csv'),index=False);written.append(name+'.csv')
    def figure(name,fig):
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig);written.append(name+'.png')
    planned=pd.DataFrame(jobs_for(cfg)); rows=pd.DataFrame(meta['rows']) if meta else pd.DataFrame()
    counts=planned.groupby(['block','initial_design_seed']).size().rename('planned').to_frame()
    if len(rows):
        counts=counts.join(rows[rows.status.eq('complete')].groupby(['block','initial_design_seed']).size().rename('complete'))
    else:
        counts['complete']=0
    counts=counts.fillna(0).reset_index();counts['complete_group']=counts.planned.eq(counts.complete)
    table('coverage',counts);table('run_status',rows.drop(columns=['checkpoints'],errors='ignore'))
    replay=root/'development/replays.json'
    if replay.exists():
        records=[]
        fits=[]
        for r in json.loads(replay.read_text()):
            records.append({k:v for k,v in r.items() if k not in ['fits','independent_prediction_differences']})
            for orientation,fit in zip(['identity','Q101'],r['fits']):
                fits.append(dict(objective=r['objective'],design=r['design'],observations=r['observations'],
                    kernel=r['kernel'],upper=r['upper'],mode=r['mode'],orientation=orientation,**fit))
        table('development_replays',pd.DataFrame(records))
        table('development_replay_fits',pd.DataFrame(fits))
    if len(snap):
        table('checkpoints',snap);table('targets',targets)
        if len(health):
            upper=rows[['run_id','lengthscale_upper']]
            health=health.merge(upper,on='run_id',validate='many_to_one')
            health['near_upper_bound']=health.lengthscale_max>=.99*health.lengthscale_upper
            health['near_lower_bound']=health.lengthscale_min<=.0303
            table('optimiser_health',health)
            table('health_summary',health.groupby(['block','method','lengthscale_upper']).agg(
                fits=('fit_status','size'),near_upper=('near_upper_bound','sum'),near_lower=('near_lower_bound','sum'),
                warnings=('warnings','sum')).reset_index())
            table('fit_statuses',health.groupby(['block','method','fit_status']).size().rename('fits').reset_index())
        derived=paired_contrasts(snap)
        for name,frame in derived.items():table(name,frame)
        pairs=derived['rotation_pairs']
        interval_specs=[('rotation_pairs',['block','objective','method','handler','calls','orientation'],'effect','rotation_intervals'),
            ('rotation_pairs',['block','objective','method','handler','calls','orientation'],'absolute_effect','absolute_rotation_intervals'),
            ('pool_quality_pairs',['block','kernel','orientation'],'quality_effect','pool_quality_intervals'),
            ('pool_quality_pairs',['block','kernel','orientation'],'search_seconds_difference','pool_timing_intervals'),
            ('bound_quality_pairs',['objective','method','orientation'],'quality_effect','bound_quality_intervals'),
            ('bound_rotation_pairs',['objective','method','orientation'],'bound_rotation_change','bound_rotation_intervals'),
            ('boundary_pairs',['method','calls','orientation'],'interaction','boundary_intervals'),
            ('boundary_budget_pairs',['method','orientation'],'change_300_minus_100','boundary_budget_intervals'),
            ('cma_budget_pairs',['block','method','handler','orientation'],'change_300_minus_100','cma_budget_intervals')]
        intervals={}
        for source,keys,value,name in interval_specs:
            if source in derived and len(derived[source]):
                result=group_intervals(derived[source],keys,value,cfg)
                result['interpretation']='pointwise 95% design-bootstrap interval; conditional on selected rotations; predeclared V4'
                table(name,result);intervals[name]=result
        if 'bo_interactions' in derived:
            for value in ['iso_pool_minus_continuous','ard_pool_minus_continuous','kernel_by_search']:
                result=group_intervals(derived['bo_interactions'],['block','objective','orientation'],value,cfg)
                result['interpretation']='pointwise 95% design-bootstrap interval; conditional on selected rotations; predeclared V4'
                result['estimand']=value
                table(value+'_intervals',result)
        usable=snap[snap.complete_group].copy()
        usable['overhead_s']=usable.search_s-usable.objective_s
        table('quality_and_timing',usable.groupby(['block','method','handler','calls','orientation']).agg(
            runs=('run_id','size'),design_groups=('initial_design_seed','nunique'),median_loss=('loss','median'),
            median_rmse=('calibration_rmse_v','median'),median_search_s=('search_s','median'),
            median_objective_s=('objective_s','median'),median_overhead_s=('overhead_s','median'),
            mean_boundary_fraction=('evaluated_boundary_fraction','mean'),mean_rejected_fraction=('rejected_fraction','mean'),
            mean_repaired_fraction=('repaired_fraction','mean')).reset_index())
        battery=usable[usable.objective.eq('battery')]
        table('heldout',battery.groupby(['block','method','handler','calls','orientation'])[['fuds80_rmse_v','fuds50_rmse_v']].median().reset_index())
        if len(targets):
            table('target_summary',targets[targets.complete_group].groupby(
                ['block','method','handler','calls','orientation','target_rmse_v']).agg(runs=('hit','size'),hits=('hit','sum'),
                hit_rate=('hit','mean'),median_calls_if_reached=('calls_to_target','median'),
                median_seconds_if_reached=('seconds_to_target','median')).reset_index())
        # Read paths once for sequential diagnostics and the primary convergence figure.
        traces={}
        selected=usable[usable.method.str.startswith('bo')|usable.method.eq('random')]
        for row in selected.itertuples():
            traces[row.run_id]=pd.read_csv(ROOT/row.run_path/'evaluations.csv')
        paths=[]
        path_keys=['block','method','initial_design_seed','search_replicate']
        for key,group in selected.groupby(path_keys):
            base=group[group.orientation.eq(-1)].iloc[0]
            u0=np.array([json.loads(u) for u in traces[base.run_id].u])
            for row in group[group.orientation.ne(-1)].itertuples():
                u1=np.array([json.loads(u) for u in traces[row.run_id].u])
                for i,distance in enumerate(np.linalg.norm(u1-u0,axis=1)):
                    paths.append(dict(zip(path_keys,key),orientation=row.orientation,evaluation=i+1,distance=float(distance),
                                      run_id=row.run_id,identity_run_id=base.run_id))
        paths=pd.DataFrame(paths);table('path_distances',paths)
        if len(paths):
            table('final_path_summary',paths[paths.evaluation.eq(100)].groupby(['block','method']).distance.agg(['median','max','size']).reset_index())
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for ax,block in zip(axes,['battery_bo','quadratic_bo']):
                for method,g in paths[paths.block.eq(block)].groupby('method'):
                    median=g.groupby(['initial_design_seed','evaluation']).distance.mean().groupby('evaluation').median()
                    ax.plot(median.index,np.maximum(median.values,1e-16),label=method,color=COLORS[method])
                ax.set(yscale='log',xlabel='Counted calls',ylabel='Mapped candidate distance',title=block+'; floor 1e-16')
                if ax.lines:ax.legend(fontsize=7)
            fig.suptitle('Median of within-design rotation averages');figure('path_matching',fig)
        if 'rotation_intervals' in intervals:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for ax,block in zip(axes,['battery_bo','quadratic_bo']):
                subset=intervals['rotation_intervals'];subset=subset[subset.block.eq(block)]
                for i,(method,g) in enumerate(subset.groupby('method')):
                    positions=np.arange(len(g))+.08*(i-2)
                    ax.errorbar(positions,g.estimate,yerr=[g.estimate-g.low,g.high-g.estimate],fmt='o',label=method,color=COLORS[method])
                ax.axhline(0,color='black',lw=1);ax.set(xticks=range(3),xticklabels=['Q404','Q505','Q606'],
                    ylabel='Mean log10 rotated / identity loss',title=block)
                if ax.lines:ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1))
            figure('bo_rotation_effects',fig)
        primary=usable[usable.block.eq('battery_bo')]
        if len(primary):
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for method,g in primary.groupby('method'):
                curves=[]
                for row in g.itertuples():
                    trace=traces[row.run_id];y=np.sqrt(trace.loss.cummin().to_numpy())*1000
                    axes[1].plot(trace.elapsed_s,y,color=COLORS[method],alpha=.12,lw=.6)
                    curves.append(pd.Series(y,index=range(1,101),name=row.initial_design_seed))
                mean_by_design=pd.concat(curves,axis=1).T.groupby(level=0).mean()
                axes[0].plot(range(1,101),mean_by_design.median(),color=COLORS[method],label=method)
                axes[1].plot([],[],color=COLORS[method],label=method)
            for ax in axes:ax.set(yscale='log',ylabel='Best calibration RMSE (mV)');ax.legend(fontsize=7)
            axes[0].set_xlabel('Counted calls');axes[1].set(xscale='log',xlabel='Elapsed search time (s)')
            fig.suptitle('Battery BO: call-panel median of design averages; individual timed paths');figure('quality_and_time',fig)
        if 'cma_identity_pairs' in derived and len(derived['cma_identity_pairs']):
            controls=derived['cma_identity_pairs'].copy();controls['absolute']=controls.effect.abs()
            keys=['block','method','handler','initial_design_seed','calls']
            spreads=[controls.groupby(keys).absolute.mean().reset_index().assign(comparison='I vs I')]
            for orientation,g in pairs[pairs.method.str.startswith('cma')].groupby('orientation'):
                spreads.append(g.groupby(keys).absolute_effect.mean().rename('absolute').reset_index().assign(comparison=f'Q{orientation} vs I'))
            spread=pd.concat(spreads);table('cma_variability_by_design',spread)
            table('cma_variability_summary',spread.groupby(['block','method','handler','calls','comparison']).agg(
                designs=('initial_design_seed','nunique'),mean_absolute_log_ratio=('absolute','mean')).reset_index())
            fig,axes=plt.subplots(2,2,figsize=(12,7),sharey=True)
            labels=['I vs I','Q404 vs I','Q505 vs I','Q606 vs I']
            for i,method in enumerate(['cma_full','cma_diag']):
                for j,calls in enumerate([100,300]):
                    ax=axes[i,j];g=spread[spread.block.eq('cma_ball')&spread.method.eq(method)&spread.calls.eq(calls)]
                    for k,label in enumerate(labels):
                        v=g[g.comparison.eq(label)].absolute
                        if len(v):
                            ax.scatter(np.full(len(v),k),v,color=COLORS[method],alpha=.5)
                            ax.plot([k-.2,k+.2],[v.mean()]*2,color='black')
                    ax.set(xticks=range(4),xticklabels=labels,ylabel='Within-design mean absolute log ratio',title=f'{method}, {calls} calls')
            fig.suptitle('CMA ball: ordinary identity variability versus rotation; descriptive, not equivalence');figure('cma_variability',fig)
        cma_state=[]
        for row in usable[usable.method.str.startswith('cma')].drop_duplicates('run_id').itertuples():
            for line in (ROOT/row.run_path/'diagnostics.jsonl').read_text().splitlines():
                d=json.loads(line);e=d['eigenvalues']
                cma_state.append(dict(run_id=row.run_id,block=row.block,method=row.method,handler=row.handler,
                    initial_design_seed=row.initial_design_seed,orientation=row.orientation,calls=d['after_evaluations'],
                    sigma=d['sigma'],spectral_ratio=max(e)/min(e)))
        if cma_state:
            state=pd.DataFrame(cma_state);table('cma_adaptation',state)
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for method,g in state[state.block.eq('cma_ball')].groupby('method'):
                med=g.groupby(['initial_design_seed','calls'])[['sigma','spectral_ratio']].mean().groupby('calls').median()
                for ax,metric in zip(axes,['sigma','spectral_ratio']):
                    ax.plot(med.index,med[metric],label=method,color=COLORS[method]);ax.set(xlabel='Counted calls',ylabel=metric,yscale='log')
                    ax.axvline(100,color='black',ls='--');ax.legend()
            fig.suptitle('CMA shape diagnostics: not directional alignment or causal evidence');figure('cma_adaptation',fig)
    all_attempts=[]
    for stage in ['development','confirmation']:
        for path in (root/stage).glob('*/summary.json'):
            all_attempts.append(json.loads(path.read_text()))
    table('all_attempts',pd.DataFrame(all_attempts).drop(columns=['checkpoints'],errors='ignore'))
    done=rows[rows.status.eq('complete')] if len(rows) else rows
    audit=dict(generated_utc=now(),execution='agent',planned_runs=len(planned),
        complete_runs=len(done),search_calls=int(done.evaluations.sum()) if len(done) else 0,
        assessment_calls=int(done.assessment_calls.sum()) if len(done) else 0,
        all_attempt_search_calls=sum(r.get('evaluations',0) for r in all_attempts),
        status='complete' if len(done)==len(planned) else 'partial',
        collection_sha256=sha256(root/'confirmation/collection.json') if meta else None,
        analysis_sha256=sha256(Path(__file__)),outputs={name:sha256(out/name) for name in written},
        interpretation='Predeclared V4; pointwise design intervals; conditional on selected rotations; no equivalence claim')
    atomic_dump(out/'validation.json',audit)
    return audit


def audit_notebook(config):
    out=ROOT/'results/followup_v4'; audit=json.loads((out/'validation.json').read_text())
    for name,digest in audit['outputs'].items():
        if sha256(out/name)!=digest:raise ValueError(f'Modified analysis output {name}')
    notebook=ROOT/'notebooks/06_v4_mechanism_and_confirmation.ipynb'
    nb=json.loads(notebook.read_text());code=[c for c in nb['cells'] if c['cell_type']=='code']
    if any(c.get('execution_count') is None for c in code):raise ValueError('Unexecuted notebook cell')
    if any(o['output_type']=='error' for c in code for o in c.get('outputs',[])):raise ValueError('Notebook cell error')
    if any('<td>...</td>' in ''.join(o.get('data',{}).get('text/html',[])) for c in code for o in c.get('outputs',[])):
        raise ValueError('Truncated notebook table')
    inventory=ROOT/'artifacts/followup_v4_operations/preserved_before.json'
    preserved=json.loads(inventory.read_text())
    changed=[name for name,digest in preserved.items() if not (ROOT/name).exists() or sha256(ROOT/name)!=digest]
    if changed:raise ValueError(f'Pre-V4 files changed: {changed[:10]}')
    if audit['status']=='complete' and (audit['search_calls']!=796000 or audit['assessment_calls']!=10560):
        raise ValueError('Unexpected complete-grid accounting')
    result=dict(status=audit['status'],complete_runs=audit['complete_runs'],search_calls=audit['search_calls'],
        assessment_calls=audit['assessment_calls'],outputs_verified=len(audit['outputs']),
        notebook_sha256=sha256(notebook),executed_cells=len(code),preserved_files=len(preserved),
        visual_review='pending human/agent inspection after execution')
    atomic_dump(out/'execution_audit.json',result)
    return result
