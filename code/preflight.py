"""Freeze runner identity, build all new fold-local reducers, then poison held-out data."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from v2_features import ROOT,RULES,INPUT,Inputs,EnvironmentBuilder,fit_spectral,spectral_path,sha,readj,writej,verify_identity
from feature_algorithms import array_hash
import training_contract as tc
from settings import PROTOCOL, MODE

DATA=None
def init_worker():
    global DATA
    verify_identity();DATA=Inputs()

def feature_job(task):
    policy,o,i,block,method,params=task
    path=spectral_path(*task)
    if path.with_suffix('.npz').exists() and path.with_suffix('.json').exists():
        info=readj(path.with_suffix('.json'))
        if info.get('run_identity_sha256')==readj(ROOT/'run_identity.json')['identity_sha256']:
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                if array_hash(z['train'])==info['train_array_hash'] and array_hash(z['validation'])==info['validation_array_hash']:return str(path),'resume'
    started=time.perf_counter()
    tr,va,info=fit_spectral(DATA,*task)
    info['run_identity_sha256']=readj(ROOT/'run_identity.json')['identity_sha256']
    info['elapsed_seconds']=time.perf_counter()-started
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'),train=tr,validation=va)
    writej(path.with_suffix('.json'),info)
    return str(path),'new'

def freeze():
    # Bind this release, these authorized inputs and these newly generated splits.
    if readj(RULES/'spatial/validation_design.json')['input_metadata_sha256'] != sha(INPUT/'metadata.csv'):
        raise ValueError('Regenerate spatial folds for the prepared input')
    run_files = [INPUT/'metadata.csv', INPUT/'modeling_inputs.npz', INPUT/'preparation.json']
    run_files += [RULES/'spatial'/name for name in ('validation_design.json','sample_groups.csv','outer_manifest.csv','inner_manifest.csv')]
    identity = dict(
        version='gf5_jars_review_release', mode=MODE,
        run_files={p.relative_to(ROOT).as_posix(): sha(p) for p in run_files},
        protocol_files={p.name: sha(p) for p in sorted(PROTOCOL.iterdir()) if p.is_file()},
        code_files={p.name: sha(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))
                    if p.name not in {'audit_release.py', 'paper_outputs.py'}},
        sample_policies=['R','R+V'], classifier_threads=1, blas_threads=1,
    )
    identity['identity_sha256']=tc.object_hash(identity)
    if (ROOT/'run_identity.json').exists():
        if readj(ROOT/'run_identity.json')['identity_sha256']!=identity['identity_sha256']:raise RuntimeError('Existing run identity differs; archive separate attempt before code changes')
    else:writej(ROOT/'run_identity.json',identity)

def main(workers):
    freeze();data=Inputs();tasks=[]
    for policy in ('R','R+V'):
        blocks=('OR','CR','FDR') if policy=='R' else ('FDR',)
        methods=data.methods
        for o,i in sorted(data.frames):
            for b in blocks:
                for method,params in methods.items():
                    for param in params:tasks.append((policy,o,i,b,method,param))
    total=len(tasks)
    if MODE == 'paper' and total != 900:
        raise ValueError('Expected 900 registered feature tasks')
    started=time.perf_counter();done=0
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        futures={pool.submit(feature_job,t):t for t in tasks}
        for f in as_completed(futures):
            try:f.result()
            except Exception as ex:
                writej(ROOT/'qa/integration_failure.json',dict(status='FAIL_BEFORE_CLASSIFIER_TRAINING',task=futures[f],error=repr(ex)))
                for future in futures:future.cancel()
                raise
            done+=1
            if done%25==0 or done==total:
                print(f'FEATURE {done}/{total} elapsed {time.perf_counter()-started:.1f}s',flush=True)
                writej(ROOT/'progress.json',dict(stage='integration_feature_preflight',completed=done,total=total,elapsed_seconds=time.perf_counter()-started,production_classifier_fits=0))
    checks=[]
    def check(name,ok):
        checks.append(dict(check=name,pass_=bool(ok)))
        if not ok:raise AssertionError(name)
    # Neutral PCA config is for builder structure QA only; not a selected main backbone.
    neutral=dict(reducer='PCA',reducer_params=data.methods['PCA'][-1])
    for policy in ('R','R+V'):
        for o,i in sorted(data.frames):
            builder=EnvironmentBuilder(data,policy,o,i,neutral)
            for combo,include,n in [(('T','H','C','L'),False,59),(('T','E','H','C','L'),False,63)]:
                tr,va,info=builder.build(combo,include)
                check(f'{policy}/{o}/{i} original candidates {n}',sum(len(data.modules[c]) for c in combo)==n)
                check(f'{policy}/{o}/{i} no S {n}',info['spectral_state_hash'] is None and not any(c.startswith('FDR_') for c in info['feature_names']))
                check(f'{policy}/{o}/{i} finite environment {n}',np.isfinite(tr).all() and np.isfinite(va).all())
                if 'LC_class' in info['layout']:
                    col=info['layout']['LC_class'];check(f'{policy}/{o}/{i} LC valid training {n}',np.isin(tr[:,col],[0,1]).all() and (tr[:,col].sum(axis=1)==1).all())
            for combo in data.combos:
                tr,va,info=builder.build(combo,True)
                check(f'{policy}/{o}/{i} combo {combo} layout finite',np.isfinite(tr).all() and np.isfinite(va).all() and len(info['feature_names'])==tr.shape[1] and tr.shape[1]==va.shape[1])
            ctx=data.contexts[policy,o,i]
            forbidden=[v for v in data.ids if v not in ctx.training_ids and v not in ctx.validation_ids]
            if forbidden:
                try:data.rows(data.env,[forbidden[0]],ctx);blocked=False
                except ValueError:blocked=True
                check(f'{policy}/{o}/{i} sealed feature request rejected',blocked)
    for policy in ('R','R+V'):
        poisoned=Inputs(poison=(policy,1,1))
        methods=data.methods
        for method,params in methods.items():
            for p in params:
                tr0,va0,a=fit_spectral(data,policy,1,1,'FDR',method,p)
                tr1,va1,b=fit_spectral(poisoned,policy,1,1,'FDR',method,p)
                check(f'{policy} poison {method}/{p} training unchanged',a['feature_state_hash']==b['feature_state_hash'] and np.array_equal(tr0,tr1) and np.array_equal(a['classifier_scaler']['mean'],b['classifier_scaler']['mean']) and np.array_equal(a['classifier_scaler']['scale'],b['classifier_scaler']['scale']))
                check(f'{policy} poison {method}/{p} validation transformed',not np.array_equal(va0,va1))
        b0=EnvironmentBuilder(data,policy,1,1,neutral);b1=EnvironmentBuilder(poisoned,policy,1,1,neutral)
        for combo in data.combos:
            tr0,va0,a=b0.build(combo,True);tr1,va1,b=b1.build(combo,True)
            check(f'{policy} poison combo {combo} train state unchanged',a['layer1_hash']==b['layer1_hash'] and a['environment_scaler']==b['environment_scaler'] and np.array_equal(tr0,tr1))
            # Existing S caches are not poisoned; the independently fitted spectral sentinel above tests S.
            if 'L' in combo:check(f'{policy} unseen LC validation zero {combo}',(va1[:,b['layout']['LC_class']]==0).all() if 'LC_class' in b['layout'] else True)
    (ROOT/'qa').mkdir(exist_ok=True)
    pd.DataFrame(checks).to_csv(ROOT/'qa/integration_checks.csv',index=False,encoding='utf-8-sig')
    writej(ROOT/'qa/integration_acceptance.json',dict(status='PASS_BEFORE_FORMAL_CLASSIFIER_TRAINING',feature_caches=total,checks=len(checks),failed=0,production_classifier_fits=0,identity_sha256=verify_identity(),elapsed_seconds=time.perf_counter()-started))
    print('INTEGRATION PASS',len(checks),f'checks, {total} feature caches, classifier fits 0',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();main(args.workers)
