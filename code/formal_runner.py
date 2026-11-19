"""Revised nested training runner. Identity-bound checkpoints and full inner predictions."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from v2_features import ROOT,RULES,Inputs,EnvironmentBuilder,readj,writej,sha,verify_identity,spectral
from feature_algorithms import native
import training_contract as tc

DATA=None
FAMILIES=('LightGBM','XGBoost','SVM','RandomForest')
REFS={'LightGBM':'LightGBM-001','XGBoost':'XGBoost-016','SVM':'SVM-031','RandomForest':'RandomForest-043'}

def init_worker():
    global DATA
    verify_identity();DATA=Inputs()

def fit_predict(policy,o,i,candidate,tr,va,feature_info,tag,save_model=False):
    ctx=DATA.contexts[policy,o,i];guard=DATA.guards[policy,o,i]
    identity=readj(ROOT/'run_identity.json')['identity_sha256']
    if 'feature_state' in feature_info:
        feature_info={k:v for k,v in feature_info.items() if k not in ('feature_state','cars_trace','cars_partitions')}
        feature_info['full_state_location']='features cache bound by policy/outer/inner/block/reducer/params and feature_state_hash'
    key=tc.object_hash([identity,policy,o,i,candidate,tag,list(ctx.training_ids),list(ctx.validation_ids),feature_info])
    predpath=ROOT/'predictions'/f'{key}.npz';logpath=ROOT/'fits'/f'{key}.json'
    # Only formal-stage fit records can resume; failed/QA calls never live here.
    if logpath.exists() and predpath.exists() and not save_model:
        log=readj(logpath)
        if log['prediction_sha256']==sha(predpath):return log,None
        raise RuntimeError('Prediction cache corrupted '+str(predpath))
    model=tc.make_classifier(candidate,o,i)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        tc.fit_current_training(model,candidate,tr,guard,ctx.training_ids)
        prob,pred=tc.predict_current_validation(model,va)
    # Only after prediction does the scoring path ask for current validation truth.
    truth=DATA.validation_y(policy,o,i,ctx.validation_ids)
    scores=tc.score_predictions(truth,pred)
    predpath.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(predpath,sample_id=np.array(ctx.validation_ids),y_true=truth,y_pred=pred,probabilities=prob,fitted_classes=np.asarray(model.classes_,dtype=int))
    log=dict(fit_key=key,run_identity_sha256=identity,policy=policy,outer_fold=o,inner_fold=i,candidate=candidate,task=tag,seed=tc.boundary_seed(o,i,candidate['candidate_id']),training_ids=ctx.training_ids,validation_ids=ctx.validation_ids,n_train_real=sum(ctx.is_real),n_train_virtual=len(ctx.is_real)-sum(ctx.is_real),output_dim=tr.shape[1],scores=scores,feature_info=feature_info,probability_file=str(predpath.relative_to(ROOT)),prediction_sha256=sha(predpath),fitted_classes=np.asarray(model.classes_,dtype=int).tolist(),elapsed_seconds=time.perf_counter()-start)
    if save_model:
        import joblib
        path=ROOT/'models'/f'{key}.joblib';path.parent.mkdir(exist_ok=True)
        joblib.dump(model,path,compress=3);log['model_file']=str(path.relative_to(ROOT));log['model_sha256']=sha(path)
    writej(logpath,log)
    return log,model

def candidate_copy(cid):return dict(next(c for c in DATA.pool if c['candidate_id']==cid))

def checkpoint_path(task):return ROOT/'checkpoints'/f'{tc.object_hash(task)}.json'

def spectral_family(task):
    policy,o,block,method,family=task
    out=checkpoint_path(['spectral',*task])
    if out.exists():return readj(out)
    candidates=[s for s in DATA.spectral if s['block']==block and s['reducer']==method and next(c['classifier'] for c in DATA.pool if c['candidate_id']==s['model_candidate_id'])==family]
    records=[]
    for s in candidates:
        c=candidate_copy(s['model_candidate_id']);c['candidate_id']=s['spectral_candidate_id']
        logs=[]
        for i in range(1,5):
            tr,va,info=spectral(DATA,policy,o,i,block,method,s['reducer_params'])
            log,_=fit_predict(policy,o,i,c,tr,va,info,['spectral',block,method,family])
            logs.append(log)
        key=tc.selection_key([l['scores'] for l in logs],np.mean([l['output_dim'] for l in logs]),c['candidate_id'])
        records.append(dict(spectral_config=s,candidate=c,selection_key=list(key),inner_fit_keys=[l['fit_key'] for l in logs],inner_scores=[l['scores'] for l in logs],mean_macro_f1=-key[0],mean_balanced_accuracy=-key[1],std_macro_f1=key[2],mean_output_dim=key[3]))
    best=min(records,key=lambda r:tuple(r['selection_key']))
    s=best['spectral_config'];tr,va,info=spectral(DATA,policy,o,0,block,method,s['reducer_params'])
    log,_=fit_predict(policy,o,0,best['candidate'],tr,va,info,['spectral',block,method,family],save_model=True)
    result=dict(kind='spectral',policy=policy,outer_fold=o,block=block,reducer=method,family=family,candidates=records,best=best,outer_fit_key=log['fit_key'],outer_scores=log['scores'],run_identity_sha256=verify_identity())
    writej(out,result);return result

def get_backbone(policy,o):
    records=[]
    for method in ('CARS','PCA','PLS-VIP','SPA'):
        for family in FAMILIES:
            path=checkpoint_path(['spectral',policy,o,'FDR',method,family])
            if not path.exists():raise RuntimeError('Incomplete FDR backbone matrix')
            records.append(readj(path))
    best=min(records,key=lambda r:tuple(r['best']['selection_key']))
    return best['best']['spectral_config'],best

def select_FS(policy,o,family,backbone):
    out=ROOT/'FS'/f'{tc.object_hash([policy,o,family])}.json'
    if out.exists():return readj(out)
    ref=candidate_copy(REFS[family])
    builders=[EnvironmentBuilder(DATA,policy,o,i,backbone) for i in range(1,5)]
    module_records=[]
    for ci,combo in enumerate(DATA.combos):
        logs=[];n=[]
        for i,b in enumerate(builders,1):
            tr,va,info=b.build(combo,True)
            log,_=fit_predict(policy,o,i,ref,tr,va,info,['FS_module',family,ci])
            logs.append(log);n.append(len(info['actual_env_factors']))
        k=tc.selection_key([l['scores'] for l in logs],np.mean(n),str(ci))
        # Frozen module-list index is numeric for final tie break.
        k=(*k[:4],ci)
        module_records.append(dict(combo_index=ci,combo=combo,selection_key=list(k),inner_fit_keys=[l['fit_key'] for l in logs],inner_scores=[l['scores'] for l in logs],actual_env_factor_counts=n))
    selected=min(module_records,key=lambda r:tuple(r['selection_key']))
    combo=selected['combo'];folds=[];union=set()
    for i,b in enumerate(builders,1):
        tr,va,info=b.build(combo,True)
        # PI baseline is its own registered fit and retains a model in memory.
        log,model=fit_predict(policy,o,i,ref,tr,va,info,['FS_PI_baseline',family],save_model=False)
        if model is None:
            # Recompute only the deterministic baseline when resuming PI before its checkpoint.
            model=tc.make_classifier(ref,o,i)
            tc.fit_current_training(model,ref,tr,DATA.guards[policy,o,i],DATA.contexts[policy,o,i].training_ids)
            writej(ROOT/'resume_extra_fits'/f'{log["fit_key"]}.json',dict(reason='PI model restored by deterministic fresh refit',fit_key=log['fit_key']))
        folds.append((i,va,info,log,model));union.update(info['layout'])
    pi=[];keep=[]
    for factor in sorted(union):
        deltas=np.zeros((4,3));available=[];repeats=[]
        for i,va,info,log,model in folds:
            present=factor in info['layout'];available.append(present)
            if present:
                truth=DATA.validation_y(policy,o,i,DATA.contexts[policy,o,i].validation_ids)
                for repeat in range(3):
                    xp=tc.grouped_permute(va,info['layout'][factor],o,i,family,factor,repeat)
                    _,pred=tc.predict_current_validation(model,xp)
                    sc=tc.score_predictions(truth,pred)
                    delta=log['scores']['macro_f1']-sc['macro_f1'];deltas[i-1,repeat]=delta
                    repeats.append(dict(inner_fold=i,repeat=repeat,seed=tc.permutation_seed(o,i,family,factor,repeat),baseline_macro_f1=log['scores']['macro_f1'],permuted_macro_f1=sc['macro_f1'],delta_macro_f1=delta))
        summary=tc.permutation_summary(deltas,available)
        pi.append(dict(factor=factor,summary=summary,repeat_records=repeats,fold_repeat_deltas=deltas.tolist()))
        if summary['keep']:keep.append(factor)
    result=dict(policy=policy,outer_fold=o,family=family,backbone=backbone,module_records=module_records,selected_module=selected,permutation_records=pi,selected_factors=keep,run_identity_sha256=verify_identity())
    writej(out,result);return result

def fusion_family(task):
    policy,o,branch,family=task
    out=checkpoint_path(['fusion',*task])
    if out.exists():return readj(out)
    backbone,_=get_backbone(policy,o)
    if branch=='FS':
        fs=select_FS(policy,o,family,backbone);combo=fs['selected_module']['combo'];factors=fs['selected_factors']
    else:combo=DATA.protocol['branches'][branch]['modules'];factors=None
    include=DATA.protocol['branches'][branch]['include_spectral']
    folds=[]
    for i in range(1,5):
        builder=EnvironmentBuilder(DATA,policy,o,i,backbone)
        tr,va,info=builder.build(combo,include,selected=factors);folds.append((i,tr,va,info))
    records=[]
    for c in DATA.pool:
        if c['classifier']!=family:continue
        logs=[]
        for i,tr,va,info in folds:
            log,_=fit_predict(policy,o,i,c,tr,va,info,['fusion',branch,family]);logs.append(log)
        key=tc.selection_key([l['scores'] for l in logs],np.mean([l['output_dim'] for l in logs]),c['candidate_id'])
        records.append(dict(candidate=c,selection_key=list(key),inner_fit_keys=[l['fit_key'] for l in logs],inner_scores=[l['scores'] for l in logs],mean_macro_f1=-key[0],mean_balanced_accuracy=-key[1],std_macro_f1=key[2],mean_output_dim=key[3]))
    best=min(records,key=lambda r:tuple(r['selection_key']))
    builder=EnvironmentBuilder(DATA,policy,o,0,backbone);tr,va,info=builder.build(combo,include,selected=factors)
    log,_=fit_predict(policy,o,0,best['candidate'],tr,va,info,['fusion',branch,family],save_model=True)
    result=dict(kind='fusion',policy=policy,outer_fold=o,branch=branch,family=family,backbone=backbone,combo=combo,selected_factors=factors,candidates=records,best=best,outer_fit_key=log['fit_key'],outer_scores=log['scores'],run_identity_sha256=verify_identity())
    writej(out,result);return result

def run_stage(name,tasks,worker,workers):
    start=time.perf_counter();done=0
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        futures={pool.submit(worker,t):t for t in tasks}
        for f in as_completed(futures):
            try:f.result()
            except Exception as ex:
                writej(ROOT/'qa/formal_failure.json',dict(status='FAIL_FORMAL_STAGE',stage=name,task=futures[f],error=repr(ex),traceback=traceback.format_exc()))
                for future in futures:future.cancel()
                raise
            done+=1
            print(f'{name}: family {done}/{len(tasks)}, {time.perf_counter()-start:.1f}s',flush=True)
            writej(ROOT/'progress.json',dict(stage=name,completed_family_tasks=done,total_family_tasks=len(tasks),elapsed_seconds=time.perf_counter()-start,status='RUNNING'))

def main(workers,phase):
    identity=verify_identity()
    if readj(ROOT/'qa/integration_acceptance.json')['status']!='PASS_BEFORE_FORMAL_CLASSIFIER_TRAINING':raise RuntimeError('Integration QA gate not passed')
    stages=[('R_spectral',[('R',o,b,m,f) for o in range(1,6) for b in ('OR','CR','FDR') for m in ('CARS','PCA','PLS-VIP','SPA') for f in FAMILIES],spectral_family),('R_fusion',[('R',o,b,f) for o in range(1,6) for b in ('N59','I63','FALL','FS') for f in FAMILIES],fusion_family),('RV_spectral',[('R+V',o,'FDR',m,f) for o in range(1,6) for m in ('CARS','PCA','PLS-VIP','SPA') for f in FAMILIES],spectral_family),('RV_fusion',[('R+V',o,b,f) for o in range(1,6) for b in ('N59','I63','FALL','FS') for f in FAMILIES],fusion_family)]
    marker=ROOT/'formal_execution_complete.json'
    previous=readj(marker) if marker.exists() else {}
    if previous and previous['run_identity_sha256']!=identity:
        raise RuntimeError('Training stage record is bound to a different run')
    completed=set(previous.get('completed_stages', []))
    for name,tasks,worker in stages:
        if phase=='all' or phase==name:
            run_stage(name,tasks,worker,workers)
            completed.add(name)
            writej(marker,dict(status='EXECUTION_COMPLETE_PENDING_RESULT_QA',
                               phase='all' if len(completed)==4 else name,
                               completed_stages=sorted(completed),run_identity_sha256=identity))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--phase',default='all',choices=['all','R_spectral','R_fusion','RV_spectral','RV_fusion']);a=ap.parse_args();main(a.workers,a.phase)
