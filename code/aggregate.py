"""Independently verify all predictions, inner selections, coverage and fit budget."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from settings import ROOT, MODE
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,cohen_kappa_score,precision_recall_fscore_support,confusion_matrix
from v2_features import Inputs,readj,writej,verify_identity,sha
from formal_runner import checkpoint_path,FAMILIES,get_backbone
import formal_runner
import training_contract as tc

def metric(y,p):
    pr,re,f1,n=precision_recall_fscore_support(y,p,labels=list(range(6)),zero_division=0)
    return dict(macro_f1=float(f1_score(y,p,labels=list(range(6)),average='macro',zero_division=0)),overall_accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),kappa=float(cohen_kappa_score(y,p,labels=list(range(6)))),per_class_precision=pr.tolist(),per_class_recall=re.tolist(),per_class_f1=f1.tolist(),support=n.tolist(),confusion_matrix=confusion_matrix(y,p,labels=list(range(6))).tolist())

def main():
    ident=verify_identity();d=Inputs();formal_runner.DATA=d
    completion=readj(ROOT/'formal_execution_complete.json')
    assert completion['status']=='EXECUTION_COMPLETE_PENDING_RESULT_QA' and completion['phase']=='all'
    assert completion['run_identity_sha256']==ident
    checks=[];rows=[];fit_logs={}
    def check(name,ok):
        checks.append(dict(check=name,pass_=bool(ok)))
        if not ok:raise AssertionError(name)
    for path in sorted((ROOT/'fits').glob('*.json')):
        log=readj(path)
        policy=log['policy'];o=log['outer_fold'];i=log['inner_fold'];ctx=d.contexts[policy,o,i]
        check(log['fit_key']+' version/IDs',log['run_identity_sha256']==ident and tuple(log['training_ids'])==ctx.training_ids and tuple(log['validation_ids'])==ctx.validation_ids)
        check(log['fit_key']+' seed',log['seed']==tc.boundary_seed(o,i,log['candidate']['candidate_id']))
        predpath=ROOT/log['probability_file'];check(log['fit_key']+' prediction SHA',sha(predpath)==log['prediction_sha256'])
        with np.load(predpath,allow_pickle=False) as z:
            y=z['y_true'];p=z['y_pred'];probs=z['probabilities'];ids=z['sample_id']
            check(log['fit_key']+' validation order/truth',tuple(ids)==ctx.validation_ids and np.array_equal(y,d.validation_y(policy,o,i,ctx.validation_ids)))
            check(log['fit_key']+' probability/argmax',probs.shape==(len(ids),6) and np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(axis=1),1) and np.array_equal(probs.argmax(axis=1),p))
            check(log['fit_key']+' MacroF1 independently reproduced',np.isclose(log['scores']['macro_f1'],f1_score(y,p,labels=list(range(6)),average='macro',zero_division=0)))
            check(log['fit_key']+' BA/OA/confusion independently reproduced',
                  np.isclose(log['scores']['balanced_accuracy'],balanced_accuracy_score(y,p))
                  and np.isclose(log['scores']['overall_accuracy'],accuracy_score(y,p))
                  and np.array_equal(log['scores']['confusion_matrix'],confusion_matrix(y,p,labels=list(range(6)))))
            check(log['fit_key']+' source counts',log['n_train_real']==sum(ctx.is_real)
                  and log['n_train_virtual']==len(ctx.is_real)-sum(ctx.is_real)
                  and (policy!='R' or log['n_train_virtual']==0))
        if i==0:
            modelpath=ROOT/log['model_file']
            check(log['fit_key']+' saved outer model SHA',sha(modelpath)==log['model_sha256'])
        info=log['feature_info']
        if log['task'][0]=='fusion':
            branch=log['task'][1]
            check(log['fit_key']+' true environment-only baseline',
                  branch not in ('N59','I63') or (not info['include_spectral'] and info['spectral_state_hash'] is None))
            if branch=='N59':
                check(log['fit_key']+' excludes GF5 indices',not set(info['actual_env_factors'])&set(d.modules['E']))
        rows.append(dict(policy=policy,outer_fold=o,inner_fold=i,candidate_id=log['candidate']['candidate_id'],classifier=log['candidate']['classifier'],strategy=log['candidate']['strategy'],task=str(log['task']),seed=log['seed'],n_train_real=log['n_train_real'],n_train_virtual=log['n_train_virtual'],dimension=log['output_dim'],macro_f1=log['scores']['macro_f1'],overall_accuracy=log['scores']['overall_accuracy'],balanced_accuracy=log['scores']['balanced_accuracy'],fit_key=log['fit_key']))
        # IDs and full preprocessing states were checked above and remain on disk.
        # Retain only fields needed for cross-record checks, avoiding several GB of repeated ID lists.
        fit_logs[log['fit_key']]={k:log[k] for k in ('fit_key','scores','output_dim','candidate','probability_file')}
        fit_logs[log['fit_key']]['feature_info']={'actual_env_factors':log['feature_info'].get('actual_env_factors',[])}
    ledger=pd.DataFrame(rows)
    expected_inner = 20 * (sum(1 + (s['block']=='FDR') for s in d.spectral) + 8*len(d.pool) + 8*(len(d.combos)+1))
    check('registered planned inner fit budget',int((ledger.inner_fold>0).sum())==expected_inner)
    check('registered planned outer fit budget',int((ledger.inner_fold==0).sum())==480)
    checkpoints=[]
    for policy,blocks in [('R',('OR','CR','FDR')),('R+V',('FDR',))]:
        for o in range(1,6):
            for b in blocks:
                for m in ('CARS','PCA','PLS-VIP','SPA'):
                    for f in FAMILIES:checkpoints.append(readj(checkpoint_path(['spectral',policy,o,b,m,f])))
            for b in ('N59','I63','FALL','FS'):
                for f in FAMILIES:checkpoints.append(readj(checkpoint_path(['fusion',policy,o,b,f])))
    check('all 480 family/outer checkpoints',len(checkpoints)==480)
    # Module selection and PI are audited separately from final classifier tuning.
    for policy in ('R','R+V'):
        for o in range(1,6):
            for family in FAMILIES:
                fs=readj(ROOT/'FS'/f'{tc.object_hash([policy,o,family])}.json')
                check(f'FS {policy}/{o}/{family} version',fs['run_identity_sha256']==ident)
                check(f'FS {policy}/{o}/{family} 32 subsets',
                      len(fs['module_records'])==32 and
                      sorted(r['combo_index'] for r in fs['module_records'])==list(range(32)))
                for module in fs['module_records']:
                    logs=[fit_logs[k] for k in module['inner_fit_keys']]
                    n=[len(l['feature_info']['actual_env_factors']) for l in logs]
                    rank=tc.selection_key([l['scores'] for l in logs],np.mean(n),str(module['combo_index']))
                    rank=(*rank[:4],module['combo_index'])
                    check(f'FS {policy}/{o}/{family}/{module["combo_index"]} module rank',
                          np.allclose(rank[:4],module['selection_key'][:4]) and rank[4]==module['selection_key'][4])
                check(f'FS {policy}/{o}/{family} module winner',
                      min(fs['module_records'],key=lambda r:tuple(r['selection_key']))==fs['selected_module'])
                selected=[]
                for pi in fs['permutation_records']:
                    a=np.asarray(pi['fold_repeat_deltas'],float);means=a.mean(axis=1);s=pi['summary']
                    keep=bool(means.mean()>.001 and (means>0).sum()>=2 and means.min()>-.01)
                    check(f'FS {policy}/{o}/{family}/{pi["factor"]} PI arithmetic',
                          a.shape==(4,3) and np.isfinite(a).all()
                          and np.allclose(means,s['fold_mean_delta'])
                          and np.isclose(means.mean(),s['mean_delta'])
                          and np.isclose(means.min(),s['min_delta'])
                          and int((means>0).sum())==s['positive_folds'] and keep==s['keep'])
                    for repeat in pi['repeat_records']:
                        check(f'FS {policy}/{o}/{family}/{pi["factor"]}/{repeat["inner_fold"]}/{repeat["repeat"]} PI seed/delta',
                              repeat['seed']==tc.permutation_seed(o,repeat['inner_fold'],family,pi['factor'],repeat['repeat'])
                              and np.isclose(repeat['baseline_macro_f1']-repeat['permuted_macro_f1'],repeat['delta_macro_f1'])
                              and np.isclose(a[repeat['inner_fold']-1,repeat['repeat']],repeat['delta_macro_f1']))
                    if keep:selected.append(pi['factor'])
                check(f'FS {policy}/{o}/{family} selected factor list',selected==fs['selected_factors'])
    for cp in checkpoints:
        for cand in cp['candidates']:
            logs=[fit_logs[k] for k in cand['inner_fit_keys']]
            check(cp['outer_fit_key']+cand['candidate']['candidate_id']+' candidate identity',
                  all(l['candidate']==cand['candidate'] for l in logs))
            key=tc.selection_key([l['scores'] for l in logs],np.mean([l['output_dim'] for l in logs]),cand['candidate']['candidate_id'])
            check(cp['outer_fit_key']+cand['candidate']['candidate_id']+' candidate rank',np.allclose(key[:4],cand['selection_key'][:4]) and key[4]==cand['selection_key'][4])
        expected=min(cp['candidates'],key=lambda c:tuple(c['selection_key']))
        check(cp['outer_fit_key']+' selected using inner only',expected['candidate']==cp['best']['candidate'])
    oofdir=ROOT/'results';oofdir.mkdir(exist_ok=True)
    allmetrics=[];selections=[]
    def assemble(name,policy,chosen):
        frames=[];foldmetrics=[]
        for cp in chosen:
            log=fit_logs[cp['outer_fit_key']]
            with np.load(ROOT/log['probability_file'],allow_pickle=False) as z:
                frame=pd.DataFrame(dict(sample_id=z['sample_id'],y_true=z['y_true'],y_pred=z['y_pred'],outer_fold=cp['outer_fold']))
                for j in range(6):frame[f'prob_{j}']=z['probabilities'][:,j]
                frame['candidate_id']=log['candidate']['candidate_id'];frame['family']=cp['family'];frame['policy']=policy;frame['branch']=name
            frames.append(frame);foldmetrics.append(dict(outer_fold=cp['outer_fold'],**metric(frame.y_true,frame.y_pred)))
            selections.append(dict(policy=policy,branch=name,outer_fold=cp['outer_fold'],family=cp['family'],candidate_id=cp['best']['candidate']['candidate_id'],inner_mean_macro_f1=cp['best']['mean_macro_f1'],outer_macro_f1=log['scores']['macro_f1'],dimensions=log['output_dim'],outer_fit_key=cp['outer_fit_key'],combo=cp.get('combo'),selected_factors=cp.get('selected_factors')))
        frame=pd.concat(frames,ignore_index=True)
        ordering={int(v):j for j,v in enumerate(d.metadata.loc[d.metadata.is_real==1,'sample_id'])}
        frame=frame.sort_values('sample_id',key=lambda s:s.map(ordering)).reset_index(drop=True)
        check(name+policy+' reference cohort ordered once OOF',len(frame)==len(ordering) and frame.sample_id.is_unique and set(frame.sample_id)==set(ordering) and np.array_equal(frame.y_true,d.metadata.set_index('sample_id').loc[frame.sample_id,'class_code_0based']))
        frame.to_csv(oofdir/f'{policy.replace("+", "")}_{name}_OOF.csv',index=False,encoding='utf-8-sig')
        stats=metric(frame.y_true,frame.y_pred)
        allmetrics.append(dict(policy=policy,branch=name,**stats,fold_metrics=foldmetrics))
    for policy in ('R','R+V'):
        spectralc=[cp for cp in checkpoints if cp['kind']=='spectral' and cp['policy']==policy and cp['block']=='FDR']
        chosen=[min([cp for cp in spectralc if cp['outer_fold']==o],key=lambda cp:tuple(cp['best']['selection_key'])) for o in range(1,6)]
        assemble('S0',policy,chosen)
        for branch in ('N59','I63','FALL','FS'):
            chosen=[min([cp for cp in checkpoints if cp['kind']=='fusion' and cp['policy']==policy and cp['branch']==branch and cp['outer_fold']==o],key=lambda cp:tuple(cp['best']['selection_key'])) for o in range(1,6)]
            assemble(branch,policy,chosen)
    for b in ('OR','CR','FDR'):
        for m in ('CARS','PCA','PLS-VIP','SPA'):
            for f in FAMILIES:
                chosen=[cp for cp in checkpoints if cp['kind']=='spectral' and cp['policy']=='R' and cp['block']==b and cp['reducer']==m and cp['family']==f]
                assemble(f'{b}_{m}_{f}','R',chosen)
    ledger.to_csv(oofdir/'fit_ledger.csv',index=False,encoding='utf-8-sig')
    writej(oofdir/'all_metrics.json',allmetrics);writej(oofdir/'selected_pipelines.json',selections)
    pd.DataFrame([{k:v for k,v in r.items() if k in ('policy','branch','macro_f1','overall_accuracy','balanced_accuracy','kappa')} for r in allmetrics]).to_csv(oofdir/'metric_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(ROOT/'qa/result_acceptance_checks.csv',index=False,encoding='utf-8-sig')
    writej(ROOT/'qa/result_acceptance.json',dict(status='PASS_FORMAL_RESULTS',checks=len(checks),failed=0,inner_classifier_fits=expected_inner,outer_refits=480,qa_classifier_fits=0,profile=MODE,recorded_resume_extra_fits=len(list((ROOT/'resume_extra_fits').glob('*.json'))),main_OOF_sets=10,descriptive_spectral_family_OOF_sets=48,all_OOF_n=int(d.metadata.is_real.sum()),run_identity_sha256=ident))
    print('FORMAL RESULTS PASS',len(checks),'checks',flush=True)

if __name__=='__main__':main()
