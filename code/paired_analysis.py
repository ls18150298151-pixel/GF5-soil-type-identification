"""Step 4: paired OOF increments and conditional spatial-block bootstrap intervals."""
from pathlib import Path
import json
import hashlib
import itertools
import numpy as np
import pandas as pd

from settings import ROOT as TRAIN, RULES

ROOT=TRAIN/'analysis'
CLASSES=['红壤','紫色土','燥红土','黄棕壤','石灰岩土','黄壤']

def readj(p):return json.loads(Path(p).read_text('utf-8'))
def writej(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def hashfile(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def metrics_cm(cm):
    cm=np.asarray(cm,float);tp=np.diagonal(cm,axis1=-2,axis2=-1);actual=cm.sum(axis=-1);pred=cm.sum(axis=-2)
    f1=np.divide(2*tp,actual+pred,out=np.zeros_like(tp),where=actual+pred!=0)
    recall=np.divide(tp,actual,out=np.zeros_like(tp),where=actual!=0)
    return np.stack([f1.mean(axis=-1),recall.mean(axis=-1),tp.sum(axis=-1)/cm.sum(axis=(-2,-1))],axis=-1)

def main():
    gate=readj(TRAIN/'qa/result_acceptance.json')
    assert gate['status']=='PASS_FORMAL_RESULTS'
    ROOT.mkdir(exist_ok=True)
    groups=pd.read_csv(RULES/'spatial/sample_groups.csv').query('is_real==1').set_index('sample_id')
    ids=groups.index.to_numpy();blocks=sorted(groups.spatial_block_id.unique());bi={b:j for j,b in enumerate(blocks)}
    gi=np.array([bi[v] for v in groups.spatial_block_id])
    mainmetrics={};oofs={};cmblocks={}
    metrics=readj(TRAIN/'results/all_metrics.json')
    for policy in ('R','R+V'):
        for branch in ('S0','N59','I63','FALL','FS'):
            frame=pd.read_csv(TRAIN/f'results/{policy.replace("+", "")}_{branch}_OOF.csv').set_index('sample_id').loc[ids].reset_index()
            assert len(frame)==len(groups) and np.array_equal(frame.y_true,groups.class_code_0based)
            cm=np.zeros((len(blocks),6,6),dtype=np.int64)
            np.add.at(cm,(gi,frame.y_true,frame.y_pred),1)
            key=(policy,branch);cmblocks[key]=cm;oofs[key]=frame
            mainmetrics[key]=next(r for r in metrics if r['policy']==policy and r['branch']==branch)
    rng=np.random.default_rng(20261010)
    # Common resampled group counts for every paired comparison.
    weights=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=2000)
    boot={key:metrics_cm(np.einsum('rb,bij->rij',weights,cm,optimize=True)) for key,cm in cmblocks.items()}
    comparisons=[('GF5_indices',('R','I63'),('R','N59')),('FDR_backbone',('R','FALL'),('R','I63')),('all_GF5_increment',('R','FALL'),('R','N59')),('environment_increment',('R','FALL'),('R','S0')),('FS_minus_FALL',('R','FS'),('R','FALL'))]
    comparisons += [('virtual_'+branch,('R+V',branch),('R',branch)) for branch in ('S0','N59','I63','FALL','FS')]
    differences=[];perclass=[];folds=[];correctness=[]
    for name,a,b in comparisons:
        ma=mainmetrics[a];mb=mainmetrics[b];fa=oofs[a];fb=oofs[b]
        sampled=boot[a]-boot[b]
        for j,metric in enumerate(('macro_f1','balanced_accuracy','overall_accuracy')):
            low,high=np.quantile(sampled[:,j],[.025,.975])
            differences.append(dict(comparison=name,A=f'{a[0]}:{a[1]}',B=f'{b[0]}:{b[1]}',metric=metric,difference_pp=100*(ma[metric]-mb[metric]),conditional_block_bootstrap_low_pp=float(low*100),conditional_block_bootstrap_high_pp=float(high*100),resamples=2000,spatial_blocks=len(blocks)))
        for k,c in enumerate(CLASSES):
            perclass.append(dict(comparison=name,soil_class=c,precision_delta_pp=100*(ma['per_class_precision'][k]-mb['per_class_precision'][k]),recall_delta_pp=100*(ma['per_class_recall'][k]-mb['per_class_recall'][k]),f1_delta_pp=100*(ma['per_class_f1'][k]-mb['per_class_f1'][k])))
        for o in range(1,6):
            cm_a=np.zeros((6,6),int);cm_b=cm_a.copy();mask=fa.outer_fold.to_numpy()==o
            np.add.at(cm_a,(fa.y_true[mask],fa.y_pred[mask]),1);np.add.at(cm_b,(fb.y_true[mask],fb.y_pred[mask]),1)
            diff=100*(metrics_cm(cm_a)-metrics_cm(cm_b))
            folds.append(dict(comparison=name,outer_fold=o,n_validation=int(mask.sum()),macro_f1_delta_pp=float(diff[0]),balanced_accuracy_delta_pp=float(diff[1]),overall_accuracy_delta_pp=float(diff[2])))
        ca=fa.y_true.to_numpy()==fa.y_pred.to_numpy();cb=fb.y_true.to_numpy()==fb.y_pred.to_numpy()
        correctness.append(dict(comparison=name,both_correct=int((ca&cb).sum()),A_only_correct=int((ca&~cb).sum()),B_only_correct=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),prediction_changed=int((fa.y_pred.to_numpy()!=fb.y_pred.to_numpy()).sum()),net_correct_change=int(ca.sum()-cb.sum())))
    for name,data in [('paired_metric_differences',differences),('paired_per_class_differences',perclass),('paired_fold_differences',folds),('paired_correctness_changes',correctness)]:pd.DataFrame(data).to_csv(ROOT/f'{name}.csv',index=False,encoding='utf-8-sig')
    selections=readj(TRAIN/'results/selected_pipelines.json');fs_details=[];freq=[];stability=[];threshold_decisions=[]
    for policy in ('R','R+V'):
        lists=[]
        for r in selections:
            if r['policy']!=policy or r['branch']!='FS':continue
            log=readj(TRAIN/'fits'/f'{r["outer_fit_key"]}.json');info=log['feature_info']
            selected=r['selected_factors'];actual=info['actual_env_factors'];lists.append(set(actual))
            fs=readj(TRAIN/'FS'/f'{hashlib.sha256(json.dumps([policy,r["outer_fold"],r["family"]],ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode("utf-8")).hexdigest()}.json')
            fs_details.append(dict(policy=policy,outer_fold=r['outer_fold'],family=r['family'],modules='+'.join(r['combo']),selected_factor_list_n=len(selected),actual_env_factor_n=len(actual),actual_total_encoded_dimension=info['output_dim'],actual_spectral_dimension=info['output_dim']-sum(len(v) for v in info['layout'].values()),actual_env_encoded_columns=sum(len(v) for v in info['layout'].values())))
            for pi in fs['permutation_records']:
                s=pi['summary']
                for cutoff in (.0005,.001,.002):
                    keep=s['mean_delta']>cutoff and s['positive_folds']>=2 and s['min_delta']>-.01
                    threshold_decisions.append(dict(policy=policy,outer_fold=r['outer_fold'],family=r['family'],factor=pi['factor'],cutoff=cutoff,keep=int(keep),registered_main_cutoff=.001))
            for f in actual:freq.append(dict(policy=policy,outer_fold=r['outer_fold'],factor=f))
        values=[len(a&b)/len(a|b) if a|b else 1 for a,b in itertools.combinations(lists,2)]
        stability.append(dict(policy=policy,mean_pairwise_Jaccard=float(np.mean(values)),min_pairwise_Jaccard=float(np.min(values)),max_pairwise_Jaccard=float(np.max(values)),all_five_intersection_n=len(set.intersection(*lists)),five_fold_union_n=len(set.union(*lists)),five_actual_counts=[len(v) for v in lists]))
    pd.DataFrame(fs_details).to_csv(ROOT/'FS_actual_dimensions.csv',index=False,encoding='utf-8-sig')
    frequency=pd.DataFrame(freq).groupby(['policy','factor']).size().reset_index(name='selected_outer_fold_count')
    frequency.to_csv(ROOT/'FS_selection_frequency.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(threshold_decisions).to_csv(ROOT/'PI_threshold_subset_diagnostics.csv',index=False,encoding='utf-8-sig')
    writej(ROOT/'FS_stability.json',stability)
    # Three-block matched family comparisons describe the complete expression workflow, not the derivative alone.
    family_metrics=[r for r in metrics if r['policy']=='R' and r['branch'] not in ('S0','N59','I63','FALL','FS')]
    matched=[]
    for method in ('CARS','PCA','PLS-VIP','SPA'):
        for family in ('LightGBM','XGBoost','SVM','RandomForest'):
            rr={b:next(r for r in family_metrics if r['branch']==f'{b}_{method}_{family}') for b in ('OR','CR','FDR')}
            matched.append(dict(reducer=method,classifier=family,OR_macro_f1=rr['OR']['macro_f1'],CR_macro_f1=rr['CR']['macro_f1'],FDR_macro_f1=rr['FDR']['macro_f1'],FDR_minus_OR_pp=100*(rr['FDR']['macro_f1']-rr['OR']['macro_f1']),FDR_minus_CR_pp=100*(rr['FDR']['macro_f1']-rr['CR']['macro_f1'])))
    pd.DataFrame(matched).to_csv(ROOT/'spectral_matched_family_differences.csv',index=False,encoding='utf-8-sig')
    notes='''# 第4步结果分析口径\n\n仅在完整新OOF通过第3步验收后执行。本脚本使用固定空间块的成对OOF重采样，2000次、种子20261010。区间描述固定预测、现有标签及空间块抽样条件下的差值波动，不覆盖训练集重抽样或重新选模的不确定性；邻近块仍可能相关，不能作为独立外部验证。\n\n报告合并OOF与五个外层折差值、逐类P/R/F1以及预测变化数量，不以正确数量净差代表所有预测相同，不把包含零的区间当成等效性证据。\n\nFS记录选择清单和实际入模因子/编码列数的区别。阈值0.05/0.10/0.20个百分点表当前只检查子集变化，尚非重新训练后的性能敏感性。需要性能敏感性时单独登记并重新调参/拟合，不能沿用主阈值OOF。\n\nOR/CR/FDR对照评价完整光谱表达流程；不单独归因于求导。未进行空间缓冲增强时，结论限于县内当前分组留出任务，不声明距离更远区域泛化。\n'''
    (ROOT/'分析口径说明.md').write_text(notes,encoding='utf-8')
    writej(ROOT/'analysis_manifest.json',dict(status='ANALYSIS_EXECUTED_PENDING_QA',source_result_acceptance_sha256=hashfile(TRAIN/'qa/result_acceptance.json'),script_sha256=hashfile(__file__),seed=20261010,bootstrap_repeats=2000,n_spatial_blocks=len(blocks),n_nonvirtual=len(groups),comparison_names=[c[0] for c in comparisons],claim_decisions={'SG':'restrict to complete FDR workflow; no derivative-only causal attribution','buffer':'county blocked held-out task; no distant extrapolation claim','PI_cutoff':'subset diagnostics now, performance sensitivity separately if subsets change','virtual_labels':'paired training sensitivity does not validate label provenance'}))
    print('Step 4 analysis generated',len(differences),'paired metric differences',flush=True)
if __name__=='__main__':main()
