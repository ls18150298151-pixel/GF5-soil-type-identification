"""Shared revised training rules. Imported by the future v2 runner; no import-time fits."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE_SEED=20260915
LABELS=np.arange(6,dtype=int)
ROOT=Path(__file__).resolve().parents[1]

def object_hash(v):
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()

def boundary_seed(outer,inner,candidate_id):
    if outer not in range(1,6) or inner not in range(5): raise ValueError('Invalid actual outer/inner boundary')
    return int(object_hash([BASE_SEED,int(outer),int(inner),str(candidate_id)])[:8],16)%(2**31-1)

def permutation_seed(outer,inner,family,factor,repeat):
    if inner not in range(1,5) or repeat not in range(3): raise ValueError('Invalid permutation boundary/repeat')
    return int(object_hash([BASE_SEED,'PI',int(outer),int(inner),str(family),str(factor),int(repeat)])[:8],16)%(2**31-1)

def valid_labels(y,require_all=False):
    a=np.asarray(y)
    if a.ndim!=1 or len(a)==0 or not np.isfinite(a).all() or not np.isin(a,LABELS).all(): raise ValueError('Expected nonempty labels 0..5')
    a=a.astype(int)
    if require_all and not np.array_equal(np.unique(a),LABELS): raise ValueError('All six training classes required')
    return a

class TrainingLabels:
    """Only explicitly registered current training IDs may supply fit labels."""
    def __init__(self,metadata,manifest,policy):
        if policy not in ('R','R+V'): raise ValueError('Unknown sample policy')
        if not metadata.sample_id.is_unique or not manifest.sample_id.is_unique: raise ValueError('Duplicate IDs')
        if not np.array_equal(metadata.sample_id,manifest.sample_id): raise ValueError('Metadata/manifest order mismatch')
        column='is_training_R' if policy=='R' else 'is_training_RV'
        self.training_ids=tuple(int(v) for v in manifest.loc[manifest[column]==1,'sample_id'])
        self.validation_ids=tuple(int(v) for v in manifest.loc[manifest.is_validation==1,'sample_id'])
        self._train=dict(zip(self.training_ids,metadata.set_index('sample_id').loc[list(self.training_ids),'class_code_0based'].astype(int)))
    def get(self,ids):
        ids=tuple(int(v) for v in ids)
        if ids!=self.training_ids: raise ValueError('Fit labels require exact ordered current training IDs; validation/sealed IDs forbidden')
        return valid_labels([self._train[v] for v in ids],require_all=True)

def balanced_training_weights(y):
    y=valid_labels(y,require_all=True)
    n=np.bincount(y,minlength=6)
    return np.asarray(len(y)/(6*n[y]),dtype=float)

FIXED={
    'RandomForest':dict(bootstrap=True,class_weight=None,n_jobs=1),
    'SVM':dict(kernel='rbf',probability=True,decision_function_shape='ovr',class_weight=None),
    'XGBoost':dict(objective='multi:softprob',num_class=6,eval_metric='mlogloss',tree_method='hist',n_jobs=1),
    'LightGBM':dict(objective='multiclass',num_class=6,subsample_freq=1,deterministic=True,force_col_wise=True,num_threads=1,verbosity=-1,class_weight=None),
}

def classifier_kwargs(candidate,outer,inner):
    name=candidate['classifier'];strategy=candidate['strategy']
    if name not in FIXED or strategy not in ('none','class_weight'): raise ValueError('Unregistered classifier/strategy; ordinary SMOTE prohibited')
    p={**FIXED[name],**candidate['params'],'random_state':boundary_seed(outer,inner,candidate['candidate_id'])}
    if strategy=='class_weight' and name!='XGBoost': p['class_weight']='balanced'
    if 'scale_pos_weight' in p: raise ValueError('Multiclass XGBoost weight must use current fit sample_weight')
    return p

def make_classifier(candidate,outer,inner):
    # Lazy imports: contract QA can use recording doubles without model training.
    name=candidate['classifier'];p=classifier_kwargs(candidate,outer,inner)
    if name=='LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**p)
    if name=='XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(**p)
    if name=='SVM':
        from sklearn.svm import SVC
        return SVC(**p)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**p)

def fit_current_training(classifier,candidate,X,training_labels,training_ids):
    y=training_labels.get(training_ids)
    if np.asarray(X).ndim!=2 or len(X)!=len(y) or not np.isfinite(X).all(): raise ValueError('Finite current training feature matrix required')
    kwargs={}
    if candidate['classifier']=='XGBoost' and candidate['strategy']=='class_weight': kwargs['sample_weight']=balanced_training_weights(y)
    classifier.fit(X,y,**kwargs)
    return classifier

def align_probability_argmax(probabilities,classes):
    classes=np.asarray(classes);p=np.asarray(probabilities,dtype=float)
    if classes.ndim!=1 or not np.array_equal(np.sort(classes),LABELS): raise ValueError('Fitted probability classes must be exactly 0..5')
    if p.ndim!=2 or p.shape[1]!=6 or not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise ValueError('Invalid probabilities')
    if not np.allclose(p.sum(axis=1),1,rtol=1e-7,atol=1e-8): raise ValueError('Probability row sums invalid')
    aligned=p[:,[int(np.flatnonzero(classes==c)[0]) for c in LABELS]]
    # Normalize only numeric roundoff; grossly invalid probabilities were rejected above.
    aligned=aligned/aligned.sum(axis=1,keepdims=True)
    return aligned,np.argmax(aligned,axis=1).astype(int)

def predict_current_validation(classifier,X):
    # A single decision path for SVM and the three tree classifiers.
    return align_probability_argmax(classifier.predict_proba(X),classifier.classes_)

def score_predictions(y_true,y_pred):
    truth=valid_labels(y_true);pred=valid_labels(y_pred)
    if len(truth)!=len(pred): raise ValueError('Prediction count mismatch')
    cm=np.zeros((6,6),dtype=int)
    np.add.at(cm,(truth,pred),1)
    tp=np.diag(cm);den=cm.sum(axis=0)+cm.sum(axis=1)
    f1=np.divide(2*tp,den,out=np.zeros(6,dtype=float),where=den!=0)
    recall=np.divide(tp,cm.sum(axis=1),out=np.zeros(6,dtype=float),where=cm.sum(axis=1)!=0)
    return dict(macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean()),overall_accuracy=float(tp.sum()/len(truth)),confusion_matrix=cm.tolist(),per_class_f1=f1.tolist(),per_class_recall=recall.tolist())

def selection_key(fold_scores,mean_dimension,candidate_id):
    if len(fold_scores)!=4: raise ValueError('Exactly four actual inner scores required')
    f1=np.array([s['macro_f1'] for s in fold_scores]);ba=np.array([s['balanced_accuracy'] for s in fold_scores])
    if not np.isfinite(f1).all() or not np.isfinite(ba).all(): raise ValueError('Missing/failed candidate fold; never silently average surviving folds')
    return (-float(f1.mean()),-float(ba.mean()),float(f1.std(ddof=0)),float(mean_dimension),str(candidate_id))

def permutation_summary(fold_repeat_deltas,available):
    a=np.asarray(fold_repeat_deltas,dtype=float);present=np.asarray(available,dtype=bool)
    if a.shape!=(4,3) or present.shape!=(4,) or not np.isfinite(a).all(): raise ValueError('PI requires four folds by three repeats')
    if np.any(a[~present]!=0): raise ValueError('Dropped factor contributes zero in each unavailable fold')
    means=a.mean(axis=1)
    return dict(fold_mean_delta=means.tolist(),mean_delta=float(means.mean()),min_delta=float(means.min()),positive_folds=int((means>0).sum()),available_folds=int(present.sum()),keep=bool(means.mean()>0.001 and (means>0).sum()>=2 and means.min()>-0.01))

def grouped_permute(X,columns,outer,inner,family,factor,repeat):
    if not columns: raise ValueError('Factor must have at least one current encoded column')
    a=np.asarray(X,dtype=float).copy()
    order=np.random.default_rng(permutation_seed(outer,inner,family,factor,repeat)).permutation(len(a))
    a[:,columns]=a[order][:,columns]
    return a
