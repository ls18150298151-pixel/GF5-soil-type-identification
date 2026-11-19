"""Input loader and strict fold-local spectral/environment feature builders."""
import os
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[v]='1'
from pathlib import Path
import json
import hashlib
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_algorithms import FoldContext,FoldPCA,PLSVIPSelector,SPASelector,array_hash,native
from cars_adapter import Step07CARSSelector
from environment_cleaner import Layer1Cleaner
import training_contract as tc

from settings import ROOT, INPUT, RULES, PROTOCOL, protocol_options

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def readj(p):return json.loads(Path(p).read_text('utf-8'))
def writej(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(native(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,p)

def verify_identity():
    identity=readj(ROOT/'run_identity.json')
    for base, entries in ((ROOT, identity['run_files']), (PROTOCOL, identity['protocol_files']), (Path(__file__).resolve().parent, identity['code_files'])):
        for path,h in entries.items():
            if sha(base/path)!=h:raise RuntimeError('Run identity mismatch: '+path)
    return identity['identity_sha256']

class Inputs:
    def __init__(self,poison=None):
        self.metadata=pd.read_csv(INPUT/'metadata.csv',float_precision='round_trip')
        self.ids=self.metadata.sample_id.to_numpy()
        with np.load(INPUT/'modeling_inputs.npz',allow_pickle=False) as z:
            if not np.array_equal(self.ids,z['sample_id']):raise ValueError('Input ID alignment failed')
            self.blocks={b:pd.DataFrame(z[b],index=self.ids,columns=[f'{b}_B{k+1:03d}' for k in range(z[b].shape[1])]) for b in ('OR','CR','FDR')}
            self.env=pd.DataFrame(z['ENV63'],index=self.ids,columns=z['ENV63_feature_names'])
            self.wavelength=z['wavelength_nm'].copy()
        self.pool,self.spectral,self.methods=protocol_options()
        self.protocol=readj(PROTOCOL/'training_protocol.json')
        catalog=pd.read_csv(PROTOCOL/'factor_catalog_63.csv')
        self.modules={m:catalog.loc[catalog.module==m,'factor'].tolist() for m in ('T','E','H','C','L')}
        self.combos=readj(PROTOCOL/'module_combinations_32.json')['combinations']
        self.frames={}
        for level in ('outer','inner'):
            frame=pd.read_csv(RULES/f'spatial/{level}_manifest.csv',float_precision='round_trip')
            for (o,i),f in frame.groupby(['context_outer','context_inner'],sort=True):
                if not np.array_equal(f.sample_id,self.ids):raise ValueError('Manifest ID/order mismatch')
                self.frames[int(o),int(i)]=f.reset_index(drop=True)
        self.contexts={};self.guards={}
        for policy in ('R','R+V'):
            for (o,i),f in self.frames.items():
                guard=tc.TrainingLabels(self.metadata,f,policy)
                tr=f.set_index('sample_id').loc[list(guard.training_ids)]
                context=FoldContext(f'v2/{policy}/outer{o}/inner{i}',guard.training_ids,guard.validation_ids,tuple(tr.spatial_block_id.astype(str)),tuple(tr.leakage_group_id.astype(str)),tuple(tr.is_real.astype(bool)),'spatial_block_id')
                self.contexts[policy,o,i]=context;self.guards[policy,o,i]=guard
        if poison is not None:
            policy,o,i=poison;ctx=self.contexts[policy,o,i]
            denied=~self.metadata.sample_id.isin(ctx.training_ids)
            self.metadata.loc[denied,'class_code_0based']=-999
            # Replace labels outside training before rebuilding guards to make leakage visible.
            self.guards[policy,o,i]=tc.TrainingLabels(self.metadata,self.frames[o,i],policy)
            rng=np.random.default_rng(920)
            for b in self.blocks.values():b.loc[list(ctx.validation_ids)]=rng.normal(100,10,(len(ctx.validation_ids),b.shape[1]))
            self.env.loc[list(ctx.validation_ids)]=rng.normal(100,10,(len(ctx.validation_ids),63))
            self.env.loc[list(ctx.validation_ids),'LC_class']=999

    def rows(self,frame,ids,ctx):
        ids=tuple(int(v) for v in ids)
        if not set(ids)<=set(ctx.training_ids)|set(ctx.validation_ids):raise ValueError('Sealed/excluded feature IDs rejected')
        return frame.loc[list(ids)]

    def training_y(self,policy,o,i):
        g=self.guards[policy,o,i];return g.get(g.training_ids)

    def validation_y(self,policy,o,i,ids):
        ctx=self.contexts[policy,o,i]
        if tuple(ids)!=ctx.validation_ids:raise ValueError('Score only exact current validation IDs')
        return tc.valid_labels(self.metadata.set_index('sample_id').loc[list(ids),'class_code_0based'].to_numpy(),require_all=True)

def feature_seed(o,i,block,method,params):
    return int(tc.object_hash([tc.BASE_SEED,'feature',int(o),int(i),block,method,params])[:8],16)%(2**31-1)

def fit_spectral(data,policy,o,i,block,method,params):
    ctx=data.contexts[policy,o,i]
    p=dict(params)
    for k in ('svd_solver','minimum_features','internal_folds'):p.pop(k,None)
    seed=feature_seed(o,i,block,method,params)
    cls={'PCA':FoldPCA,'PLS-VIP':PLSVIPSelector,'SPA':SPASelector,'CARS':Step07CARSSelector}[method]
    transformer=cls(seed=seed,**p)
    X=data.rows(data.blocks[block],ctx.training_ids,ctx)
    Y=np.eye(6)[data.training_y(policy,o,i)] if transformer.supervised else None
    transformer.fit(X,Y,context=ctx)
    tr=transformer.transform(X).to_numpy()
    va=transformer.transform(data.rows(data.blocks[block],ctx.validation_ids,ctx)).to_numpy()
    scaler=StandardScaler().fit(tr)
    tr=scaler.transform(tr);va=scaler.transform(va)
    state=transformer.state_summary()
    info=dict(policy=policy,outer_fold=o,inner_fold=i,block=block,reducer=method,params=params,seed=seed,training_ids=ctx.training_ids,validation_ids=ctx.validation_ids,feature_names=list(transformer.feature_names_out_),output_dim=tr.shape[1],feature_state=state,feature_state_hash=transformer.state_hash(),classifier_scaler=dict(mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_),train_array_hash=array_hash(tr),validation_array_hash=array_hash(va),fit_label_scope='training only' if transformer.supervised else 'labels not accessed')
    if method=='CARS':info['cars_trace']=transformer.history_;info['cars_partitions']=[dict(train=t,validation=v,excluded=e) for t,v,e in transformer.partitions_]
    return np.ascontiguousarray(tr),np.ascontiguousarray(va),info

def spectral_path(policy,o,i,block,method,params):
    key=tc.object_hash([policy,o,i,block,method,params])
    return ROOT/'features'/key

def spectral(data,policy,o,i,block,method,params):
    path=spectral_path(policy,o,i,block,method,params)
    if not path.with_suffix('.npz').exists():raise RuntimeError('Missing preflight accepted feature cache: '+str(path))
    info=readj(path.with_suffix('.json'))
    if info['run_identity_sha256']!=readj(ROOT/'run_identity.json')['identity_sha256']:raise RuntimeError('Feature cache identity changed')
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:tr=z['train'].copy();va=z['validation'].copy()
    if array_hash(tr)!=info['train_array_hash'] or array_hash(va)!=info['validation_array_hash']:raise RuntimeError('Feature cache corrupted')
    ctx=data.contexts[policy,o,i]
    if tuple(info['training_ids'])!=ctx.training_ids or tuple(info['validation_ids'])!=ctx.validation_ids:raise RuntimeError('Feature boundary changed')
    return tr,va,info

class EnvironmentBuilder:
    def __init__(self,data,policy,o,i,spec_config=None):
        self.data=data;self.policy=policy;self.o=o;self.i=i;self.ctx=data.contexts[policy,o,i];self.spec_config=spec_config
        self.cache={}

    def build(self,combo,include_spectral,selected=None):
        combo=tuple(combo)
        if combo not in self.cache:
            cols=[c for m in combo for c in self.data.modules[m]]
            num=[c for c in cols if c!='LC_class']
            # Only allowed context rows are handed to the cleaner.
            env=self.data.rows(self.data.env,self.ctx.training_ids+self.ctx.validation_ids,self.ctx)
            cleaner=Layer1Cleaner(env,numeric_cols=num,categorical_cols=['LC_class'] if 'LC_class' in cols else []).fit(self.ctx.training_ids)
            self.cache[combo]=cleaner
        cleaner=self.cache[combo]
        factors=[c for c in cleaner.kept_numeric_ if selected is None or c in selected]
        parts_tr=[];parts_va=[];names=[];layout={};spec_info=None
        if include_spectral:
            if self.spec_config is None:raise ValueError('S requires current newly selected FDR backbone')
            cfg=self.spec_config
            tr,va,spec_info=spectral(self.data,self.policy,self.o,self.i,'FDR',cfg['reducer'],cfg['reducer_params'])
            parts_tr.append(tr);parts_va.append(va);names+=spec_info['feature_names']
        if factors:
            tr=cleaner.impute_numeric(self.ctx.training_ids,factors);va=cleaner.impute_numeric(self.ctx.validation_ids,factors)
            scaler=StandardScaler().fit(tr)
            start=len(names)
            for j,c in enumerate(factors):layout[c]=[start+j]
            names+=factors;parts_tr.append(scaler.transform(tr));parts_va.append(scaler.transform(va))
            scaler_info=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist())
        else:scaler_info={}
        if 'L' in combo and (selected is None or 'LC_class' in selected) and cleaner.lc_categories_:
            layout['LC_class']=list(range(len(names),len(names)+len(cleaner.lc_categories_)))
            names+=list(cleaner.lc_onehot_cols_);parts_tr.append(cleaner.onehot(self.ctx.training_ids));parts_va.append(cleaner.onehot(self.ctx.validation_ids))
        tr=np.column_stack(parts_tr) if parts_tr else np.empty((len(self.ctx.training_ids),0))
        va=np.column_stack(parts_va) if parts_va else np.empty((len(self.ctx.validation_ids),0))
        if min(tr.shape)==0 or not np.isfinite(tr).all() or not np.isfinite(va).all():raise ValueError('No finite feature matrix')
        info=dict(training_ids=self.ctx.training_ids,validation_ids=self.ctx.validation_ids,combo=list(combo),include_spectral=bool(include_spectral),requested_selected_factors=selected,actual_env_factors=list(layout),feature_names=names,output_dim=tr.shape[1],layout=layout,layer1=cleaner.state_summary(),layer1_hash=cleaner.state_hash(),environment_scaler=scaler_info,LC_unknown_validation=cleaner.lc_unknown_categories(self.ctx.validation_ids),spectral_state_hash=spec_info['feature_state_hash'] if spec_info else None)
        return tr,va,info
