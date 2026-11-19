"""New metadata-only spatial nested CV. Never reads predictor matrices or fits models."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedGroupKFold

from settings import INPUT, ROOT, MODE

OUT = ROOT / 'spatial'
BASE_SEED = 20260915
OFFSETS = [(0, 0), (1000, 0), (0, 1000), (1000, 1000)]
CLASSES = ['红壤', '紫色土', '燥红土', '黄棕壤', '石灰岩土', '黄壤']

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def writej(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def csv(p, x):
    pd.DataFrame(x).to_csv(p, index=False, encoding='utf-8-sig', float_format='%.17g')

def splits(y, g, n, seed):
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y, g))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((INPUT / 'preparation.json').read_text(encoding='utf-8'))
    m = pd.read_csv(INPUT/'metadata.csv', float_precision='round_trip').fillna({'exact_coordinate_group_id':'', 'old_exact_coordinate_group_id':''})
    before = sha(INPUT/'metadata.csv')
    assert m.sample_id.is_unique
    real_mask = m.is_real.to_numpy()==1
    ri = np.flatnonzero(real_mask)
    if MODE == 'paper' and (len(ri), int(m.is_virtual.sum())) != (719, 52):
        raise ValueError('Paper profile expects the frozen cohort counts')
    y = m.soil_class_canonical.to_numpy()
    yc = m.class_code_0based.to_numpy()
    assert np.array_equal(np.array(CLASSES)[yc], y)
    totals = np.bincount(yc[ri], minlength=6)
    xy = m[['easting','northing']].to_numpy()
    d = cdist(xy[ri], xy[ri])
    candidates, feasible = [], []
    for ox, oy in OFFSETS:
        cell = np.floor((xy-[ox,oy])/2000).astype(np.int64)
        groups = np.array([f'B_{a}_{b}' for a,b in cell])
        for seed in range(BASE_SEED, BASE_SEED+50):
            partition = splits(y[ri], groups[ri], 5, seed)
            counts = np.array([np.bincount(yc[ri[v]], minlength=6) for t,v in partition])
            b = sum(Fraction(abs(5*int(n)-int(t)), int(t)) for row in counts for n,t in zip(row,totals))/30
            b += sum(Fraction(abs(5*int(n)-len(ri)),len(ri)) for n in counts.sum(axis=1))/20
            nearest = np.concatenate([d[np.ix_(v,t)].min(axis=1) for t,v in partition])
            median = float(np.median(nearest))
            ok = bool(counts.min()>=3)
            candidates.append(dict(offset_x_m=ox,offset_y_m=oy,seed=seed,feasible=int(ok),min_validation_class_n=int(counts.min()),balance_B=float(b),balance_exact=str(b),nearest_median_m=median,nearest_min_m=float(nearest.min()),selected=0))
            if ok: feasible.append((b,-median,ox,oy,seed,partition,groups))
        print(f'offset ({ox}, {oy}): 50 candidates', flush=True)
    assert feasible, 'No feasible prespecified outer partition'
    winner = min(feasible,key=lambda z:z[:5])
    b,negative_median,ox,oy,seed,partition,groups = winner
    for c in candidates: c['selected']=int((c['offset_x_m'],c['offset_y_m'],c['seed'])==(ox,oy,seed))
    m['spatial_block_id']=groups
    m['leakage_group_id']=np.where(m.exact_coordinate_group_id!='',m.exact_coordinate_group_id,'U_'+m.sample_id.astype(str))
    # Historical rounded pairs are audit fields, not asserted to be the same true XY.
    assert (m.groupby('leakage_group_id')[['easting','northing']].nunique()==1).all().all()
    m['outer_fold']=0
    outer_rows,inner_rows,contexts,inner_search,near_rows=[],[],[],[],[]

    def record(o,i,tr,va,sealed,virtual_allowed):
        f=m.copy()
        f['context_outer']=o; f['context_inner']=i
        val=np.zeros(len(m),dtype=bool);val[va]=True
        train_r=np.zeros(len(m),dtype=bool);train_r[tr]=True
        train_rv=train_r|virtual_allowed
        role=np.full(len(m),'excluded_virtual_by_policy_R',dtype=object)
        role[sealed]='sealed_outer_validation_real'
        role[tr]='train_real';role[va]='validation_real'
        role_rv=role.copy()
        role_rv[~real_mask]='excluded_spatial_holdout_virtual'
        role_rv[virtual_allowed]='train_virtual_only'
        if i:
            outer_held=outer_val[o]
            role_rv[(~real_mask)&np.isin(groups,groups[outer_held])]='sealed_outer_excluded_virtual'
        f['role_R']=role;f['role_RV']=role_rv
        f['is_training_R']=train_r.astype(int);f['is_training_RV']=train_rv.astype(int);f['is_validation']=val.astype(int)
        (outer_rows if i==0 else inner_rows).append(f)
        for policy,mask in [('R',train_r),('R+V',train_rv)]:
            training=np.flatnonzero(mask)
            assert len(training)>0 and len(va)>0
            assert not set(groups[training])&set(groups[va])
            assert not set(m.leakage_group_id.iloc[training])&set(m.leakage_group_id.iloc[va])
            assert not set(training)&set(sealed)
            assert not set(va)&set(sealed)
            assert np.bincount(yc[training],minlength=6).min()>0
            assert np.bincount(yc[va],minlength=6).min()>0
            counts_tr=np.bincount(yc[training],minlength=6)
            counts_va=np.bincount(yc[va],minlength=6)
            contexts.append(dict(outer_fold=o,inner_fold=i,policy=policy,n_train_real=len(tr),n_train_virtual=int((mask&~real_mask).sum()),n_validation_real=len(va),n_sealed_outer_real=len(sealed),n_train_blocks=len(set(groups[training])),n_validation_blocks=len(set(groups[va])),**{f'train_{k}':int(v) for k,v in enumerate(counts_tr)},**{f'validation_{k}':int(v) for k,v in enumerate(counts_va)}))
            dist=cdist(xy[va],xy[training]).min(axis=1)
            near_rows.extend(dict(outer_fold=o,inner_fold=i,policy=policy,sample_id=int(m.sample_id.iloc[idx]),nearest_training_m=float(v)) for idx,v in zip(va,dist))

    outer_val={o:ri[v] for o,(t,v) in enumerate(partition,1)}
    for o,va in outer_val.items(): m.loc[va,'outer_fold']=o
    for o,(t,v) in enumerate(partition,1):
        tr,va=ri[t],ri[v]
        outer_virtual=(~real_mask)&~np.isin(groups,groups[va])
        record(o,0,tr,va,np.array([],dtype=int),outer_virtual)
        # Preserve first original seed; if necessary try increasing seeds using labels/groups only.
        chosen=None
        for iseed in range(BASE_SEED+o,BASE_SEED+o+200):
            ip=splits(y[tr],groups[tr],4,iseed)
            minimum=min(int(np.bincount(yc[tr[iv]],minlength=6).min()) for it,iv in ip)
            train_min=min(int(np.bincount(yc[tr[it]],minlength=6).min()) for it,iv in ip)
            ok=minimum>=1 and train_min>=1
            inner_search.append(dict(outer_fold=o,seed=iseed,min_validation_class_n=minimum,min_training_class_n=train_min,feasible=int(ok),selected=int(ok)))
            if ok: chosen=(iseed,ip);break
        assert chosen is not None, f'No inner six-class partition for outer {o}'
        for i,(it,iv) in enumerate(chosen[1],1):
            itr,iva=tr[it],tr[iv]
            allowed=outer_virtual&~np.isin(groups,groups[iva])
            record(o,i,itr,iva,va,allowed)
    csv(OUT/'sample_groups.csv',m)
    csv(OUT/'outer_candidates_200.csv',candidates)
    csv(OUT/'inner_seed_search.csv',inner_search)
    csv(OUT/'outer_manifest.csv',pd.concat(outer_rows,ignore_index=True))
    csv(OUT/'inner_manifest.csv',pd.concat(inner_rows,ignore_index=True))
    csv(OUT/'context_counts.csv',contexts)
    csv(OUT/'nearest_training_distances.csv',near_rows)
    writej(OUT/'validation_design.json',dict(version='luxi_spatial_XY_v2_20261010',status='GENERATED_PENDING_INDEPENDENT_QA',input_metadata_sha256=before,block_size_m=2000,coordinate_crs=preparation['projected_crs'],class_order=CLASSES,n_nonvirtual=len(ri),n_virtual=int(m.is_virtual.sum()),outer_folds=5,inner_folds=4,outer_search=dict(offsets_m=OFFSETS,seeds=list(range(BASE_SEED,BASE_SEED+50)),n_candidates=200,n_feasible=len(feasible),minimum_outer_validation_per_class=3,selection_order=['rational balance_B ascending','pooled nearest real-training distance median descending','offset_x ascending','offset_y ascending','seed ascending'],predictor_or_classifier_result_access=False),selected=dict(offset_x_m=ox,offset_y_m=oy,seed=seed,balance_B=float(b),balance_exact=str(b),nearest_real_training_median_m=-negative_median),inner_search=dict(initial_seed='20260915+outer_fold',fallback='first increasing seed, at most 200 candidates, all train/validation six classes; labels/groups only',seeds={str(o):next(r['seed'] for r in inner_search if r['outer_fold']==o and r['selected']) for o in range(1,6)}),duplicate_policy='Exact projected XY groups are held in one spatial block',sample_policies={'R':'nonvirtual training and validation only','R+V':'same nonvirtual validation IDs; virtual training only outside BOTH outer and current inner held blocks'},classifier_fits=0,feature_transform_fits=0,random_10fold_auxiliary='not run; spatial CV is current primary protocol',distance_interpretation='2km block side does not imply 2km train-validation buffer'))
    assert sha(INPUT/'metadata.csv')==before
    print(json.dumps(dict(selected=[ox,oy,seed],feasible=len(feasible),contexts=25),ensure_ascii=False),flush=True)

if __name__=='__main__': main()
