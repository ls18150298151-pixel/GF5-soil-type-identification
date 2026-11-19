"""Registered finite FS cutoff sensitivity; a separate ledger from main training."""
import sys
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from settings import ROOT as MAIN, PROTOCOL

HERE = Path(__file__).resolve().parent
OUT = MAIN / 'pi_sensitivity'
import formal_runner as fr
import training_contract as tc
from v2_features import Inputs, EnvironmentBuilder, readj, writej, sha, verify_identity

def initialize():
    verify_identity()
    fr.DATA = Inputs()
    # Feature caches remain in MAIN via v2_features. Only new fit/model ledger is redirected.
    fr.ROOT = OUT

def family_task(task):
    cutoff, outer, family = task
    fs_path = MAIN / 'FS' / f'{tc.object_hash(["R", outer, family])}.json'
    primary_path = MAIN / 'checkpoints' / f'{tc.object_hash(["fusion", "R", outer, "FS", family])}.json'
    fs = readj(fs_path)
    primary = readj(primary_path)
    selected = sorted(p['factor'] for p in fs['permutation_records']
                      if p['summary']['mean_delta'] > cutoff
                      and p['summary']['positive_folds'] >= 2
                      and p['summary']['min_delta'] > -.01)
    path = OUT / 'checkpoints' / f'{tc.object_hash(list(task))}.json'
    if path.exists():
        return readj(path)
    unchanged = selected == sorted(fs['selected_factors'])
    evidence = dict(cutoff=cutoff, outer_fold=outer, family=family,
                    primary_fs_sha256=sha(fs_path), primary_checkpoint_sha256=sha(primary_path),
                    selected_factors=selected, unchanged_subset=unchanged,
                    parent_run_identity_sha256=verify_identity(),
                    sensitivity_registration_sha256=sha(PROTOCOL / 'PI_sensitivity_registration.json'))
    if unchanged:
        evidence.update(best=primary['best'], outer_fit_key=primary['outer_fit_key'],
                        outer_scores=primary['outer_scores'], prediction_origin='MAIN',
                        candidate_records=primary['candidates'])
        writej(path, evidence)
        return evidence
    backbone = fs['backbone']
    combo = fs['selected_module']['combo']
    folds = []
    for inner in range(1, 5):
        builder = EnvironmentBuilder(fr.DATA, 'R', outer, inner, backbone)
        tr, va, info = builder.build(combo, True, selected=selected)
        folds.append((inner, tr, va, info))
    records = []
    for candidate in fr.DATA.pool:
        if candidate['classifier'] != family:
            continue
        logs = []
        for inner, tr, va, info in folds:
            log, _ = fr.fit_predict('R', outer, inner, candidate, tr, va, info,
                                    ['PI_sensitivity', cutoff, family])
            logs.append(log)
        rank = tc.selection_key([l['scores'] for l in logs],
                                np.mean([l['output_dim'] for l in logs]), candidate['candidate_id'])
        records.append(dict(candidate=candidate, selection_key=list(rank),
                            inner_fit_keys=[l['fit_key'] for l in logs],
                            inner_scores=[l['scores'] for l in logs],
                            mean_macro_f1=-rank[0]))
    best = min(records, key=lambda r: tuple(r['selection_key']))
    builder = EnvironmentBuilder(fr.DATA, 'R', outer, 0, backbone)
    tr, va, info = builder.build(combo, True, selected=selected)
    log, _ = fr.fit_predict('R', outer, 0, best['candidate'], tr, va, info,
                            ['PI_sensitivity', cutoff, family], save_model=True)
    evidence.update(best=best, outer_fit_key=log['fit_key'], outer_scores=log['scores'],
                    prediction_origin='SENSITIVITY', candidate_records=records)
    writej(path, evidence)
    return evidence

def main(workers):
    gate = readj(MAIN / 'qa/result_acceptance.json')
    assert gate['status'] == 'PASS_FORMAL_RESULTS'
    verify_identity()
    registration = readj(PROTOCOL / 'PI_sensitivity_registration.json')
    identity = dict(parent_run_identity_sha256=gate['run_identity_sha256'],
                    registration_sha256=sha(PROTOCOL / 'PI_sensitivity_registration.json'),
                    script_sha256=sha(__file__))
    OUT.mkdir(exist_ok=True)
    writej(OUT / 'run_identity.json', readj(MAIN / 'run_identity.json'))
    identity_path = OUT / 'sensitivity_identity.json'
    if identity_path.exists():
        assert readj(identity_path) == identity
    else:
        writej(identity_path, identity)
    tasks = [(c, o, f) for c in registration['sensitivity_cutoffs']
             for o in range(1, 6) for f in fr.FAMILIES]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
        futures = {pool.submit(family_task, t): t for t in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f'PI sensitivity: {done}/40 families', flush=True)
    writej(OUT / 'execution_complete.json', dict(status='EXECUTION_COMPLETE_PENDING_QA',
                                                 **identity))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    main(parser.parse_args().workers)
