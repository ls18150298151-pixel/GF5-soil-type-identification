"""Independent sensitivity QA and paired OOF assembly, without classifier fitting."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from settings import ROOT as MAIN, PROTOCOL

HERE = Path(__file__).resolve().parent
OUT = MAIN / 'pi_sensitivity'
from v2_features import readj, writej, sha, verify_identity, Inputs
import training_contract as tc
from formal_runner import FAMILIES

def main():
    parent = verify_identity()
    identity = readj(OUT / 'sensitivity_identity.json')
    assert identity['parent_run_identity_sha256'] == parent
    assert identity['script_sha256'] == sha(HERE / 'run_PI_sensitivity.py')
    assert identity['registration_sha256'] == sha(PROTOCOL / 'PI_sensitivity_registration.json')
    assert readj(OUT / 'execution_complete.json')['status'] == 'EXECUTION_COMPLETE_PENDING_QA'
    data = Inputs()
    checks = []
    def check(name, ok):
        checks.append(dict(check=name, passed=bool(ok)))
        if not ok:
            raise AssertionError(name)
    logs = {}
    for path in (OUT / 'fits').glob('*.json'):
        log = readj(path)
        context = data.contexts[log['policy'], log['outer_fold'], log['inner_fold']]
        check(path.stem + ' scope', log['policy'] == 'R' and
              tuple(log['training_ids']) == context.training_ids and
              tuple(log['validation_ids']) == context.validation_ids and
              log['run_identity_sha256'] == parent)
        check(path.stem + ' seed', log['seed'] == tc.boundary_seed(
            log['outer_fold'], log['inner_fold'], log['candidate']['candidate_id']))
        prediction = OUT / log['probability_file']
        check(path.stem + ' hash', sha(prediction) == log['prediction_sha256'])
        with np.load(prediction, allow_pickle=False) as z:
            check(path.stem + ' IDs/truth', tuple(z['sample_id']) == context.validation_ids and
                  np.array_equal(z['y_true'], data.validation_y('R', log['outer_fold'],
                                     log['inner_fold'], context.validation_ids)))
            prob = z['probabilities']
            check(path.stem + ' probabilities', prob.shape == (len(context.validation_ids), 6)
                  and np.isfinite(prob).all() and (prob >= 0).all()
                  and np.allclose(prob.sum(axis=1), 1)
                  and np.array_equal(prob.argmax(axis=1), z['y_pred']))
            check(path.stem + ' metric', np.isclose(log['scores']['macro_f1'],
                  f1_score(z['y_true'], z['y_pred'], labels=range(6), average='macro', zero_division=0)))
        logs[log['fit_key']] = log
    reports = []
    changed_inner = changed_outer = 0
    registration = readj(PROTOCOL / 'PI_sensitivity_registration.json')
    for cutoff in registration['sensitivity_cutoffs']:
        frames = []
        for outer in range(1, 6):
            records = []
            for family in FAMILIES:
                result = readj(OUT / 'checkpoints' / f'{tc.object_hash([cutoff, outer, family])}.json')
                source_fs = MAIN / 'FS' / f'{tc.object_hash(["R", outer, family])}.json'
                source_cp = MAIN / 'checkpoints' / f'{tc.object_hash(["fusion", "R", outer, "FS", family])}.json'
                fs = readj(source_fs)
                expected = sorted(p['factor'] for p in fs['permutation_records']
                                  if p['summary']['mean_delta'] > cutoff
                                  and p['summary']['positive_folds'] >= 2
                                  and p['summary']['min_delta'] > -.01)
                check(f'{cutoff}/{outer}/{family} subset', expected == result['selected_factors'])
                check(f'{cutoff}/{outer}/{family} source hashes', sha(source_fs) == result['primary_fs_sha256']
                      and sha(source_cp) == result['primary_checkpoint_sha256'])
                same = expected == sorted(fs['selected_factors'])
                check(f'{cutoff}/{outer}/{family} reuse decision', same == result['unchanged_subset'])
                if same:
                    cp = readj(source_cp)
                    check(f'{cutoff}/{outer}/{family} exact reuse', result['candidate_records'] == cp['candidates']
                          and result['outer_fit_key'] == cp['outer_fit_key'])
                else:
                    candidates = [c for c in data.pool if c['classifier'] == family]
                    check(f'{cutoff}/{outer}/{family} candidate coverage',
                          {r['candidate']['candidate_id'] for r in result['candidate_records']} ==
                          {c['candidate_id'] for c in candidates})
                    changed_inner += 4 * len(candidates)
                    changed_outer += 1
                    for candidate in result['candidate_records']:
                        fit = [logs[k] for k in candidate['inner_fit_keys']]
                        rank = tc.selection_key([f['scores'] for f in fit],
                                      np.mean([f['output_dim'] for f in fit]), candidate['candidate']['candidate_id'])
                        check(f'{cutoff}/{outer}/{family}/{candidate["candidate"]["candidate_id"]} rank',
                              np.allclose(rank[:4], candidate['selection_key'][:4])
                              and rank[4] == candidate['selection_key'][4])
                    check(f'{cutoff}/{outer}/{family} inner winner',
                          min(result['candidate_records'], key=lambda r: tuple(r['selection_key'])) == result['best'])
                records.append(result)
            best = min(records, key=lambda r: tuple(r['best']['selection_key']))
            origin = MAIN if best['unchanged_subset'] else OUT
            log = readj(origin / 'fits' / f'{best["outer_fit_key"]}.json')
            with np.load(origin / log['probability_file'], allow_pickle=False) as z:
                frame = pd.DataFrame(dict(sample_id=z['sample_id'], y_true=z['y_true'],
                                          y_pred=z['y_pred'], outer_fold=outer,
                                          cutoff=cutoff, family=best['family']))
                for k in range(6):
                    frame[f'prob_{k}'] = z['probabilities'][:, k]
                frames.append(frame)
        oof = pd.concat(frames, ignore_index=True).set_index('sample_id')
        order = data.metadata.loc[data.metadata.is_real == 1, 'sample_id'].to_numpy()
        check(f'{cutoff} coverage', oof.index.is_unique and len(oof) == len(order) and set(oof.index) == set(order))
        oof = oof.loc[order].reset_index()
        oof.to_csv(OUT / f'R_FS_cutoff_{cutoff}_OOF.csv', index=False, encoding='utf-8-sig')
        baseline = pd.read_csv(MAIN / 'results/R_FS_OOF.csv').set_index('sample_id').loc[order]
        reports.append(dict(cutoff=cutoff, macro_f1=float(f1_score(oof.y_true, oof.y_pred,
                         labels=range(6), average='macro', zero_division=0)),
                         balanced_accuracy=float(balanced_accuracy_score(oof.y_true, oof.y_pred)),
                         overall_accuracy=float(accuracy_score(oof.y_true, oof.y_pred)),
                         changed_prediction_count=int((oof.y_pred.to_numpy() != baseline.y_pred.to_numpy()).sum())))
    check('exact additional inner budget', sum(l['inner_fold'] > 0 for l in logs.values()) == changed_inner)
    check('exact additional outer budget', sum(l['inner_fold'] == 0 for l in logs.values()) == changed_outer)
    check('registered maximum budget', changed_inner <= 1440 and changed_outer <= 40)
    pd.DataFrame(reports).to_csv(OUT / 'threshold_performance_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(OUT / 'acceptance_checks.csv', index=False, encoding='utf-8-sig')
    writej(OUT / 'acceptance.json', dict(status='PASS_PI_SENSITIVITY', checks=len(checks),
           failed=0, additional_inner_fits=changed_inner, additional_outer_fits=changed_outer,
           thresholds=registration['sensitivity_cutoffs'], parent_run_identity_sha256=parent))
    print('PI sensitivity PASS', len(checks), 'checks', flush=True)

if __name__ == '__main__':
    main()
