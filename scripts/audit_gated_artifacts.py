"""Verify saved artifacts and reproduction of previous baselines; VS Code entry."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',default='reports/gated-fusion-v0.6.0')
    args=parser.parse_args();public=ROOT/args.report
    results=json.loads((public/'results.json').read_text());checks=[]
    prior={0:ROOT/'runs/encoder-adapter-20261008-175751/results.json',
           2:ROOT/'runs/strict-adapter-siglip2-20261008-175825/results.json',
           4:ROOT/'runs/strict-adapter-siglip2-20261008-175825/results.json'}
    for row in results:
        run=ROOT/row['output'];manifest=json.loads((run/'source-manifest.json').read_text())
        for name,digest in manifest['source_sha256'].items():
            if sha(run/'source_snapshot'/name)!=digest:raise RuntimeError('Source snapshot mismatch: '+name)
        for name,digest in manifest['input_sha256'].items():
            if sha(run/'input_manifests'/Path(name).name)!=digest:raise RuntimeError('Input snapshot mismatch')
        for name in ['environment.json','experiment-config.json','training-log.jsonl','training-ids.json','metrics.json']:
            if not (run/name).is_file():raise RuntimeError('Missing artifact '+name)
        ids=json.loads((run/'training-ids.json').read_text())
        if len(ids)!=row['actual_training_size'] or len(ids)!=len(set(ids)):raise RuntimeError('Training IDs mismatch')
        previous=json.loads(prior[row['shots']].read_text())
        # Full-data and K-shot wrappers use seed and, where present, shots.
        matching=[r for r in previous if r['seed']==row['seed'] and r.get('shots',0)==row['shots']]
        if len(matching)!=1:raise RuntimeError('Previous result not uniquely matched')
        for name,res in row['results'].items():
            for suffix in ['.pt','-predictions.jsonl']:
                if not (run/(name+suffix)).is_file():raise RuntimeError('Missing model artifact')
            if res['parameter_device']!='mps:0' or res['checkpoint_same_device_exact']!='passed':raise RuntimeError('MPS/reload check failed')
            if name!='fusion-gated-mlp':
                before=matching[0]['results'][name]['test']['macro_f1']
                if abs(before-res['test']['macro_f1'])>1e-12:raise RuntimeError('Baseline reproduction differs')
        checks.append({'run':row['output'],'source_hashes':len(manifest['source_sha256']),
                       'input_hashes':len(manifest['input_sha256']),'training_ids':len(ids),
                       'checkpoints':len(row['results']),'device':'mps:0','baseline_reproduction':'exact macro-F1'})
    report={'status':'passed','runs':len(checks),'checkpoints':sum(r['checkpoints'] for r in checks),'checks':checks}
    (public/'artifact-audit.json').write_text(json.dumps(report,indent=2));print('ARTIFACT_AUDIT_PASSED',len(checks))
