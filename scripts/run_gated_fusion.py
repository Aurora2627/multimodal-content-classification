"""Prespecified SigLIP2 matched-MLP/gate comparison; launch from VS Code."""
import json, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    seeds=[42,43,44];shots=[0,2,4]
    prefix='gated-siglip2-'+time.strftime('%Y%m%d-%H%M%S')
    report=ROOT/'runs'/prefix;report.mkdir(parents=True,exist_ok=False)
    plan={'backend':'siglip2','seeds':seeds,'shots':shots,'gate_hidden':32,'classifier_hidden':128,
          'gate_initialization':'neutral 0.5, matched baseline classifier/RNG',
          'selection':'validation macro-F1 only; no test selection',
          'ablations':['fixed-neutral-gate','missing-text','missing-image','shuffled-image'],
          'scope':'frozen encoders; reused test/validation; exploratory; gate has extra parameters'}
    (report/'plan.json').write_text(json.dumps(plan,indent=2));results=[]
    for k in shots:
        for seed in seeds:
            output='runs/'+prefix+'-k'+str(k)+'-s'+str(seed)
            command=[sys.executable,str(ROOT/'scripts/train_torch.py'),'--backend','siglip2',
                     '--device','mps','--include-adapter','--include-gate','--shots',str(k),
                     '--seed',str(seed),'--output',output]
            with (report/('k'+str(k)+'-s'+str(seed)+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics=json.loads((ROOT/output/'metrics.json').read_text())
            if k and metrics['actual_training_size']!=6*k:raise RuntimeError('Incorrect K-shot count')
            results.append({'shots':k,'seed':seed,'output':output,'results':metrics['results'],
                            'actual_training_size':metrics['actual_training_size']})
            (report/'results.json').write_text(json.dumps(results,indent=2))
            print('GATE_COMPLETED',k,seed,flush=True)
    aggregate=[]
    for k in shots:
        for head in results[0]['results']:
            group=[r['results'][head] for r in results if r['shots']==k]
            variants={'intact':[r['test']['macro_f1'] for r in group]}
            for variant in group[0].get('ablations',{}):
                variants[variant]=[r['ablations'][variant]['macro_f1'] for r in group]
            for variant,values in variants.items():
                aggregate.append({'shots':k,'model':head,'variant':variant,'mean':statistics.mean(values),
                                  'sample_std':statistics.stdev(values),'seed_scores':dict(zip(map(str,seeds),values)),
                                  'trainable_parameters':group[0]['trainable_parameters']})
    (report/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    lines=['# SigLIP2 可学习模态门控实验','', '全量/每类 2/4 条训练样本，种子 42/43/44，均值±样本标准差。相同训练 ID、分类器初始化、优化器和早停协议。门控额外参数单列；替换模态为训练均值。打乱图片只测配对敏感性，不能证明事实核验能力。固定门控消融保留训练后的分类器，不能替代独立训练的 MLP 基线。','', '| 每类样本（0=全量） | 模型 | 评测条件 | Macro-F1 | 参数 |','|---:|---|---|---:|---:|']
    for r in aggregate:lines.append('| {shots} | {model} | {variant} | {mean:.4f} ± {sample_std:.4f} | {trainable_parameters} |'.format(**r))
    lines+=['','编码器冻结，测试集已用于多轮探索。三种子不足以断言统计显著性；305 条验证集对 12/24 条训练样本的选择影响较大。']
    (report/'summary.md').write_text('\n'.join(lines)+'\n')
    public=ROOT/'reports'/prefix
    public.mkdir(parents=True,exist_ok=False)
    for name in ['plan.json','aggregate.json','results.json','summary.md']:
        (public/name).write_bytes((report/name).read_bytes())
    print('GATED_EXPERIMENT_FINISHED',report,flush=True)
