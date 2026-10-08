"""Fixed-seed strict 2/4-shot baseline and feature Adapter comparisons in VS Code."""
import argparse, json, subprocess, sys, time, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['clip','siglip2'],required=True)
    args=p.parse_args();seeds=[42,43,44];shots=[2,4]
    prefix='strict-adapter-'+args.backend+'-'+time.strftime('%Y%m%d-%H%M%S')
    report=ROOT/'runs'/prefix;report.mkdir(parents=True,exist_ok=False)
    plan={'backend':args.backend,'seeds':seeds,'shots':shots,'standardization':'selected training examples only','selection':'validation macro-F1 only','scope':'reused validation/test; exploratory'}
    (report/'plan.json').write_text(json.dumps(plan,indent=2));results=[]
    for k in shots:
        for seed in seeds:
            output='runs/'+prefix+'-k'+str(k)+'-s'+str(seed)
            command=[sys.executable,str(ROOT/'scripts/train_torch.py'),'--backend',args.backend,'--device','mps','--include-adapter','--shots',str(k),'--seed',str(seed),'--output',output]
            with (report/('k'+str(k)+'-s'+str(seed)+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            metrics=json.loads((ROOT/output/'metrics.json').read_text())
            if metrics['actual_training_size']!=6*k:raise RuntimeError('Incorrect K-shot sample count')
            results.append({'shots':k,'seed':seed,'output':output,'results':metrics['results']})
            (report/'results.json').write_text(json.dumps(results,indent=2))
            print('FEWSHOT_COMPLETED',k,seed,flush=True)
    lines=['# '+args.backend+' 严格少样本与特征 Adapter','', '每类 2/4 条训练样本，种子 42/43/44；均值±样本标准差。相同种子下所有模型使用相同训练 ID；标准化只拟合抽样训练数据。验证集保留 305 条，测试集 304 条，属于有较大验证集的探索协议。','', '| 每类样本 | 模型 | 测试 Macro-F1 |','|---:|---|---:|']
    for k in shots:
        for head in results[0]['results']:
            values=[r['results'][head]['test']['macro_f1'] for r in results if r['shots']==k]
            lines.append('| %d | %s | %.4f ± %.4f |'%(k,head,statistics.mean(values),statistics.stdev(values)))
    (report/'summary.md').write_text('\n'.join(lines)+'\n');print('FEWSHOT_FINISHED',report,flush=True)
