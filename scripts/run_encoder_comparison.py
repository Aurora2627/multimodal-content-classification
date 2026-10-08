"""Prespecified three-seed full-data baseline/adapter comparison; run in VS Code."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--backends",nargs="+",choices=["clip","siglip2"],default=["clip","siglip2"])
    args=parser.parse_args()
    prefix = "encoder-adapter-" + time.strftime("%Y%m%d-%H%M%S")
    plan = {"seeds":SEEDS,"backends":args.backends,"protocol":"All candidates reported; selection by validation; full data, not K-shot"}
    report_dir = ROOT / "runs" / prefix
    report_dir.mkdir(parents=True, exist_ok=False)
    (report_dir / "plan.json").write_text(json.dumps(plan, indent=2))
    results = []
    for backend in plan["backends"]:
        if backend == "siglip2":
            deadline = time.monotonic() + 900
            while not (ROOT / "reports/siglip2-download.json").exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("Official SigLIP2 weights not ready after 15 minutes; CLIP results preserved")
                print("Waiting for official SigLIP2 download", flush=True)
                time.sleep(30)
        for seed in SEEDS:
            output = "runs/" + prefix + "-" + backend + "-s" + str(seed)
            command = [sys.executable, str(ROOT / "scripts/train_torch.py"), "--backend",backend,
                       "--device","mps","--include-adapter","--seed",str(seed),"--output",output]
            with (report_dir / (backend + "-s" + str(seed) + ".log")).open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            metrics = json.loads((ROOT / output / "metrics.json").read_text())
            results.append({"backend":backend,"seed":seed,"output":output,"results":metrics["results"]})
            (report_dir / "results.json").write_text(json.dumps(results, indent=2))
            print("COMPARISON_COMPLETED", backend, seed, flush=True)
    import statistics
    lines = ["# 冻结编码器 baseline 与特征 Adapter：全量训练三种子对比", "",
             "均值±样本标准差；种子 42/43/44。既有测试集的探索结果，不是严格少样本实验。", "",
             "| 编码器 | 分类方式 | 测试 Macro-F1 |", "|---|---|---:|"]
    for backend in plan["backends"]:
        for head in results[0]["results"]:
            values = [r["results"][head]["test"]["macro_f1"] for r in results if r["backend"]==backend]
            lines.append("| %s | %s | %.4f ± %.4f |" % (backend,head,statistics.mean(values),statistics.stdev(values)))
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("COMPARISON_FINISHED", report_dir, flush=True)
