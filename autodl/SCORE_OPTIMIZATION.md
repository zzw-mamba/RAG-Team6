# 低分排查与基线优化

针对基础 RAG 大量拒答、检索缺失多跳证据的情况，新增可独立验证的优化入口。
实际竞赛结果与预测文件不随代码上传；新增策略的收益必须用真实模型评测。

作业 PDF 要求 Qwen2.5-7B 回答、LLaMA-3.1-8B 批判、RL 和三组对比，没有要求每个答案必须完全由当前检索文本证明。历史 REPORT.md 的 evidence_v2 使用混合检索、BGE、百科补全及模型知识，不能等同于当前 AutoDL 的小型 BM25/严格拒答实现，也不能拿历史成绩当本次成绩。

## 本次修复

- 原 allow_model_knowledge 只替换了一个句子，仍保留“任一关系缺失就 unknown”的要求。现在知识补全使用独立提示词，明确区分文本证据与 prior knowledge；不编造引用。默认严格模式保留。
- coverage 通道优先保留公开 ctxs（直至 max_documents），再交错加入每个查询的独立排名。第二跳不再因缺少原问题词语而被全部重排到尾部。新增词项 AND 通道，仍保留 OR 召回；补充资料每标题最多三个段落。
- 优化配置使用 16,384 输入 tokens、最多 32 篇、每篇最多 4,000 字符；实际超预算仍按现有预算器压缩，记录输入/选择/可见文档数量。它不会保证保留无限数量的全部文本。
- 可选 Qwen 查询规划：最多三条实体/缺失关系查询，最多两轮，JSON 约束；默认优化 planned 配置仅一轮。规划仅接收公开问题与文档，不接收标准答案、rationale 或当前猜测。格式失败保留真实记录并停止该轮规划，不伪造查询。

## 四组真实对照

`baseline` 是当前配置上的 legacy 检索与严格回答；`coverage` 仅改证据保留/预算；`knowledge` 再允许事实知识补全；`planned` 再启用一轮定向检索规划。
注意：baseline 是本次源码下的重跑对照，不承诺逐字重现旧版本的输出或成绩。

仅 tune 分数用于选择：normalized_em 优先，平局依次比较 token_f1 和推理耗时。
独立 confirmation 仅比较已冻结选择与 baseline；不使用 confirmation 重选配置。
若 confirmation EM 低于 baseline，predict 拒绝启动。未下降也不代表统计显著提高，必须阅读差值、题数及失败案例；本地 EM 不是官方 Kaggle Accuracy。
每个阶段都保留完整真实轨迹、manifest、指标与 unknown/文档覆盖诊断。
“标准答案字符串在可见文档中”仅作诊断代理，不等于完整推理链存在。
预测使用独立 optimization/score-v2 目录，保留原来的提交，不覆盖或混接旧结果。

## 新服务器运行

代码分支 `feat/rag-score-v2`。在 `/root/autodl-fs/zzw/RAG-Team6-fixed` 获取并切换分支后：

```bash
cd /root/autodl-fs/zzw/RAG-Team6-fixed/autodl
mkdir -p /root/autodl-fs/zzw/logs
nohup bash run_score_v2.sh tune /root/autodl-fs/zzw/configs/new-server.json \
  > /root/autodl-fs/zzw/logs/score-v2-tune.log 2>&1 < /dev/null &
```

入口优先使用项目 .venv，否则复用 `/root/autodl-tmp/zzw/RAG-Team6/autodl/.venv/bin/python`。
可用 `RAG_PYTHON=/absolute/path/to/python` 指定其他已装好环境，不移动旧 venv 或基础 Python。
TMP、缓存均放在 work_dir 的父目录，仍需确认磁盘有可用空间。
无需重装 Torch、下载模型或重建已有 prepared/corpus；训练标签只用于推理之后的离线指标。

阅读 `rag-team6-work/optimization/score-v2/selection.json` 后，依次运行：

```bash
bash run_score_v2.sh confirmation /root/autodl-fs/zzw/configs/new-server.json
bash run_score_v2.sh predict /root/autodl-fs/zzw/configs/new-server.json
```

也可用 `all` 顺序执行三个阶段，通过检查后才预测。中断后重跑同一阶段，签名一致时续跑；修改代码/配置后改用 `SCORE_RUN_NAME=score-v2-new`，不能复用旧签名结果。

### 全量预测并行提速

需要只加载一份 Qwen 权重时，使用 [单模型批量入口](BATCHED_INFERENCE.md)，先通过真实配对 pilot 后再续跑；下面保留多进程方案用于对照。

`run_score_v2.sh` 的 predict 阶段默认使用两个独立 Qwen worker，不在父进程多加载一个 Qwen；tune/confirmation 保持原有逻辑。`SCORE_WORKERS=1` 可恢复顺序运行，96GB 显存环境可以先测 `SCORE_WORKERS=4`。并行可能改变浮点执行细节，不保证逐字相同输出，也不预先承诺速度倍率。

本次提速仅修改包外调度脚本，推理包源码、配置、模型和输出签名不变，已完成的 tune/confirmation 和 test 前缀可继续使用；不会因为换 worker 数而强制重跑。父进程顺序写出，子进程仅接收公开字段。

在前两个阶段已完成后，先用下一批 32 道未完成问题进行独立测速：

```bash
python scripts/score_predict_fast.py --config /root/autodl-fs/zzw/configs/new-server.json \
  --workers 2 --benchmark 32 --stop-sequential
```

`--stop-sequential` 仅终止相同配置与实验名下旧的 optimize/predict 进程，不终止 tune、confirmation、训练或其他项目。若已有另一并行任务占用同一实验锁，会报错，不强制停止。
测速不写生产 test.jsonl，使用独立空缓存，防止把前一次 benchmark 的缓存命中当成加速；每次都从同一生产进度之后取样。完整记录在 `optimization/score-v2/benchmarks/workers-*/runtime.json`，包括模型启动时间、每题秒数、每小时题数和预计剩余小时。小样本 ETA 仅作估算。

比较 2/4 workers 后再后台续跑：

```bash
nohup python -u scripts/score_predict_fast.py \
  --config /root/autodl-fs/zzw/configs/new-server.json --workers 2 \
  > /root/autodl-fs/zzw/logs/score-v2-parallel.log 2>&1 < /dev/null &
```

保持代码、配置和模型不变时，沿用原本 `<selected>/test.jsonl` 的已完成前缀并自动导出同一 test.csv。初答缓存可复用，但 benchmark 的隔离缓存不混入生产测速记录。没有目标 GPU 的真实吞吐测试，不能据此宣称 2 倍或 4 倍加速。

最终提交是 `work_dir/optimization/score-v2/<selected>/test.csv`，由原有 exporter 按官方模板顺序完整校验导出。它是优化后的 **普通 RAG**，不应标成 RL 成绩。
先恢复可信的普通 RAG 基线，再用相同新配置、相同初答与相同 confirmation 样本，对原始 critic 和旧 RL adapter 做配对比较；旧 adapter 的训练分布不同，收益需要重新验证，不能直接承诺。

## 验证范围

本地回归检查覆盖证据保留、分支独立排名、标题去重、SQL 查询、知识提示、规划 schema、公有字段隔离、调用计数与 tune 选择/签名保护。无 7B/8B 权重与目标 GPU，不宣称新增策略已实测提高 Kaggle 或达到 0.5。服务器 tune/confirmation 是必要的真实效果检查。
