# 单份 Qwen 权重的批量推理

此入口用于已经完成 tune/confirmation 的 score-v2 普通 RAG。采用现有 Transformers 的真实批量 `generate`，不是多个进程重复加载权重，也不是 vLLM。沿用现有 Torch、Transformers、bitsandbytes 和 JSON 解码依赖。

## 执行方式

- 仅一个进程加载一份冻结 Qwen。多个 RAG 控制线程各有自己的检索连接/状态，通过同一个请求 broker 提交生成请求；所有 GPU 生成调用在同一个线程执行。
- broker 短暂合并请求，再按生成预算与输入长度分组，左侧 padding 并构造正确 attention mask。输入证据仍由原 fit_messages 预算，批处理不会为了凑 batch 额外截断证据。
- 对 `retrieval_plan_trigger=always`，先完成查询规划/检索，再生成一次答案。旧流程在规划之前及多轮规划之间生成的答案没有传入规划器，最后也被覆盖；新版不再生成这些被丢弃的答案。规划器及最终回答的公开输入保持一致，仍必须通过真实配对 pilot。unknown 条件触发及无规划配置保持原流程。
- 每个样本各自维护 JSON parser、可见文档 ID 和 EOS 状态。答案、查询规划和格式修复可以分批推进，引用范围不串题。生成结束后按首个 EOS 裁去批量 padding。
- JSON 过滤器只处理当前行生成的 token 前缀，不在每一步将整个长提示拷回 CPU。微型 Qwen 测试对比它与原完整前缀解码约束的等价性，包括 Unicode。
- `--max-batch-tokens` 限制批内 padding 后的输入+输出总预算；单个长请求不被裁短。CUDA OOM 时自动拆小同一批真实请求，单样本仍 OOM 就报错，不伪造答案。
- 主线程仍使用原 run_dataset 校验签名并按原顺序保存。失败保留已完成前缀；未完成的并发请求不跳题。线程退出时在各自所属线程关闭 SQLite。

批量提高的是吞吐量，并不会自动提高准确率；KV cache 和激活值会随批量增加。浮点计算形状可能改变生成结果，因此必须跑目标 GPU 配对 pilot。

## 服务器执行

在 `/root/autodl-fs/zzw/RAG-Team6-fixed` 拉取 `feat/rag-score-v2` 后：

```bash
cd /root/autodl-fs/zzw/RAG-Team6-fixed/autodl
mkdir -p /root/autodl-fs/zzw/logs
nohup bash run_score_v2_batch.sh all \
  /root/autodl-fs/zzw/configs/new-server.json \
  > /root/autodl-fs/zzw/logs/score-v2-batch.log 2>&1 < /dev/null &
```

默认 batch size=8、padding 后 token 总预算 147456、pilot 32 道 confirmation 题，面向当前 96GB 单卡环境。预算覆盖 8 条 16K 提示及其输出，避免小预算反复把一批拆成几批。它先停止同配置/实验名下旧的顺序或并行预测，等实验锁释放后再加载模型；不停止 tune/confirmation、训练或其他项目。不需要重新选择配置或重新训练。现有 reranker_path 必须为 null；不静默关闭用户配置的重排器。

`all` 依次执行：

1. 只加载一次 Qwen，用同一组真实 confirmation 题先顺序推理，再批量推理。二者使用独立空初答缓存；tokenizer JSON 过滤数据在计时前统一初始化，避免只把这项开销算到顺序模式。
2. 输出 EM、F1、答案变化数、每小时题数、观察到的加速比和 CUDA 显存峰值。小样本速度/精度都只是 pilot 观察，不是统计显著性或 Kaggle 分数保证。
3. 只有 pilot 的批量 EM 不低于顺序 EM，且测得耗时更短，才允许生产预测。失败时停止，查看报告或改 batch size 后重新 pilot，不强行切换。
4. pilot 进程退出，生产进程重新加载一份 Qwen，接着 `<selected>/test.jsonl` 的已完成前缀预测，最终导出相同位置的 test.csv。各阶段不会同时驻留多份模型。

可以只做验证/只续跑：

```bash
bash run_score_v2_batch.sh validate /root/autodl-fs/zzw/configs/new-server.json
bash run_score_v2_batch.sh predict /root/autodl-fs/zzw/configs/new-server.json
```

更改批量大小、token 预算或 pilot 规模：

```bash
BATCH_SIZE=2 MAX_BATCH_TOKENS=32768 BATCH_PILOT_SIZE=32 \
  bash run_score_v2_batch.sh all /root/autodl-fs/zzw/configs/new-server.json
```

最大 batch 可设 16，默认仍为 8。16 个长输入需要同时提高 token 预算，否则仍会拆成小批：

```bash
BATCH_SIZE=16 MAX_BATCH_TOKENS=294912 BATCH_PILOT_SIZE=32 \
  bash run_score_v2_batch.sh all /root/autodl-fs/zzw/configs/new-server.json
```

该上限覆盖 16 条 16K 提示及输出，不是预先分配的显存量。模型权重仍只有一份，KV cache 和激活值随批量增加。显存不足时拆批；速度是否更快以相同 32 题的 rows_per_hour 为准，16 不保证比 8 快，也不自动提高准确率。改变批量需重新 pilot，batch=8 的验证记录不能代替。尚未在目标 96GB GPU 实测 batch 16。

验证报告：`work_dir/optimization/score-v2/batch-validation/batch-8-tokens-147456.json`。
配对原始轨迹保存在 `batch-validation/pilot-*/{sequential,batch}.jsonl`，便于逐题检查差异。
生产调度统计：`<selected>/test.batch-runtime.json`；轨迹标明 execution_backend。
验证记录绑定配置、模型、推理源码、批处理脚本、主要依赖版本、GPU 型号/计算能力及 confirmation 数据；上述内容变化后需要重新 pilot。

本次只修改包外 backend/入口和测试，原推理包源码及配置不变，可以延续原先签名的已完成进度。调度文件的哈希另外记录，不以删改旧 manifest 绕过验证。plan-before-answer 使用独立版本的初答缓存键，轨迹记录 execution_strategy，真实生成调用数如实计数，不把少调用一轮描述成模型训练收益。

`generation_stats` 另外记录请求数、实际批次大小、预算分组数、因 token/count 上限增加的批次数、原始/补齐提示 token 数、输出 token 数、JSON 前缀 CPU 时间和 generate 总墙钟时间。前缀 CPU 时间不含 GPU 同步或 Transformers 构造 mask 的开销，不能当作完整 CPU 性能剖析。

## 验证范围

本地使用 Torch 2.8.0 / Transformers 4.56.2 的真实微型 Qwen 随机权重，在 CPU 检查左 padding、不同长度输出、逐行引用隔离、顺序与批量输出、Unicode 前缀约束和 OOM 拆批控制；另检查模型只初始化一次、并发请求合并、顺序写出、错误传播、已完成结果无需加载模型和 pilot 门槛。plan-first 回归分别比较一轮/两轮、空查询、无效规划和 unknown 触发，断言规划输入、最终回答输入及证据保持一致。测试不是研究成绩。

尚无目标 96GB CUDA 显卡上 7B 权重的实测速率或显存结果；服务器 pilot 是必要检查。当前仅支持冻结 Qwen 的确定性普通 RAG，不用于 critic 采样、LoRA 训练或 GRPO log-prob 计算。默认多进程入口仍保留，可用于比较。
