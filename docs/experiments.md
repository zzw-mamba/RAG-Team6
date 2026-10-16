# 实验与提交规范

## 三组主实验

| 名称 | 回答模型 | 批判模型 | 参考配置 |
|---|---|---|---|
| vanilla | 固定 Qwen | 无 | configs/vanilla.json |
| critique | 同一 Qwen | SFT LLaMA | configs/critique.json |
| rl | 同一 Qwen | SFT 后 GRPO LLaMA | configs/rl.json |

保持验证题、候选上下文、检索文档与顺序、回答提示词、生成参数、输入预算和最大轮数一致。改变检索设置时三组一起重跑，或作为单独消融。

建议先 30 题确认格式，再选择固定的 pilot 验证子集，最后全量验证。`--limit` 仅按固定输入顺序截取；它不进行随机分层抽样，需要分层时由数据负责人另产出固定子集。

## 日志

每次 `infer` 输出：

- `run.json`：数据/代码/配置指纹、提示词版本、Git、Python、平台、状态。
- `traces.jsonl`：每题问题、检索文档、每次初答/批判/修正、错误、停止原因与耗时。
- `metrics.json`：执行 `evaluate` 后生成。

推理轨迹不保存标准答案或原始训练 rationale。实验文件仍可能包含私有比赛题和文档，保留在忽略目录供组内按比赛要求共享。

```bash
raglab infer --input data/processed/split-v1/validation.jsonl --config configs/critique.json --run-dir outputs/critique-pilot --limit 30
# 同一命令恢复，limit 也必须相同
raglab infer --input data/processed/split-v1/validation.jsonl --config configs/critique.json --run-dir outputs/critique-pilot --limit 30 --resume
raglab evaluate --input data/processed/split-v1/validation.jsonl --run-dir outputs/critique-pilot
python scripts/capture_environment.py --output outputs/critique-pilot/environment.json
```

评测默认使用 run 记录的同一数据与 limit，拒绝缺失预测；`--allow-partial` 仅用于明确的中断诊断，报告覆盖率。`compare` 拒绝覆盖率不足的评测。

恢复会核对配置、数据、代码以及 Git 状态。HTTP 服务权重可变、HF Hub 可变 revision、同一路径适配器替换等情况，必须由团队额外记录和控制，不要在实验中途替换部署模型。

## 指标定义

- `normalized_em`：大小写归一、去 ASCII 标点、去英语冠词、折叠空白后，与任一标准答案相等。
- `token_f1`：同一归一规则下 token 重叠 F1，取标准答案中的最大值；yes/no/unknown 使用严格对应。
- `mean_revisions`：成功生成的新回答次数；不含初答、批判次数和格式重试。
- `mean_seconds` / `median_seconds` / `p95_seconds`：每题检索到最终答案的墙钟时间，含修正与格式重试，不含初次模型载入。
- `correction_rate`：初始 EM 错的题中，最终 EM 对的比例。
- `regression_rate`：初始 EM 对的题中，最终 EM 错的比例。
- `stop_reasons`：模型主动停止、达到轮数上限、格式失败等计数。

**上述 EM/F1 尚未与 Kaggle 自定义 Accuracy 评分代码对齐。** 分母为零的纠错/误改率记录为 null，不伪造为零。

GPU 型号、批大小、服务是否冷启动、量化、并发、缓存都影响耗时。可运行的比较工具只能检查部分条件，仍需负责人审查环境记录。当前推理是逐题调用，没有批量吞吐优化。

## 结果表模板

| 系统 | 验证题数 | 本地 EM | F1 | 平均修正 | 平均秒数 | 纠错率 | 误改率 | Kaggle Public |
|---|---|---|---|---|---|---|---|---|
| vanilla | 待运行 | | | | | | | |
| critique | 待运行 | | | | | | | |
| rl | 待运行 | | | | | | | |

Mock 结果另列为工程验证，不能填入正式结果表。

建议消融：未训练批判、固定轮数/自动停止、无修正成本奖励、检索 top-k、多轮训练状态。

## 案例与最终提交

至少保留：错→对、对→对并及时停止、对→错、检索缺证据导致失败。以真实轨迹说明，不只截取最终答案。

全量测试为 12,576 题。先用 pilot 估算耗时并预留重跑时间。测试推理无 gold；提交列仅 `id,answer`，每题一个简短最终答案，日志另存。

上传前检查本地 CSV、模型版本与 Public/Private 选择策略。线上每次提交有额度，不能代替本地验证。上传与选择最终提交由团队指定负责人完成，本框架不自动调用提交 API。
