# 排查大量 unknown 与修正宽松评分

## 已修正的评分漏洞

旧版 `substring_accuracy` 直接判断标准答案是否为预测的字符子串，会将 `unknown` 中的 `no` 误判为正确；`known` 也有类似问题。这个宽松指标可能虚高。严格 EM 不受此问题影响。

新指标版本为 `answer-metrics-v2`：

- `substring_accuracy` 保留原字段名，改为完整词边界的短语匹配；yes/no/noanswer/unknown 采用归一化后完整一致检查。
- `legacy_substring_accuracy` 保留旧算法数值，仅用于追溯，不能作为有效正确率。
- `legacy_substring_false_positive_count` 报告旧值命中、新值未命中的题数。
- 文档中答案字符串出现的诊断同样改为词边界匹配；它仍不代表证据链充分。

## 先重算旧日志，不必重新跑模型

在新版分支仓库根目录执行：

```bash
git pull --ff-only
export PYTHONPATH="$PWD/src"

python -m naive_rag.cli report \
  --input data/evaluation-v1/dev.jsonl \
  --run-dir outputs/validation-v1/dev-input \
  --output outputs/validation-v1/dev-input-report-v2
```

此命令重读已有预测并用原来的标准答案评分，不加载 Qwen，不修改旧日志或旧报告。输入必须与原运行完全相同，报告目录必须是新目录。没有逐题输出时，不能仅凭聚合指标重算正确的宽松分数。

新摘要除 EM/F1 外还会显示拒答数、非拒答题数、旧指标误命中数、候选和实际输入中答案短语出现比例、拒答且输入含答案短语的题数、发生正文截断的题数。`summary.json` 还包含按 yes/no 和其他答案类型分组的结果、平均与最大输入 token 数，以及各类诊断数量。

## 如何判断下一步

| 观察 | 可继续检查的方向 |
|---|---|
| 候选中少见答案短语 | 候选检索可能缺信息，也可能是别名或 yes/no 等诊断局限；抽样人工核对 |
| 候选有短语，实际输入没有 | 检查文档选择和正文截断 |
| 拒答且输入含短语较多 | 核对两跳关系是否完整，检查提示是否过于保守；不能仅凭短语出现认定拒答有错 |
| 有完整证据仍拒答 | 比较下面的提示词对照，并核对模型权重与输出 |

`cases.json` 会保留每类最多 5 个例子和实际模型输入文档，可结合 `per_question.csv` 筛选拒答题人工复核。没有这些逐题材料时，无法区分真实证据缺失、实体混淆、截断、提示策略和模型能力问题。

## 单次回答提示词对照

原配置保持原提示，版本记为 `direct-v1`，方便复现。新增 `configs/autodl_baseline_best_supported.json`，提示版本 `best-supported-v1`：强调通过中间实体连接不同文档、比较题需要对齐双方属性、间接证据足够时应回答；无法找到任何有依据的答案时仍可输出 unknown。它不增加模型调用、批判、答案修正或额外检索，也不将标准答案送入模型。

先在同一份 20 题 pilot 上试验，使用新输出目录：

```bash
python -m naive_rag.cli evaluate \
  --input data/evaluation-v1/pilot.jsonl \
  --config configs/autodl_baseline_best_supported.json \
  --output outputs/validation-v1/pilot-best-supported
```

对照需要同时看 EM/F1、拒答比例和错误案例。拒答减少不自动代表正确率提高，不应为了降低拒答率而强制无依据猜测。pilot 有积极信号后，再在相同 500 题 dev 上比较：

```bash
python -m naive_rag.cli evaluate \
  --input data/evaluation-v1/dev.jsonl \
  --config configs/autodl_baseline_best_supported.json \
  --output outputs/validation-v1/dev-best-supported
```

旧配置仍默认 `prompt_mode=direct`，新提示必须显式选用。代码或配置更新后不要续跑旧的半成品运行，使用新目录；重评分旧的完整预测则用 `report`。已有旧分数不等同于新提示的效果，尚未在目标 GPU 实测新提示收益。holdout 不参与此轮调参。
