# 使用 train 验证朴素 RAG

`train` 提供标准答案，可以留出一批题评估当前 RAG。baseline 本身不需要训练；这里使用训练数据的标签做离线验证，不是用这些答案指导模型生成。本方案尚未在服务器运行出真实准确率。

## 固定数据与使用边界

按大小写、空格归一化后的问题分组，用 seed=42 的 SHA256 哈希固定分配：

| 部分 | 约占唯一问题比例 | 当前用途 |
|---|---:|---|
| train | 80% | 留给后续批判模型训练，不用于本次效果报告 |
| dev | 10% | 从中按固定哈希优先级抽 500 题，比较配置 |
| holdout | 10% | 从中固定抽 1000 题，选定配置后做一次最终确认 |

另将 dev 中的前 20 道固定抽样题导出为 pilot，它是 dev 的子集，不是第三个独立评测集。不会直接取原始大文件的前 500 行。相同归一化问题只留首次出现的一行，避免重复题加权；数据重排可能改变重复题的代表行，因此必须保留数据源 SHA256 与划分清单。同一问题绝不会跨 dev/holdout，近义问题或同实体问题仍可能关联。

预处理流式遍历完整 train，校验字段和标签，导出 `pilot.jsonl/dev.jsonl/holdout.jsonl/evaluation_split.json`。训练用的 80% 不另存一份大文件；后续训练应使用 `naive_rag.validation.partition(question, seed)` 相同规则筛选 `train` 分区，排除整个 dev 与 holdout 分区，不能仅排除抽中的 1500 题。

**train 留出集与官方测试集可能分布不同；本地高分不能直接推算 Kaggle 分数。**Qwen 预训练是否见过公开数据集也无法由该划分证明。这里主要验证当前链路和配置的实际效果。

## 在服务器运行

在 `feat/naive-rag-baseline-v2` 分支的仓库根目录执行。Python 3.11/3.12，已安装匹配服务器 CUDA 的 PyTorch，然后安装项目：

```bash
python -m pip install -e '.[gpu]'
```

下面假定完整训练文件位于 `data/raw/train.json`；若在其他目录，只替换 `--input` 路径。正式流程使用新的 `prepare-eval`，不与此前通用的 `split` 划分混用。

```bash
# 1. 只准备一次，之后所有配置共用这些文件
naive-rag prepare-eval --input data/raw/train.json --output data/evaluation-v1 --seed 42 --dev-size 500 --holdout-size 1000

# 2. 20 题真实 Qwen pilot，先确认输入、显存、格式与耗时
naive-rag evaluate --input data/evaluation-v1/pilot.jsonl --config configs/autodl_baseline.json --output outputs/validation-v1/pilot

# 3. 500 题正式开发验证，运行主基线
naive-rag evaluate --input data/evaluation-v1/dev.jsonl --config configs/autodl_baseline.json --output outputs/validation-v1/dev-input

# 4. 相同 500 题，比较 BM25 重排
naive-rag evaluate --input data/evaluation-v1/dev.jsonl --config configs/autodl_baseline_bm25.json --output outputs/validation-v1/dev-bm25
```

两配置都使用本地 `/root/autodl-fs/zzw/models/Qwen2.5-7B-Instruct`，其余模型精度、token 预算、文档上限和生成参数相同。每题候选不多于 10 篇时，BM25 对照主要改变文档顺序；候选超过 10 篇时还可能改变选中文档集合。

用 dev 的归一化 EM 为主要指标，结合 F1 和耗时选定配置。然后对固定的 1000 题 holdout 做最终确认。例如选中原顺序配置：

```bash
naive-rag evaluate --input data/evaluation-v1/holdout.jsonl --config configs/autodl_baseline.json --output outputs/validation-v1/holdout-selected
```

若选中 BM25，则将最后一条命令配置改为 `configs/autodl_baseline_bm25.json`。不要根据 holdout 分数继续反复改提示或调参；它用于确认 dev 的结论。没有真实实验前不能预设哪种配置更好。

验证中断可在同一命令后加 `--resume`。输入、配置和代码必须与该次运行一致。完整报告已存在时，续跑会展示原报告；需要重新分析时使用下述 `report` 命令另建报告目录。

```bash
naive-rag report --input data/evaluation-v1/dev.jsonl --run-dir outputs/validation-v1/dev-input --output outputs/validation-v1/dev-input-report-review
```

## 输出和判读

每次 `evaluate` 在普通推理日志之外生成 `report/`：

| 文件 | 如何使用 |
|---|---|
| `report.md` | 直接阅读指标摘要 |
| `summary.json` | 样本数、EM/F1、宽松子串命中率、EM Wilson 95% 区间、unknown 比例、平均/P50/P95 耗时、按 type 分组结果 |
| `per_question.csv` | 逐题问题、标准答案、预测、得分、耗时、截断文档数、答案字符串诊断，可下载查看 |
| `cases.json` | 最多各 5 个正确及不同诊断类型的失败样例，包含实际送入模型的文档 |

分组仅在原始数据提供字符串 `type` 时保留，否则归为 `unspecified`，不根据问题文本猜类别。运行前检查标签完整性；报告重算分数，核对输入 SHA256、预测 ID 与样本数，不把遗漏题排除后给出虚高分数。

归一化 EM 忽略英文大小写、标点、冠词和空格差异，多答案取最大值；F1 测量词重叠。子串命中率是较宽松的参考值。这些都不是已验证的 Kaggle Accuracy。

Wilson 区间只描述当前样本下 EM 的抽样不确定性，不覆盖数据关联、模型记忆、选择偏差或调参影响。调参集分数带有选参偏差，最终以独立 holdout 确认结果为准。

默认批大小为 1，适合比较逐题延迟。计时包含检索与提示构建、GPU 生成，不含权重加载和日志写入，未排除首次推理预热；较大 batch 的计时为分摊值，不是单请求延迟。建议先用同一 pilot 预热并记录运行顺序，必要时在相同硬件状态下复测，避免把热机差异解释成方法提升。

## 失败诊断的边界

自动检查答案字符串是否出现在：原候选、选中文档、实际模型正文中。它可提示“可能缺证据”“选择或截断可能丢掉线索”“有答案字样仍答错”。这些标签不是已经确认的因果解释，也不是支持文档 Recall@K。

尤其 yes/no 答案，字符串存在几乎不能说明证据充分；实体题中出现最终答案也不代表两跳关系都成立。应结合 `cases.json` 人工检查实体对应、关系链、日期比较、候选证据和截断位置。

输出含参考答案，仅用于本地评估；不会传入模型提示，不会作为比赛提交或自动上传 Git。正式测试集继续用 `run` 与 `submission`，无答案的测试文件会被 `evaluate` 在推理前拒绝。

无模型环境可用 `--dry-run` 验证评估流程，报告会醒目标注 DRY-RUN；其分数、区间、耗时都不构成模型效果。
