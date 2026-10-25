# RAG Critic Lab

面向 10 人小组协作的《人工智能原理与应用》RAG 大作业项目。

目标：在 2WikiMultiHopQA 上比较 **普通 RAG → 训练后的批判修正 RAG → 强化学习优化批判模型的 RAG**。

回答模型固定为 `Qwen/Qwen2.5-7B-Instruct`，批判模型为 `meta-llama/Llama-3.1-8B-Instruct`。

## AutoDL 实际训练入口

**AutoDL.Art Qwen2.5 镜像用户**：兼容版本在 `feat/autodl-qwen-image` 分支，支持复用 PyTorch 2.2.1 / CUDA 11.8，不下载升级 Torch/CUDA。切换该分支后，在 `autodl/` 执行 `bash run_qwen_image.sh`；完整命令和路径见 [镜像运行说明](autodl/README.md#直接使用-autodlart-qwen25-镜像pytorch-221--cuda-118)。

单卡48GB及以上环境使用 [autodl/](autodl/README.md)，包含本地双模型加载、全局补检索、最多两轮批判修正、真实LoRA轨迹GRPO训练、检查点选择、三组对照和提交CSV导出。默认使用以下服务器路径：

- Qwen：`/root/Qwen2.5`
- LLaMA：`/root/autodl-tmp/models/Llama-3.1-8B-Instruct`
- 数据：`/root/autodl-tmp/data`
- 运行输出：`/root/autodl-tmp/rag-team6-work`

在AutoDL的PyTorch CUDA镜像中执行，需要Python 3.11或3.12：

```bash
cd /root/autodl-tmp
git clone https://github.com/zzw-mamba/RAG-Team6.git
cd RAG-Team6/autodl
bash run_all.sh
```

一键入口依次安装、准备数据、检查GPU、训练选模、独立评估和导出全量CSV，并逐阶段保存日志。默认训练64个问题组，每组4条真实轨迹，每16组保存检查点，使用tune128选模，再在独立confirmation256上比较普通RAG、原始LLaMA批判和RL批判。仅需训练评估时运行 `bash run_all.sh --skip-predict`；后台运行、失败续跑和参数见[一键运行说明](autodl/README.md#一键运行全部流程)。

安装脚本会在项目环境中补装缺失的CUDA版PyTorch，也可复用镜像已有的可用版本。还提供[离线依赖包和Docker环境配置](autodl/environment/README.md)，可以一次准备、跨匹配机器复用。已完成AutoDL模块本地49项测试，包括微型LLaMA的真实梯度、冻结参数、断点恢复、安装环境检查、一键入口控制流程和环境包完整性检查；实际7B/8B模型的CUDA量化、显存和准确率仍需在目标GPU上验证，详见 [验证范围](autodl/VALIDATION.md)。该版本不包含历史报告中的pilot12权重。下面保留原始框架的开发、离线演示和监督训练接口说明。

## 先跑起来：无需 GPU

先克隆团队仓库：

```bash
git clone https://github.com/zzw-mamba/RAG-Team6.git
cd RAG-Team6
```

需要 **Python 3.11 或 3.12**。macOS 自带 Python 3.9 不适用。以下命令从项目根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
raglab smoke --output outputs/my-first-smoke
raglab demo --run-dir outputs/my-first-smoke/critique
```

Windows 激活命令：`.venv\Scripts\Activate.ps1`。如果已经使用 Python 3.12，把首行替换为对应解释器即可。

打开终端打印的本地地址 `http://127.0.0.1:8765`，可以筛选问题并查看检索文档、初答、批判、修正和最终答案。

`smoke` 使用仓库自带的 **3 条虚构样本与脚本化 Mock 模型**，故意制造初答错误以检查纠错流程。它没有训练模型，生成的分数不能作为作业实验结果；其中 `rl` 仅验证模式接线，没有进行 RL 更新。真实后端出错不会降级为 Mock。

每次实验使用新的输出目录，或对同一次 `infer` 使用 `--resume`。

## 项目结构

```text
rag-critic-lab/
├── autodl/                   # 单卡48GB实际训练、评估、提交入口
├── src/raglab/
│   ├── schema.py             # 公共数据、答案、批判、轨迹契约
│   ├── data.py               # 比赛字段适配、JSON/JSONL、分组划分
│   ├── retrieval.py          # 候选文档 BM25 / 原顺序基线
│   ├── models.py             # Mock / Transformers / HTTP 模型后端
│   ├── prompts.py            # 证据约束与结构化输出提示词
│   ├── pipeline.py           # 检索→回答→批判→修正→停止
│   ├── experiments.py        # 日志、断点续跑、实验一致性检查
│   ├── evaluation.py         # 本地 EM/F1、耗时、纠错率、误改率
│   ├── submission.py         # ID 与行数校验、CSV 导出
│   ├── demo.py               # 本地轨迹回放界面
│   └── training/
│       ├── annotations.py    # 待审核批判样本、训练数据验证
│       ├── sft.py            # LoRA/QLoRA 监督训练入口
│       ├── rewards.py        # 下游回答效果奖励
│       └── grpo.py           # 一步批判策略的 GRPO 训练入口
├── configs/                  # 三组实验、HTTP、SFT、GRPO 配置
├── fixtures/                 # 可提交 Git 的虚构测试数据
├── tests/                    # 无需 GPU 的行为与集成测试
├── data/{raw,processed,annotations}/  # 真实数据，已忽略
├── checkpoints/              # 适配器及权重，已忽略
├── outputs/                  # 运行日志、指标、CSV，已忽略
├── docs/                     # 架构、分工、训练、实验与比赛说明
└── .github/                  # CI、PR 与 Issue 模板
```

## 真实数据与基线

先在 Kaggle 登录本人的参赛账号，取得本次比赛的四个文件，放到 `data/raw/`。不要用原版公开数据的答案回填本次测试集。

```bash
python -m pip install -e '.[data]'
raglab inspect --input data/raw/train.json --limit 3
raglab inspect --input data/raw/test.json --limit 3
raglab prepare --input data/raw/train.json --output data/processed/split-v1
```

1.65GB JSON 使用 `ijson` 流式读取；没有该依赖时会拒绝把大 JSON 整体载入内存。训练数据按问题哈希划分，同一规范化问题不会跨训练集与验证集。之后合成批判数据也必须保持这个划分。

`ctxs` 的完整内层结构尚未下载核对。适配器支持常见的 `title/text`、`contents`、`passage`、`sentences` 和原版 `[title, sentences]`；遇到未知结构会报错，请数据负责人核实后显式补充。

在 Linux CUDA 训练机安装可选依赖并验证显存后：

```bash
python -m pip install -e '.[train,quant]'
raglab infer --input data/processed/split-v1/validation.jsonl --config configs/vanilla.json --run-dir outputs/baseline-pilot --limit 30
raglab evaluate --input data/processed/split-v1/validation.jsonl --run-dir outputs/baseline-pilot
```

LLaMA 下载可能需要先取得模型访问权限。真实配置会下载大模型，4-bit 配置需要 CUDA 和 bitsandbytes。CPU/Mac 可运行核心开发与 Mock；默认真实配置不是 Mac 训练配置。

## 批判训练与三组对比

完整步骤见 [训练指南](docs/training.md)。先收集待审核样本：

```bash
raglab collect-annotations --input data/processed/split-v1/train.jsonl --config configs/vanilla.json --output data/annotations/train.pending.jsonl --split train --limit 100
```

审核 `target_critique`、`reviewed`、`reviewer` 后，另存 `train.reviewed.jsonl`。没有审阅标记或混入验证样本时，训练入口会拒绝执行。

```bash
raglab train-sft --config configs/training/sft.json --annotations data/annotations/train.reviewed.jsonl --check-only
raglab train-sft --config configs/training/sft.json --annotations data/annotations/train.reviewed.jsonl
raglab train-grpo --config configs/training/grpo.json --annotations data/annotations/train.reviewed.jsonl --check-only
raglab train-grpo --config configs/training/grpo.json --annotations data/annotations/train.reviewed.jsonl
```

GRPO 默认通过本地兼容 Chat Completions 的 HTTP 服务调用冻结的 Qwen；服务需要事先部署。也可改用 HF 后端，但两个模型同时驻留的显存需要实测。当前训练代码是单进程 LoRA/QLoRA 起点，尚未在真实 GPU 上验证。多卡分布式和多轮轨迹 RL 留给后续开发。

训练产生适配器后，使用相同验证集分别运行 `configs/vanilla.json`、`configs/critique.json`、`configs/rl.json`。三个配置的回答模型、检索和预算相同，后两者使用不同批判适配器。

```bash
raglab infer --input data/processed/split-v1/validation.jsonl --config configs/critique.json --run-dir outputs/critique-v1
raglab evaluate --input data/processed/split-v1/validation.jsonl --run-dir outputs/critique-v1
```

每个运行目录包含 `run.json` 与 `traces.jsonl`；评测后增加 `metrics.json`。模型载入时间不计入每题时间，HTTP 服务冷启动需单独控制。`compare` 会检查数据、代码、检索文档、回答模型和预算是否一致：

```bash
raglab compare --runs outputs/vanilla-v1 outputs/critique-v1 outputs/rl-v1 --output outputs/comparison-v1.json
```

上面三个目录需要事先各自完成推理与评测。更多设置见 [实验规范](docs/experiments.md)。

## Kaggle 导出

确认全量测试推理成功后：

```bash
raglab infer --input data/raw/test.json --config configs/rl.json --run-dir outputs/test-rl-v1
raglab export --traces outputs/test-rl-v1/traces.jsonl --template data/raw/sample_submission.csv --output outputs/submission-rl-v1.csv
```

导出会检查 `id,answer`、12,576 条结果、完整 ID 集、重复 ID、空答案及格式失败；按模板顺序写出。Mock 不允许从正式 CLI 导出。命令不会上传到 Kaggle。

当前仅知道 Kaggle 以 Accuracy 排序，详细匹配规则没有公开；本项目指标名称使用 `normalized_em` / `token_f1`，不把本地分数冒充线上 Accuracy。

## 10 人协作与 GitHub

团队仓库：[zzw-mamba/RAG-Team6](https://github.com/zzw-mamba/RAG-Team6)。先阅读 [分工与里程碑](docs/team.md)、[贡献指南](CONTRIBUTING.md)。建议比赛期间保持 **Private**，邀请另外 9 人，通过 Issue 认领模块、分支开发、PR 合并。代码已配置 CI；邀请和分支保护由仓库管理员在 GitHub 设置。

队友克隆后，从功能分支开始开发：

```bash
git switch main
git pull --ff-only
git switch -c feat/your-task
```

执行前检查 `git status --short`，确认没有真实数据、模型、Token。`CODEOWNERS` 提供了模块示例，填写真实 GitHub 用户名后才生效。代码许可证尚未选择；数据和模型沿用各自条款。

比赛约束与待确认项见 [作业说明](docs/assignment.md)，架构见 [模块契约](docs/architecture.md)。
