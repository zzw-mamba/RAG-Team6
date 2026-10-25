# Team6：AutoDL 可执行 RAG 与批判强化学习项目

面向你的**单卡48GB及以上**环境，默认路径已经配置好：

| 用途 | 路径 |
|---|---|
| 冻结的回答模型 | `/root/Qwen2.5` |
| 可训练批判模型 | `/root/autodl-tmp/models/Llama-3.1-8B-Instruct` |
| 原始数据 | `/root/autodl-tmp/data` |
| 缓存、索引、训练与结果 | `/root/autodl-tmp/rag-team6-work` |

这是完整实现，包含真实模型调用、批判LoRA训练、GRPO反向传播、检查点、验证、三组对照和CSV导出。**它不包含上一份报告中的pilot12权重，也没有在你的AutoDL GPU上实际测过准确率。** 先执行下面的真实GPU smoke检查，再正式训练。不能承诺直接复现57.81%，也不预设RL一定优于普通RAG。

## 直接使用 AutoDL.Art Qwen2.5 镜像（PyTorch 2.2.1 / CUDA 11.8）

用户实机输出为 `2.2.1+cu118`，与网页标注的2.2.2不同；项目按实机版本兼容，无需升级Torch。

镜像兼容版本位于 `feat/autodl-qwen-image` 分支。已有仓库执行：

```bash
cd /root/autodl-tmp/RAG-Team6
git fetch origin
git switch feat/autodl-qwen-image
git pull --ff-only
cd autodl
bash run_qwen_image.sh
```

专用入口自动使用镜像的 `/root/Qwen2.5/text-generation-webui/installer_files/env/bin/python`，无需手动激活 Conda。它创建继承镜像 Torch 的项目 `.venv`；旧项目环境先重命名备份为 `.venv.backup-*`。安装前检查镜像 Torch/CUDA 可用性，**不会安装或升级 Torch，也不会解析下载 NVIDIA CUDA 依赖**。首次仍需安装固定版本的项目依赖（其中 bitsandbytes 有一定体积），默认使用清华 PyPI 镜像；网络失败会停止，重跑同一入口即可。

兼容栈固定为 Transformers 4.45.2、PEFT 0.13.2、Accelerate 0.34.2、bitsandbytes 0.43.3、NumPy 1.26.4，详见 [依赖清单](environment/requirements-qwen-image.txt)。这些包只安装在项目 `.venv`，镜像的聊天 WebUI 环境保持独立。训练概率计算适配旧版 `num_logits_to_keep` 接口，保留 LoRA/GRPO、冻结回答模型和检查点恢复流程。

使用 [configs/autodl-qwen-image.json](configs/autodl-qwen-image.json)：Qwen 路径为 `/root/Qwen2.5/text-generation-webui/models`；LLaMA、数据、输出仍使用上表路径。Qwen 的 models 下须有唯一的 7B Instruct HF 模型目录；有多个时请复制配置，将 `qwen_path` 指向具体目录，再通过 `CONFIG=/absolute/path/config.json bash run_qwen_image.sh` 启动。两个模型都需要原始 HF **safetensors** 权重和 tokenizer，不能把 GGUF/AWQ/GPTQ 文件直接交给此 NF4 训练入口。镜像页面没有说明自带 LLaMA，需要保留你已准备的 LLaMA 权重。

原来的 `bash run_all.sh` 和 `bash scripts/install.sh` 也会在检测到镜像 Python 时自动选择兼容模式；`RAG_ENV_PROFILE=modern` 可使用下文原安装方式。兼容模式的 `--from`、`--skip-predict`、日志和恢复规则与原入口一致，例如：

```bash
# 先完成训练和独立评估
bash run_qwen_image.sh --skip-predict

# 训练中断后沿用已有检查点
bash run_qwen_image.sh --from train

# 镜像的 Python 安装位置与官方说明不同时指定实际路径
IMAGE_PYTHON=/actual/image/env/bin/python bash run_qwen_image.sh
```

项目已在真实 PyTorch 2.2.1 和 2.2.2 的 CPU 环境通过微型 Qwen/LLaMA、LoRA 梯度及断点恢复测试；7B/8B 的 GPU 量化、显存和最终效果仍需入口中的真实 smoke 验证。无需启动镜像的 `python server.py` 聊天服务。

## 环境也可以打包复用

本项目训练需要PyTorch，但不必每台机器都重新联网安装。已提供[环境打包说明](environment/README.md)、`Dockerfile`和离线依赖包工具。当前AutoDL是Linux x86_64、Python3.10，离线包应在同系统架构、同Python主/次版本机器准备一次：

```bash
bash scripts/download_env.sh
WHEELHOUSE="$PWD/.cache/wheelhouse" bash run_all.sh
```

第一步仍需下载一次；可在网络正常的另一台Linux机器准备好，再把整个wheelhouse搬过来。第二步使用校验后的本地文件离线安装。下载将Torch本体与NVIDIA依赖分源处理，并增加重试；已经完成的wheel保留复用。仓库提供打包工具，不内置数GB依赖或已构建镜像。Docker方案用于有Docker管理权限的主机，不假定AutoDL容器内还能启动Docker。

## 一键运行全部流程

已经克隆仓库后，在服务器执行：

```bash
cd /root/autodl-tmp/RAG-Team6
git pull --ff-only
cd autodl
bash run_all.sh
```

总入口会依次调用 **安装依赖 → 准备数据和索引 → GPU小规模检查 → 正式训练与选模 → 独立评估和Demo → 全量测试预测及三份CSV导出**。每个阶段在终端显示进度，并保留单独日志；任意阶段失败立即停止，保留原退出码，不继续后续训练或导出。Linux的`flock`可用时会阻止此项目内重复启动一键任务。

需要断开终端后继续运行时：

```bash
nohup bash run_all.sh > run_all.log 2>&1 &
tail -f run_all.log
```

退出`tail`不会结束后台任务。详细阶段日志位于`autodl/.cache/run-all/`，运行开始时会打印本次目录。

默认64个训练问题组、每组4条轨迹、每16组保存检查点，默认名称`pilot64`。完成后CSV位于`/root/autodl-tmp/rag-team6-work/test/pilot64/`，评估位于`/root/autodl-tmp/rag-team6-work/evaluation/pilot64/`。

常用选项：

```bash
# 只完成训练评估，暂不运行全量测试
bash run_all.sh --skip-predict

# 环境已安装好，从数据准备开始
bash run_all.sh --from prepare

# 排除训练阶段的问题后，从训练继续；沿用已有检查点恢复规则
bash run_all.sh --from train

# 从全量预测阶段继续
bash run_all.sh --from predict
```

仍支持`CONFIG`、`RUN_NAME`、`PILOT_GROUPS`、`SAVE_EVERY`、`RESUME`和`WARM_START`。例如`PILOT_GROUPS=16 SAVE_EVERY=4 bash run_all.sh`自动使用名称`pilot16`；如手动指定`RUN_NAME`则优先使用该值。各阶段沿用原有恢复约束，第一次检查点之前中断的训练不会被伪装成可精确恢复。原来的`bash scripts/all.sh`也会转到这个完整入口。

## 一、通用 CUDA 镜像的上传与安装

此节对应 `RAG_ENV_PROFILE=modern`；使用上方 Qwen 镜像专用入口时不执行这里的 Torch 升级。通用环境推荐选择 AutoDL 的 **Python 3.11/3.12、GPU可用**环境，以及支持BF16的RTX4090/L20/L40S/A系列等显卡。安装脚本复用已有的可用CUDA PyTorch；如果没有Torch、只有CPU版或版本低于2.6，则在项目`.venv`里安装`torch==2.8.0`的CUDA12.8构建，不修改Miniconda base，也不重装系统CUDA。默认NF4量化两模型，并使用梯度检查点；实际峰值显存需通过smoke确认。

在 AutoDL 终端克隆团队仓库并进入训练目录：

```bash
cd /root/autodl-tmp
git clone https://github.com/zzw-mamba/RAG-Team6.git
cd RAG-Team6/autodl
bash scripts/install.sh
```

安装脚本先创建独立`.venv`，再检查或安装PyTorch，随后安装锁定的Transformers/PEFT依赖，运行CPU测试，并检查CUDA、模型架构和数据第一条样本。缺少Torch不会在创建环境前直接退出；下载CUDA PyTorch可能较大，请等安装完成。

### 如果Terminal提示“没有torch”

终端的`python`可能是`/root/miniconda3/bin/python`，而项目使用`autodl/.venv/bin/python`。已经克隆仓库的用户先更新，再重新安装：

```bash
cd /root/autodl-tmp/RAG-Team6
git pull --ff-only
cd autodl
bash scripts/install.sh
source .venv/bin/activate
python -c 'import sys,torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())'
```

最后一行应输出项目`.venv`的解释器路径及`True`。`bash scripts/install.sh`的环境激活只在脚本内部生效，所以在终端直接执行Python命令前仍需`source .venv/bin/activate`；也可以始终使用`.venv/bin/python`。

默认CUDA12.8安装源对应[PyTorch官方2.8.0发布说明](https://pytorch.org/get-started/previous-versions/#v280)。其他驱动环境可通过`TORCH_INDEX_URL`选择该版本另一个官方CUDA安装源；已有CUDA构建却无法访问GPU时，脚本会明确报错，不把它当成可训练的CPU环境继续运行。

确认数据目录是：

```text
/root/autodl-tmp/data/
├── train.json
├── test.json
├── demos.json
└── sample_submission.csv
```

模型目录必须有HF格式的`config.json`、safetensors权重分片和tokenizer文件。`/root/Qwen2.5`若只是父目录，代码会寻找一层或两层下唯一的模型目录；存在多个候选就明确报错。不会把名称相近的其他大小模型静默替换进来。NF4训练入口使用原始BF16/FP16权重，不接受预先量化的AWQ/GPTQ权重。

## 二、第一次运行：准备数据和真实GPU检查

```bash
cd /root/autodl-tmp/RAG-Team6/autodl
bash scripts/prepare.sh
bash scripts/smoke.sh
```

`prepare.sh`执行：

1. 流式读取约GB级JSON，不一次性加载完整训练集。
2. 按规范化问题哈希分组，划分90%训练、5%调参、5%独立确认，重复问题不会跨集合。
3. 排除训练数据中与test问题重合的样本；只保留推理所需公开字段与独立的答案标签，舍弃rationale/qa_pairs。
4. 固定抽取128题tune、256题confirmation和2048题训练池。
5. 仅用训练划分的`ctxs`建立全局SQLite FTS5索引。每个验证/测试问题仍可以使用该题自带的`ctxs`，但其上下文不写进全局索引。

`smoke.sh`使用4道调参题检查普通RAG、原始LLaMA批判和训练后批判，另执行2组真实训练。`backward_probe.json`记录LoRA反向传播是否得到有限非零梯度；这个诊断反向传播**不执行优化器更新**，不当作RL成绩。正常RL是否实际更新则另见`summary.json`中的`optimizer_updates`。

CPU测试使用明确的合成样本和预设响应测试控制流程，不是大模型研究结果。正式模型调用没有Mock替代或网络失败兜底。

如果smoke途中报显存不足，复制配置，降低`max_input_tokens`至3072、`document_chars`至1100，再换一个新的`work_dir`重跑。不要在旧结果上改配置后继续比较。

## 三、训练：推荐先跑64组pilot并选择检查点

```bash
cd /root/autodl-tmp/RAG-Team6/autodl
bash scripts/train_pilot.sh
```

脚本默认每题采样4条完整轨迹，每条最多2轮批判；处理64个不同训练问题，每16组保存一次检查点；随后在tune128上评估各检查点，并按EM、较少改错、较少修正、较早检查点依次选择。

真正执行训练的命令等价于：

```bash
source .venv/bin/activate
python -m autodl_rag --config configs/autodl.json train \
  --run-dir /root/autodl-tmp/rag-team6-work/training/pilot64 \
  --groups 64 \
  --save-every 16
```

直接执行此命令后，再运行`bash scripts/train_pilot.sh`，脚本会跳过已完成训练，继续检查点验证和选择。不要同时启动两份训练/推理任务占用同一张卡或写同一个工作目录。

希望更快查看趋势，可以独立跑16组、每4组保存一次：

```bash
PILOT_GROUPS=16 SAVE_EVERY=4 RUN_NAME=pilot16 bash scripts/train_pilot.sh
RUN_NAME=pilot16 bash scripts/evaluate.sh
```

这是一套新的训练，不是原报告中已有pilot12权重的恢复。组数不等于优化器步数；组内4条轨迹奖励完全相同则跳过更新。训练脚本遇到总更新数为0会停止后续模型选择，要求检查日志，不会把未更新模型称作成功训练。

### 后台运行

```bash
cd /root/autodl-tmp/RAG-Team6/autodl
nohup bash scripts/train_pilot.sh > pilot64.log 2>&1 &
tail -f pilot64.log
```

`Ctrl+C`退出tail不会结束后台训练。

### 断点恢复

重新执行`bash scripts/train_pilot.sh`，若训练未完成且有`latest.json`，会自动从最近完整检查点恢复优化器与Python/CPU/CUDA随机状态。未写入检查点的日志会先备份，再从保存位置重放，不把重复尝试累计成新实验结果。

也可显式指定：

```bash
RESUME=/root/autodl-tmp/rag-team6-work/training/pilot64/checkpoints/group-000032 \
  bash scripts/train_pilot.sh
```

必须是本次运行中最新的完整检查点，且配置、总代数、源码、依赖版本、数据与模型记录一致。如果首次检查点前就中断，保留原日志并换`RUN_NAME`重新开始。smoke首次训练中断同理，应换工作目录或将其不完整训练目录另存后重跑。

### 更大训练预算

先看tune趋势，不默认训练越久越好。在接触独立确认集前，可以将另一个训练预算作为调参候选：

```bash
PILOT_GROUPS=256 SAVE_EVERY=32 RUN_NAME=pilot256 bash scripts/train_pilot.sh
```

这是从基座开始的独立256组运行。若要从已有LoRA开始一轮新训练，可增加`WARM_START=/absolute/adapter/path`；这会重置优化器和随机状态，使用冻结的起始适配器作reference，并从训练池开头取题，**不是精确断点续训，也不保证所有题都是新题**。

独立confirmation用于冻结方案后的确认。如果看过它再改方案，应另留新的确认集，不继续把旧集称作未参与选择的独立测试。

## 四、评估与过程Demo

```bash
bash scripts/evaluate.sh
```

该步骤在相同confirmation256上运行：

| 输出方法 | 含义 |
|---|---|
| rag | 固定Qwen＋检索，无批判 |
| critic | 原始LLaMA批判，无RL训练 |
| rl | tune选出的LoRA批判模型 |

三组复用相同的初答、证据和检索配置。输出目录：

```text
/root/autodl-tmp/rag-team6-work/evaluation/pilot64/
├── rag.jsonl
├── critic.jsonl
├── rl.jsonl
├── *.metrics.json
├── comparison.json
└── demo.html
```

`comparison.json`包括归一化EM、严格EM、token F1、纠错/改错、平均修正次数、调用次数、耗时、停止原因、判断/实体题分项及配对bootstrap区间。它是本地评估，不冒充Kaggle未公开的自定义Accuracy。

`logical_seconds`＝初答首次实际测得的检索与生成时间＋本方法新增耗时；`wall_seconds`是当前运行墙钟时间，缓存命中时可能很小。比较方法使用前者，并同时披露缓存及模型加载未计入逐题耗时，避免把缓存命中当作模型加速。

`demo.html`可下载到电脑用浏览器打开，展示前50条真实轨迹。JSONL中还保留初答实际可见证据、每轮批判证据、候选证据、门控判断、原始输出和停止原因，可用来挑成功与失败案例。

## 五、全量预测和提交CSV

完成训练与评估后执行：

```bash
bash scripts/predict.sh
```

默认生成三份全量CSV，使用模板原始ID与顺序：

```text
/root/autodl-tmp/rag-team6-work/test/pilot64/rag.csv
/root/autodl-tmp/rag-team6-work/test/pilot64/critic.csv
/root/autodl-tmp/rag-team6-work/test/pilot64/rl.csv
```

导出要求预测ID与模板集合完全一致、无重复、答案非空且单行。部分运行不能导出正式CSV；不会填充占位答案补行。真实回答的`unknown`与文件缺失占位不是一回事。预测不读取test标签，代码不自动上传Kaggle。

若只想生成RL版本：

```bash
source .venv/bin/activate
ADAPTER="$(cat /root/autodl-tmp/rag-team6-work/training/pilot64/selection.adapter.txt)"
python -m autodl_rag --config configs/autodl.json run \
  --split test --mode rl --adapter "$ADAPTER" \
  --output /root/autodl-tmp/rag-team6-work/test/pilot64/rl.jsonl --resume
python -m autodl_rag --config configs/autodl.json export \
  --predictions /root/autodl-tmp/rag-team6-work/test/pilot64/rl.jsonl \
  --output /root/autodl-tmp/rag-team6-work/test/pilot64/rl.csv
```

全量预测可能远慢于小规模验证。根据smoke/验证实际速度估算时间，预留CSV检查和提交时间。可以分阶段执行；`run_all.sh`默认包含全量预测，需要先查看训练评估结果时使用`--skip-predict`。

## 六、实现说明与效果边界

### 检索与生成

检索候选来自题目自带ctxs、仅训练上下文构建的全局FTS5 BM25、候选段落中的中间实体扩展和批判模型提出的定向查询。保留本题局部文档配额，避免多跳第二跳证据因问题词重叠低而全部被丢弃。对长文保留实体介绍和相关句子，再按token预算收缩，不静默截掉问题。

默认只需要你已有的两个模型，不下载外部百科或额外大模型。它**不是**原报告包含DBpedia/BGE的完全复现，因此分数不能直接与该报告等同。

如果你另外准备了本地`BAAI/bge-reranker-v2-m3`，可以在新配置中设置：

```json
{
  "reranker_path": "/root/autodl-tmp/models/bge-reranker-v2-m3",
  "reranker_device": "cpu"
}
```

请编辑完整配置里的对应字段；上面只是字段示意。CPU重排较慢，可在显存允许并完成smoke后改为`cuda:0`。更换检索方案后，三组都要用同一方案重新评估，不能只给RL组增加检索资源。

### 批判、修正与停止

最多两轮：LLaMA批判→定向补检索→Qwen修正→证据门控。默认Qwen以A/B互换顺序评估候选，两次都支持候选才接受，降低位置偏差。它仍是同一个Qwen判断，不是独立事实验证器，也不保证一定提升准确率。

`critic_stop`、`max_revisions`、`invalid_critique`、`invalid_revision`、`change_rejected`分别记录。无效批判保留已有有效答案，但不当作正确自主停止。最后一次拒绝也不会掩盖此前接受的有害修正，奖励与评估始终对比最初初答和最终答案。

Qwen初答/修正格式失败最多真实重答一次并计入调用；基础设施异常直接报错。CSV导出不悄悄重生成答案，因此验证轨迹与提交预测使用同一套结果语义。

### 强化学习

只对LLaMA的q/k/v/o LoRA参数求梯度，Qwen、检索和reference冻结。每个训练问题的4个分支共享同一个初始状态，分别采样最多两轮的完整轨迹。标签只用于训练奖励、验证评分，从不进入生成器、批判或重排输入。

默认轨迹奖励：

```text
基础分：错→对 +2；对→对 +1；对→错 -3；错→错 -0.3
最终奖励 = 基础分
         - 0.05 × 修正尝试次数
         - 0.50 × 无效批判/修正次数
         - 0.25 × 正确初答上提出错误候选的次数
```

候选即使被拒绝也产生修正成本；危险候选惩罚与最终改错指标分别记录。相比上一份报告，默认降低修正与危险提案惩罚以供验证，**这只是待检验的设计选择，不是已经证实的改进**。

训练使用组内中心化优势、标准差下限、裁剪策略目标与KL项。每组一次更新；当前组采样后立即更新，不跨组重用旧轨迹。采样与token likelihood均使用温度0.8，top-p=1、top-k=0、无重复惩罚。逐条轨迹/决策反向传播，避免同时保留四条大模型计算图。原始输出格式无效也有采样token和负奖励，不丢掉坏样本制造虚假收益。

训练计数不是独立评估分数；多次尝试同一题不能当作多个验证样本。当前默认单种子，bootstrap只反映固定模型在这批题上的样本变化，不包含训练种子的变异。

### 配置和双卡扩展

完整配置：`configs/autodl.json`。通过`CONFIG=/absolute/custom.json bash scripts/...`选择其他配置；修改重要设置时使用新的`work_dir`保存独立结果。

你当前单卡48GB配置是`qwen_device=cuda:0`、`critic_device=cuda:0`。如果以后有两卡，可设置Qwen为`cuda:0`，critic为`cuda:1`，仍运行单个Python进程。不要用torchrun/accelerate多进程启动这个训练器。

模型身份记录包括配置/tokenizer哈希及权重文件大小/修改时间，**没有对数十GB基座权重逐字节计算内容哈希**；LoRA与预测文件使用内容SHA256。跨机器严格复现实验时应另行保存基座文件的完整校验清单。

## 七、代码结构与协作

```text
autodl_rag/
  core.py         配置、流式读写、公开字段边界、解析与奖励
  data.py         问题分组划分、数据指纹、固定抽样
  retrieval.py    训练上下文索引、BM25、桥接实体和补检索
  models.py       本地Qwen/LLaMA、NF4 LoRA、token likelihood
  pipeline.py     回答、批判、修正、验证门控和轨迹
  training.py     完整轨迹GRPO、优化器、随机状态与检查点
  experiment.py   预测恢复、评分、模型选择、对照和CSV
  cli.py          命令行与HTML过程展示
configs/          AutoDL实际路径配置
scripts/          安装、准备、smoke、训练、评估和预测命令
tests/            纯CPU合成数据测试
```

建议队友分别负责数据/检索、生成/批判、训练/奖励、评估/案例、Demo/报告。不要把原始数据、模型、LoRA权重、缓存或训练日志提交到Git；它们都输出到代码目录之外的工作目录。

## 八、依赖依据和验证范围

量化与适配器接口参考官方版本文档：[Transformers 4.56.2 bitsandbytes](https://huggingface.co/docs/transformers/v4.56.2/en/quantization/bitsandbytes)、[PEFT 0.17量化训练](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization)。该实现不依赖TRL，以避免训练器版本间的奖励回调和分组行为差异。

交付时可在本地完成数据、检索、解析、奖励、缓存、恢复、导出和流程控制的CPU测试；目标GPU加载、NF4内核、实际反向传播和准确率需在你的AutoDL上执行smoke及独立评估后确认。GPU检查生成的文件保留在`work_dir/smoke/`，可用于后续定位环境或显存问题。
