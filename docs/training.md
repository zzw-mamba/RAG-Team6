# 批判模型训练指南

## 当前完成程度

已提供可调用的 SFT 与 GRPO 入口，包含真实模型加载、LoRA/QLoRA、Trainer、适配器保存和训练 manifest。核心契约和奖励有本地测试；**真实模型下载、GPU 训练、显存和第三方训练库组合尚未实测**。

选用的依赖范围是兼容性起点，不是已通过 GPU 验证的锁文件。先在目标 Linux CUDA 机创建独立环境并安装 `.[train,quant]`，使用小数据试跑，成功后记录 `pip freeze`。不要直接在全量数据上试验依赖。

## 一、准备批判状态

先对原始训练数据执行 `prepare`，仅从训练划分生成训练状态。对验证划分另外生成验证状态。

```bash
raglab collect-annotations --input data/processed/split-v1/train.jsonl --config configs/vanilla.json --output data/annotations/train.pending.jsonl --split train --limit 100
raglab collect-annotations --input data/processed/split-v1/validation.jsonl --config configs/vanilla.json --output data/annotations/validation.pending.jsonl --split validation --limit 30
```

输出每行包含：

| 字段 | 用途 |
|---|---|
| id / split / source_sha256 | 样本来源与划分 |
| question / documents / current_answer | 批判模型可以看到的状态 |
| gold_answers | 仅供审阅与训练奖励使用，不进入模型提示词 |
| target_critique | 需要补齐的批判监督目标 |
| reviewed / reviewer / notes | 审阅记录 |
| synthetic / generator | 标明虚构数据与生成配置 |

已附的 `rationale` 没有自动用来监督训练，也不会进入推理提示词。

## 二、审阅

复制到 `*.reviewed.jsonl`，逐条检查：

- `revise`：有明确证据证明当前回答/说明存在错误，并指出具体错误和可执行修改。
- `stop`：回答已得到支持，或当前文档确实无法支持进一步修正；后者应说明证据缺失，不当作答案正确。
- 不得通过标准答案偷偷补充检索文档没有的事实。
- `evidence_ids` 均存在且实际支持反馈，而不只是随便引用一个 ID。
- 不因句式或风格要求无意义修改；正确答案、错误答案、缺失证据都需要覆盖。

示例：

```json
{
  "target_critique": {
    "decision": "revise",
    "feedback": "doc_1 确定导演身份，doc_2 给出其出生地；当前答案使用了另一个人的出生地，请据此修正。",
    "evidence_ids": ["doc_1", "doc_2"]
  },
  "reviewed": true,
  "reviewer": "填写真实审阅者或审批流程标识"
}
```

可以后续接入经课程许可的教师模型辅助生成，但仍需质量抽查、记录来源。当前未实现自动教师生成，不能把 `reviewed=false` 批量改为 true 代替审阅。

## 三、SFT

```bash
raglab train-sft --config configs/training/sft.json --annotations data/annotations/train.reviewed.jsonl --validation data/annotations/validation.reviewed.jsonl --check-only
raglab train-sft --config configs/training/sft.json --annotations data/annotations/train.reviewed.jsonl --validation data/annotations/validation.reviewed.jsonl
```

无需 GPU 的入口验证也可使用：

```bash
raglab train-sft --config configs/training/sft.json --annotations fixtures/annotations.reviewed.jsonl --check-only
```

虚构 fixtures 只能用于 check-only，不能通过真实训练入口运行。check-only 验证 JSON 和审核契约，不检查 GPU、Tokenizer 或依赖是否可用于训练。

监督损失仅覆盖 assistant 的目标批判与结束 token；问题、文档和当前回答的 labels 为 `-100`，padding 同样忽略。长于最大序列长度的样本会报错，不通过截掉反馈目标来凑长度。验证集不参与训练，若规范化问题重叠则拒绝运行。

默认输出 `checkpoints/critic-sft`。不要覆盖已有实验：为不同训练修改输出目录，并同步推理配置的 adapter 路径。

## 四、GRPO

当前实现为**一步批判动作的下游结果奖励**：

```text
当前问题/证据/回答
    → LLaMA 采样 stop 或批判反馈
    → stop: 保留回答；revise: 固定 Qwen 修改一次
    → 与训练标签比较，计算奖励
    → GRPO 更新 LLaMA LoRA 适配器
```

它适合作为第一版强化学习实验，尚不是完整多轮轨迹优化。推理阶段仍可多轮修正；团队可加入后续轮次的训练状态或扩展为轨迹回报。

奖励默认：

```text
R = 1.0 × 最终EM
  + 0.5 × (最终EM - 初始EM)
  + 0.1 × (最终F1 - 初始F1)
  - 0.05 × 是否请求修正
  - 1.0 × 是否输出格式无效
```

它鼓励纠错，惩罚误改和无意义修正。引用 ID 校验只是格式约束，**尚无自动证据蕴含奖励**。当前后续模型格式失败也会受到无效结果惩罚；可进一步研究区分反馈质量与回答模型格式失败的归因。服务中断、网络异常不会被转换成政策奖励，会中止训练。

默认 GRPO 配置从 SFT adapter 继续训练，通过 HTTP 调用冻结 Qwen：

1. 在团队选定的推理服务上部署指定 Qwen，并保持权重、解码配置固定。
2. 服务需支持 `/v1/chat/completions`；在配置中填写 `base_url` 和服务使用的模型名。
3. 默认模型名严格为 `Qwen/Qwen2.5-7B-Instruct`，避免误用其他模型。HF 路径也可用，但额外显存需实测。
4. 设置 SFT adapter 路径，检查 GRPO 参数、显存、候选反馈数量。
5. 执行 `train-grpo --check-only`，然后少量 steps 试跑。

`beta` 控制 TRL 的 KL 正则。PEFT 场景具体参考策略由所选 TRL 版本决定，不应未经检查就声称是“相对 SFT 策略的 KL”；若实验要求冻结 SFT 作为参考，需显式实现和验证参考适配器。

## 五、硬件与环境

- 默认单进程训练，只使用 `CUDA_VISIBLE_DEVICES` 暴露的第一张卡；没有实现 DDP/FSDP 启动器。
- Qwen 与 LLaMA 的全精度权重合计约 30GB 以上，训练还需激活/优化器/生成缓存；24GB 配置需要量化、控制上下文并实测。
- 若 GPU 不支持 BF16，需同时修改模型 dtype 与训练 fp16/bf16 参数。
- 推理 `device_map=auto` 不是两个模型共享单卡显存的调度器；必要时配置 `max_memory`、卸载或使用独立 HTTP 服务。
- GRPO 候选数、上下文长度、参考策略与奖励生成都会影响显存和速度。不要把能载入模型当作能训练。
- 配置中的适配器、本地缓存、训练输出不会提交 Git。

保存的 `training_manifest.json` 包含配置、数据校验值、Git 信息和训练库版本；另运行环境采集脚本记录 GPU。训练入口不自动恢复 checkpoint；需要后续接入 `resume_from_checkpoint` 时验证优化器、随机状态与数据位置。

## API 参考

- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
- [TRL GRPO 与自定义奖励](https://huggingface.co/docs/trl/grpo_trainer)
- [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

实际使用以所选版本文档和 GPU 试跑为准。
