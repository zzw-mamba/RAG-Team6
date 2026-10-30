# 模块契约与扩展点

## AutoDL 训练模块

低分优化入口见 [基线优化说明](../autodl/SCORE_OPTIMIZATION.md)。可选 coverage 检索与有界 Qwen 查询规划仅接收公开字段；模型知识补全使用独立提示词，引用仍严格限定可见文档。新配置先用 tune 选定，再独立 confirmation 检查，最后导出普通 RAG 提交，不替代三组批判训练实验。默认 legacy/严格回答保持不变；源码签名变化后使用新输出目录。

`autodl/`是独立可安装的服务器运行模块，入口为`python -m autodl_rag`，配置和完整操作见[AutoDL说明](../autodl/README.md)。原`src/raglab`接口保持原样。

该模块只将问题、公开上下文及当前模型输出传给模型，标签只用于训练奖励和独立评估。全局检索库仅取训练划分的上下文，每题自带ctxs仍属于该题的公开输入。基线和批判方法共享同一初答缓存及证据。

冻结Qwen的回答和顺序互换门控使用受约束JSON解码，答案字段优先、解释长度有上限、引用限定为预算后可见文档ID。训练中的LLaMA不使用该约束，保持采样和梯度概率一致。`probe-answer`只检查指定公开问题的真实Qwen输出，不读取标签或计算准确率；它与正式初答共享缓存和推理版本。

并行测试预测由包外scripts/predict_parallel.py调度相同Pipeline实例，spawn子进程独立持有模型和只读检索连接，父进程沿用run_dataset顺序保存。只向worker传递公开字段，限制预取数量；失败保留已完成前缀，不跳题。调度实现和测速记录独立于原推理源码签名，允许既有预测续跑。

普通 RAG 还可使用包外 scripts/batched_qwen.py：一份冻结 Qwen 由单个生成线程持有，多条 Pipeline/检索线程通过 broker 合并生成请求。每行独立 JSON/引用/EOS 状态，左 padding，按预算/长度分组，父控制器仍顺序保存。scripts/score_predict_batch.py 先在目标 GPU 做独立缓存的顺序/批量配对 pilot，EM 不降且测得更快才允许生产；模型、脚本与依赖版本变化需重测。仅适用于确定性 Qwen RAG，不修改训练分布或批判停止语义，详见[批量推理](../autodl/BATCHED_INFERENCE.md)。

可选 vLLM 服务端使用独立环境和一份 BF16 Qwen，客户端沿用相同证据预算发送 token IDs 与逐题 JSON Schema，通过本机 completions API 连续调度。客户端不加载 GPU 权重；其 load_in_4bit=false/backend/deployment 配置与原 NF4 输出隔离，试跑对照真实 HF 报告后才允许独立完整提交，不能混用不同精度预测。部署、HTTP 协议和验证范围见 [vLLM 说明](../autodl/VLLM_INFERENCE.md)。

与原框架的主要区别是：批判JSON使用`action=stop/revise`；修正时允许定向补检索；候选通过Qwen顺序互换验证后接受；强化学习按最多两轮完整轨迹计算奖励。`max_rounds`配置限制批判循环次数，对应停止原因`max_revisions`，与`critic_stop`、`invalid_critique`、`invalid_revision`、`change_rejected`分别记录。已有有效终答会在批判格式失败后保留，该失败不计作正确自主停止。

原框架和AutoDL模块使用不同运行目录与结果格式，不直接混合评分。AutoDL的CPU/微型模型测试不替代目标GPU的真实pilot验证。

## 数据边界

`Example` 包含研究所需的 labels，但推理接口仅接收：

```python
pipeline.run(sample_id, question, contexts)
```

检索器接收问题与候选 `Document`；回答和批判提示词只包含这些公开信息及当前回答。训练标签仅用于评测、人工审阅与奖励计算。

`normalize()` 是唯一原始字段适配入口。未知 ctxs 结构报错，不静默跳过。原始训练题没有 ID 时创建 `row:N` 内部 ID；测试 ID 必须保留，导出时以模板校验，不重新编号。

`prepare` 输出规范 JSONL 与 manifest，按规范化问题哈希分组，使用相同 seed 保持划分。若准备失败且无 complete manifest，不使用部分文件训练。

## 模型接口

```python
class ChatBackend(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...
```

后端返回文本，由 schema 统一解析。Mock 仅读取虚构文档中的 DemoFact；HF 通过 chat template 调用指定模型；HTTP 对接兼容 `/v1/chat/completions` 的服务。

回答 JSON：

```json
{"answer":"一个最终答案","explanation":"简要证据说明","evidence_ids":["doc_0"]}
```

批判 JSON：

```json
{"decision":"revise","feedback":"根据 doc_0 修正具体事实","evidence_ids":["doc_0"]}
```

`decision` 只能是 `stop` 或 `revise`，两者都要说明原因；引用 ID 必须来自实际检索文档。格式错误最多重试配置次数，每次记录错误。引用存在不等于逻辑支持成立，后者需要模型、人工审阅与实验分析。

## 流程语义

1. 检索一次，固定候选文档及顺序。
2. Qwen 产生初答；失败时该题报错，不写伪造的正常结果。
3. 非 vanilla 模式调用批判模型。
4. `stop` 结束；`revise` 且未达上限时让 Qwen 修改，增加一次成功修正计数。
5. 回到步骤 3。最后一次允许修改后仍会检查是否获得模型停止信号。

| stop_reason | 含义 |
|---|---|
| vanilla | 普通 RAG 单次回答 |
| critic_stop | 批判模型决定结束，不自动等于答案正确 |
| max_revisions | 控制器达到修正上限，不计作自主停止 |
| invalid_critique | 批判输出在重试后仍无效，保留上个有效答案 |
| invalid_revision | 修正输出无效，保留上个有效答案 |

正式 CSV 导出拒绝含未解决格式失败的轨迹。

## 运行与恢复

`run.json` 记录输入 SHA256、完整配置、源码 SHA256、提示词版本、Git revision、Python 和平台。每题 JSONL 写完 flush/fsync，恢复前校验指纹。只截掉最后一条缺少换行的尾部记录；已完整写出的损坏行会报错。

当前不做多进程推理和批量调度。若团队新增并发，要保持 ID 唯一、日志顺序可恢复、GPU 内存可控。HTTP 模型权重版本须由服务负责人另外记录，客户端无法证明服务权重始终不变。

## 建议扩展

- 检索负责人：在 `retrieve` 后加入 dense/hybrid/reranker，注意多跳第二跳召回。
- 模型负责人：批量 HF/vLLM、上下文预算、服务端结构化约束、准确的 token 统计。
- 批判数据负责人：多轮状态、噪声证据、停止样本和教师生成质量控制。
- RL 负责人：从当前一步反馈奖励扩展到完整多轮轨迹回报。
- Demo 负责人：当前为离线回放；增加新问题实时输入时仍使用同一 public-only 推理入口。

新增方法另加配置与结果，不覆盖基线。公共契约变化要同步数据、训练、评测和 Demo 使用方。
