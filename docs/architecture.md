# 模块契约与扩展点

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
