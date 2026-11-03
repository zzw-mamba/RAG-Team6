# 课程参考仓库与当前 baseline 的对应关系

阅读日期：2026-11-03。阅读了 README 和相关源代码，没有安装或运行第三方仓库。

后续评分修订：当前项目 `answer-metrics-v2` 为避免 `unknown` 中的 `no` 误命中，已对短语边界与类别答案增加检查；该指标不再与参考仓库原始字符子串 acc 完全相同。原算法值仅以 `legacy_substring_accuracy` 留作追溯。

固定阅读版本：

- RAG-ReasonAlignment：`11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a`。
- FlashRAG：`7926ac3c3c5ce0781f53d60ebac48fc511abed12`。

## 两个仓库分别提供什么

RAG-ReasonAlignment 的 README 说明，RAG 流程基于 FlashRAG，训练基于 LLaMA-Factory。`eval/inference_vllm.py` 将实验分为初始 rationale、批判、修正等阶段，通过文件传递阶段输出；`train/train_llama_rationale.yaml` 提供 SFT/LoRA 训练配置。它适合作为后续批判数据合成、批判模型监督训练与修正流程的参考。不能仅凭这些脚本宣称已经完成课程要求的自动停止和 RL 优化。

FlashRAG 提供可组合的检索、提示词、生成和评测组件。`SequentialPipeline.run()` 的主要链路是检索、保存结果、构建提示、生成、评测；refiner 是可选组件。注意同一类的 `naive_run()` 注释明确是 without RAG，即无检索直接生成，与本项目所说的“朴素 RAG”不是同一个含义。

来源：[ReasonAlignment README](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/README.md)、[分阶段推理](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/eval/inference_vllm.py)、[SFT 配置](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/train/train_llama_rationale.yaml)、[FlashRAG pipeline](https://github.com/RUC-NLPIR/FlashRAG/blob/7926ac3c3c5ce0781f53d60ebac48fc511abed12/flashrag/pipeline/pipeline.py)。

## 对当前方案的具体影响

| 方面 | 参考实现观察 | 当前实现与取舍 |
|---|---|---|
| 预检索数据 | ReasonAlignment 从 `ctxs` 构建上下文；离线检索另有 `data/retrieve.py` | 继续使用题目自带 `ctxs`，不立即建立全维基索引 |
| 文档选择 | ReasonAlignment 使用前 n_docs，并根据前两个 score 的比较决定是否反转选中文档 | 新增 `input` 模式保留输入顺序；不在分数含义未确认时复制反转启发式；原 BM25 模式仍可用 |
| 普通回答 | FlashRAG quick-start 要求只给答案；ReasonAlignment 的 cot 分支生成证据解释 | 当前使用一次生成的直接短答案；单次生成 CoT 也可以没有批判，但不在本次新增 |
| 输入长度 | 两者均有最大输入长度和截断相关代码 | 保留显式 token 预算；正文按篇分配预算是本项目的设计，不是对参考实现的原样复现 |
| 生成框架 | FlashRAG 有 HF/vLLM 等生成器；ReasonAlignment 推理脚本直接使用 vLLM | 当前只实现 HF 和流程检查后端；尚未接入 vLLM，不声称复现其吞吐 |
| 评测 | FlashRAG 分别提供 EM、F1、子串 acc；ReasonAlignment 检查答案是否出现在 rationale 中 | 保留严格 EM/F1，新增单列 `substring_accuracy`，不把宽松命中率叫作 Kaggle Accuracy |
| 过程存档 | FlashRAG 保存 retrieval_result、prompt、pred 等 | 继续保存文档、提示、模型原始输出、初始与最终答案、修正次数和时间 |

来源：[ReasonAlignment 上下文构建](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/eval/data_utils.py)、[FlashRAG quick-start](https://github.com/RUC-NLPIR/FlashRAG/blob/7926ac3c3c5ce0781f53d60ebac48fc511abed12/examples/quick_start/simple_pipeline.py)、[FlashRAG 提示构建](https://github.com/RUC-NLPIR/FlashRAG/blob/7926ac3c3c5ce0781f53d60ebac48fc511abed12/flashrag/prompt/base_prompt.py)、[FlashRAG 指标](https://github.com/RUC-NLPIR/FlashRAG/blob/7926ac3c3c5ce0781f53d60ebac48fc511abed12/flashrag/evaluator/metrics.py)、[ReasonAlignment 指标](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/eval/metrics.py)。

## 不能直接照搬的地方

静态阅读发现 ReasonAlignment 的 README 示例参数与当前 argparse 并不完全一致，数据路径还包含实验目录硬编码。`nocot` 分支引用的 `NoCoTRAGPromptTemplate` 在该版本 `eval/data_prompt.py` 中没有对应定义，保存结果还要求 `answers` 字段。不能未经适配就直接跑无标签 Kaggle 测试集；这些是源代码检查结论，不是实测报错。

其 YAML 虽然配置了 `generator_max_input_len=6144`，但推理脚本另行初始化 vLLM，设置 `max_model_len=32768`、输出上限 5120，并按模型名称指定采样参数；主 rationale chat 分支直接返回消息列表，不能仅凭 YAML 断言该路径始终强制 6144 token 输入上限。当前 baseline 则在送入模型前显式检查实际 chat template token 数。

FlashRAG 的非 API 字符串提示截断采用保留头尾的策略；这与当前“给各篇文档分配正文预算”不同，两者均可能丢失关键证据，不能宣称某种策略未经实验就一定更优。

来源：[ReasonAlignment 推理入口](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/eval/inference_vllm.py)、[ReasonAlignment 提示代码](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/eval/data_prompt.py)、[ReasonAlignment 配置](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/configs/eval.yaml)、[FlashRAG 截断](https://github.com/RUC-NLPIR/FlashRAG/blob/7926ac3c3c5ce0781f53d60ebac48fc511abed12/flashrag/prompt/base_prompt.py)。

## 建议的实验顺序

进一步参考数据集经典实验后，正式主配置 `baseline.json` 已改为 input 模式；BM25 对照改用 `baseline_bm25.json`，`baseline_precomputed.json` 与主配置采用相同设置。先在固定验证集比较主配置与 BM25 对照，冻结配置后再接入批判模型。新的完整方案见 [经典 baseline 适配设计](BASELINE_DESIGN.md)。当前仍是独立的轻量实现，尚未迁移到 FlashRAG 框架，也不是对 ReasonAlignment 的完整复现。
