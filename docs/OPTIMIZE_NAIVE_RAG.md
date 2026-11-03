# 朴素 RAG 的本地提分实验

当前工作继续保持 Qwen2.5-7B-Instruct、一次生成、无批判模型。目标是用固定 dev 的真实结果决定是否换配置，不能预设下面的方案已经提分。现有已运行配置和日志保留。

## 本轮比较

| 名称 | 配置 | 改动与目的 |
|---|---|---|
| baseline | `autodl_baseline_best_supported_batch16.json` | 同批大小重跑当前基线，控制 batch 变化影响 |
| bm25 | `autodl_best_supported_bm25_batch16.json` | 保持短答案提示与预算，仅按问题相关度排列候选 |
| evidence | `autodl_evidence_answer_batch16.json` | 保持原顺序，在一次生成中先写简短证据摘要，再给带标签的最终答案；输出预算为 256 token |

前两组都是 64 token 输出上限，第三组为了容纳证据摘要改为 256，因此该组比较的是“证据摘要提示与相应预算”的整体变化，不把差异全归因于单个词句。

证据摘要参考课程仓库要求模型识别有用文档并解释证据连接的做法：[RAG-ReasonAlignment 提示模板](https://github.com/upup-wei/RAG-ReasonAlignment/blob/11ab28179d6cbd0d0c4cb491a6fd3e1b1c53536a/eval/data_prompt.py)。此处仍是一次生成，不是该仓库的批判修正循环，也不是 IRCoT 的交替补检索。

如果每题有 10 篇候选，两种排序保留相同文档集合，不能提高候选召回。若有更多文档，Top 10 的集合可能改变。BM25 的效果必须实测；不要把它说成已经补齐缺失的第二跳。

当前没有截断的运行，不应先靠增加输入长度求提分。批大小主要影响吞吐，较大 batch 不保证提高答案质量。本文配置在目标 GPU 上尚未测试收益、显存或速度。

## 输出协议

evidence 模式要求模型生成简短证据摘要与一个 `<answer>最终答案</answer>`。日志保留全部原始输出，EM/F1 与提交 CSV 只使用标签内的答案。

缺少闭合标签、多个答案标签或空答案会报错，将原始输出保存到运行目录的 `generation_error.json`。不会把未完成的证据文字当成最终答案，也不会悄悄替换为 unknown。格式问题先在 pilot 排查；更改配置或代码后使用新运行目录。

## 先检查 20 题，再用固定 500 题比较

在新版仓库根目录执行，使用已激活的模型环境：

```bash
git pull --ff-only
export PYTHONPATH="$PWD/src"

# 三种配置顺序运行，避免同一 GPU 上同时加载三份模型
bash scripts/tune_naive_rag.sh data/evaluation-v1/pilot.jsonl outputs/naive-rag-tune-pilot-v1

# pilot 正常后，使用已有的固定 dev，不能重新挑容易的问题
bash scripts/tune_naive_rag.sh data/evaluation-v1/dev.jsonl outputs/naive-rag-tune-v1
```

重复运行相同脚本会对已有 manifest 的运行请求 `--resume`，源代码、配置和输入改变时仍会拒绝续跑。完整报告已存在时直接复用经过指纹检查的报告。一个输出根目录对应一次固定实验，完成后不要手动替换其中报告。

服务器后台运行：

```bash
mkdir -p outputs
nohup bash scripts/tune_naive_rag.sh \
  data/evaluation-v1/dev.jsonl outputs/naive-rag-tune-v1 \
  > outputs/naive-rag-tune-v1.log 2>&1 &

tail -f outputs/naive-rag-tune-v1.log
```

脚本以当前 `python` 运行，可通过 `RAG_PYTHON` 指定解释器。没有 GPU 时，可用第三个参数 `--dry-run` 检查脚本，报告会明确标注不代表模型效果。

## 如何选配置

最终阅读 `outputs/naive-rag-tune-v1/comparison/comparison.md`。比较器核对输入 SHA256、评分版本、后端和问题 ID，逐题统计比基准多答对多少题、原来正确现在答错多少题，输出 `paired_changes.csv` 供抽查；不会将不同题集上的分数直接相减。

优先看 dev 的归一化 EM，其次看 F1、净增正确数、误改题、拒答、格式与速度。只减少拒答或增加宽松命中率不算有效提升。若新方案未胜过基准，就保留现有配置，不强行替换。

本轮不使用已查看过的 holdout 反复选参，也不根据 Kaggle 公开榜反复试提示。选出候选方案后，可从原保留分区中未使用的问题建立新的确认集，再生成一次全量测试预测。

若上述单轮阅读改动仍无法改善，下一步应优先人工审查失败样例，区分真实证据缺失和阅读失败。补充检索需要确定允许的语料与索引，并作为独立检索增强实验；不能把训练问题答案索引进去，也不能用金标准答案选择测试文档。
