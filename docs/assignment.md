# 作业与比赛约束

核实日期：2026-10-16；来源为用户提供的课程 PDF，以及登录后的私有 Kaggle 页面。

## 必须完成

- 2WikiMultiHopQA 多跳问答。
- Qwen2.5-7B-Instruct 回答，LLaMA-3.1-8B-Instruct 训练为批判模型。
- 文档检索、初答、批判、反馈修正、自动停止。
- 强化学习进一步优化批判模型，自行设计奖励与训练方案。
- 比较普通 RAG、批判修正、RL 优化三组。
- 报告答案 Accuracy、平均修正次数、推理时间与典型成功/失败案例。
- Demo 展示问题、文档、初答、批判、修正、最终答案。

## Kaggle 已确认

| 项目 | 内容 |
|---|---|
| 截止 | 北京时间 2026-10-30 00:00，即 10 月 29 日结束时 |
| 队伍 | 最多 10 人，每人只能加入一队 |
| 文件 | train.json、test.json、demos.json、sample_submission.csv |
| 大小 | 训练约 1.65GB，测试约 100.72MB，总约 1.75GB |
| 测试结构 | id、question、ctxs，无标准答案 |
| 训练预览 | question、answers、qa_pairs、rationale、ctxs |
| 候选文档 | 已查看样本每题 10 篇，未做全量核实 |
| 提交 | 12,576 条预测＋表头，id/answer 两列，不改 ID |
| 排名 | 最终答案 Accuracy，最终以 Private Leaderboard 为准 |
| 最终提交 | 最多选择 2 份 |
| 日限额 | 规则写 10 次，读取时窗口显示剩余 5 次，以实际窗口为准 |
| 数据页许可 | CC BY-NC-SA 4.0 |

比赛结果应来自本组实现的系统。建议 GitHub 比赛期间保持 Private，邀请组员协作。

## 仍待确认

1. Kaggle Accuracy 的归一化、别名等具体匹配规则。
2. 完整训练样本数、ctxs 内部键名、全量候选文档分布。
3. 是否允许额外检索语料、外部教师模型、额外训练数据。
4. Public/Private 测试比例与最终课程报告格式。

现有 rationale 有质量风险，不能自动当作 gold reasoning。本项目也不利用公开原数据中的答案回填比赛 test。

来源：[Overview](https://www.kaggle.com/competitions/rag/overview)、[Data](https://www.kaggle.com/competitions/rag/data)、[Rules](https://www.kaggle.com/competitions/rag/rules)、[Submissions](https://www.kaggle.com/competitions/rag/submissions)。

课程后续通知和比赛页面变更需要人工重新核实；本文件不是自动监控。
