# 协作约定

## 第一次加入

1. 接受仓库邀请，克隆仓库，按 README 安装 Python 3.11/3.12 开发环境。
2. 运行单元测试和一次 Mock smoke；无需先下载模型。
3. 在 Issue 中认领角色和任务，填清输入、输出、验收条件。
4. 对应模块接口参考 `docs/architecture.md`。真实数据从 Kaggle 各自获取，不进 Git。

## 分支与 PR

- `main` 保持可以安装、通过 CI；建议禁止直接推送。
- 分支如 `feat/data-adapter`、`feat/critic-sft`、`fix/reward-stop`、`docs/report`。
- 一项改动一个 PR，关联 Issue，说明行为变化与验证证据。
- 修改共享 schema、输出格式、训练数据字段时，提前在对应 Issue 讨论并同步使用方。
- 合并前至少 1 位其他同学审查；公共接口与奖励设计建议 2 人审查。
- 推荐 squash merge；不要同时在多人分支重写公共历史。

```bash
git switch main
git pull --ff-only
git switch -c feat/your-task
# 完成工作后
python -m ruff check .
python -m unittest discover -s tests -v
git add <本次修改的文件>
git commit -m "feat: describe the concrete change"
git push -u origin feat/your-task
```

GitHub 上创建 PR，等待 CI 与代码审查，不在自己的功能分支直接改其他同学正在维护的接口。

## 必须保留的研究边界

- 回答模型和批判模型的推理输入不得包含标准答案、训练 rationale 或支持事实标注。
- 先划分训练/验证，再合成数据。相同问题不得跨集合；验证集不得用于奖励更新。
- `fixtures/` 只放虚构样本；现有 Mock 分数不是实验结果。
- `rationale` 没有自动视为正确，批判样本要记录审阅者与来源。
- 网络错误、模型载入失败不得降级为 Mock 或奖励为零后继续训练。
- 新检索器改变主实验条件时，三组实验一起重跑或另列消融。

## 文件与实验记录

- `data/raw`、`data/processed`、`data/annotations`、`checkpoints`、`outputs` 已忽略。
- 新模型或大数据应通过团队约定的存储共享，Git 中只留配置、版本和校验值。
- `.env.example` 只写变量名。不要提交 Token、账号 cookie、`kaggle.json`。
- 云端 CUDA 环境稳定后将 `python -m pip freeze` 输出存到该实验目录，记录 GPU、服务版本与训练命令。
- 日志目录不复用；断点续跑仅用于同一输入、配置、代码。恢复时会丢弃最后一条未写完的 JSONL 行并重跑该题。
- 若修改提示词语义，同步增加 `PROMPT_VERSION`，方便追溯。

## GitHub 管理员首次设置

1. 新建 Private 仓库、邀请其他 9 位成员，确认所有人在同一 Kaggle 队伍。
2. 开启 Issues，按 `docs/team.md` 创建任务。
3. 为 `main` 设置 PR 合并、至少 1 人审查和 `core` 检查通过；权限/套餐可能影响可用保护功能。
4. 填入 `CODEOWNERS` 的真实账号，不填写虚构用户名。
5. 约定 GPU 使用时段、输出存储位置、适配器命名和最后负责提交的人。

本地框架不会自动创建远端仓库、邀请成员、推送代码或提交比赛答案。
