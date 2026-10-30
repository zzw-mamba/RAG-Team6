# vLLM 连续批处理候选

用于 score-v2 普通 RAG，目标是绕开 HF 固定批次等待和逐 token Python JSON 过滤开销。
固定 vLLM 0.11.2，独立虚拟环境；本机 HTTP 连续调度、xgrammar JSON Schema、CUDA graph、前缀缓存。仅服务端加载一份 Qwen BF16 权重，客户端不加载 GPU 权重，不调用商业 API。

这是新的推理精度/引擎，不保证任何提速倍率或准确率。必须真实验证后使用。

## 环境和数据位置

- 原项目 .venv 保持不变，不向其中安装 vLLM；新的 `/root/autodl-fs/zzw/.venv-vllm` 使用独立依赖。
- vLLM 与 Torch/CUDA 二进制有配套要求，因此新环境允许安装自己的依赖，不复用旧环境 Torch。首次安装需要下载数 GB 包并有相应磁盘余量；TMP、pip 临时文件和编译缓存都放在 autodl-fs。
- 继续读取现有 Qwen2.5-7B-Instruct 原始 safetensors，不下载另一份模型。
- 服务仅监听 127.0.0.1:8106，默认使用 65% GPU 显存预算、最多 32 个序列、18000 context。新环境支持情况需目标 Blackwell GPU 实测；首次内核编译和图捕获可能较慢。
- 本地 tokenizer 先使用原 fit_messages 分配证据预算，再把完整 prompt token IDs 发给服务端，禁用额外 special tokens，并核对返回 prompt_tokens。每题使用独立引用 enum 的 JSON Schema；不因结构化失败降级成自由文本，不伪造 token IDs。
- 远端客户端使用既有 plan-before-answer 控制器，公开字段和有序写出规则不变。并发请求不再受客户端整批完成屏障限制，由 vLLM 连续调度。

## 先试跑

在服务器拉取 feat/rag-score-v2 后执行：

```bash
cd /root/autodl-fs/zzw/RAG-Team6-fixed/autodl
mkdir -p /root/autodl-fs/zzw/logs
nohup bash run_score_v2_vllm.sh trial /root/autodl-fs/zzw/configs/new-server.json \
  > /root/autodl-fs/zzw/logs/vllm-trial.log 2>&1 < /dev/null &
```

trial 顺序执行独立环境安装、停止相同实验的旧预测、启动本机服务、32 题真实验证；不自动全量重跑。安装阶段原任务可继续，服务启动前才停止旧预测并等待写出锁释放。其他项目任务不受影响。
前置条件：score-v2 的 tune/confirmation 已完成，且已有相同配置/模型/数据的 HF batch 32 题真实报告。

它选择已通过 HF 报告中测得最快的一组，读取其真实 batch.jsonl，核对与当前 confirmation 前 32 题 ID 相同，然后比较 vLLM 的 EM/F1 和答案变化。HF 计时来自之前的真实运行，GPU 负载可能不同，因此速度比是参考值，不是同时同场的严格性能实验。
每次试跑使用独立答案缓存与独立 cache_salt，防止读取旧答案或旧题目的完整 KV 前缀制造虚假提速。Schema 编译/HTTP/检索均计入新试跑耗时。

报告在 `work_dir/optimization/score-v2/vllm-validation/concurrency-32.json`，包括准确率、每小时题数、历史 HF 对比、旧预测完成数、旧任务预计剩余时间以及 vLLM 全量重跑预计时间。
默认要求同题 EM 不下降，且速度至少达到历史最快 HF 的 1.5 倍；32 题只是 pilot，不构成统计显著性或 Kaggle 得分保证。

查看日志：

```bash
tail -f /root/autodl-fs/zzw/logs/vllm-trial.log
tail -f /root/autodl-fs/zzw/logs/vllm-server.log
```

## 通过后预测

```bash
nohup bash run_score_v2_vllm.sh predict /root/autodl-fs/zzw/configs/new-server.json \
  > /root/autodl-fs/zzw/logs/vllm-predict.log 2>&1 < /dev/null &
```

BF16 与原来的 NF4 不混用：新结果是 `<selected>/test.vllm.jsonl` 和 `test.vllm.csv`，保留原 test.jsonl/test.csv。首次需要重新预测全部测试题；新的 BF16 运行中断后可以续跑自己的前缀。
因此要同时比较新全量重跑时间与旧任务剩余时间，旧任务接近完成时，换引擎未必值得。

`all` 可安装、启动、验证通过后自动预测。`install` 仅安装；`serve` 仅启动；`validate` 在已启动的服务上重测。默认 stage 是 trial。
可调环境变量：VLLM_CONCURRENCY=32、VLLM_MEMORY_UTILIZATION=0.65、VLLM_PORT=8106、VLLM_GPU=0、VLLM_PILOT_SIZE=32、VLLM_MIN_SPEEDUP=1.5。
pilot 题数必须与已有 HF 报告相同。服务端资源配置变更需重启自己的服务并重新验证；不能直接复用旧精度/部署的验证记录。

## 服务管理

部署记录在 `/root/autodl-fs/zzw/vllm/server.json`；仅在服务实际就绪后发布，包含模型文件身份、BF16 精度、启动参数、依赖版本及 GPU 信息。
同部署再次启动会复用已运行的服务；不同配置不会强行覆盖或停止现有服务。
停止自己的服务：

```bash
/root/autodl-tmp/zzw/RAG-Team6/autodl/.venv/bin/python - <<'PY'
import json, os, signal
from pathlib import Path
p = Path('/root/autodl-fs/zzw/vllm/server.pid.json')
record = json.loads(p.read_text())
pid = record['launcher']
cmd = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
if not any(Path(arg).name == 'vllm_server.py' for arg in cmd):
    raise SystemExit('记录的 PID 已不属于这个服务，请检查进程')
if '--config' not in cmd:
    raise SystemExit('进程缺少项目配置，拒绝停止')
cfg = Path(cmd[cmd.index('--config') + 1])
if not cfg.is_absolute():
    cfg = Path(f'/proc/{pid}/cwd').resolve() / cfg
if str(cfg.resolve()) != record['config']:
    raise SystemExit('进程属于其他项目配置，拒绝停止')
os.kill(pid, signal.SIGTERM)
PY
```

先停止客户端，再停止服务。若 CUDA graph/编译在特定 GPU 上失败，可显式设置 VLLM_ENFORCE_EAGER=1 后启动服务再测；这不是自动降级，也不承诺仍有同等速度。

## 验证范围与官方依据

本地测试覆盖传输字段、原始 token prompt、逐题 JSON 引用范围、模型/版本/预算核对、无本地权重加载、线程有序写出、CSV 导出和历史真实报告身份检查。测试使用明确标注的合成传输数据，不是研究成绩。
尚未在目标 CUDA GPU 上安装/运行 vLLM 或实测 7B 吞吐；该步骤由 trial 完成。旧 HF 路径和模型均保留。

- [vLLM 0.11.2 安装与 Blackwell/CUDA 要求](https://docs.vllm.ai/en/v0.11.2/getting_started/installation/gpu/)
- [结构化输出](https://docs.vllm.ai/en/v0.11.2/features/structured_outputs/)
- [Completions API 扩展参数](https://docs.vllm.ai/en/v0.11.2/serving/openai_compatible_server/)
