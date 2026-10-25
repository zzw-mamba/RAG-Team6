# 交付验证记录

日期：2026-10-25（北京时间）。

## 已完成的本地验证

- 31项测试通过：21项数据/检索/解析/奖励/缓存/流程/导出测试，4项真实微型LLaMA与PEFT张量测试，6项PyTorch缺失检测、安装决策与新进程验证测试。
- 微型模型测试使用PyTorch 2.8.0、Transformers 4.56.2、PEFT 0.17.1、Accelerate 1.10.1，在CPU上运行，不下载预训练权重。
- 已确认completion token概率与完整因果前向一致，温度一致；LoRA获得非零梯度并发生变化；基座参数保持不变。
- 已确认禁用LoRA的基座reference及warm-start reference适配器不随actor更新而变化。
- 已通过真实优化器、LoRA保存/加载、随机状态与日志恢复的两组微型训练测试：从第1组恢复后的最终适配器与连续训练2组逐张量完全相同。
- ijson 3.4.0流式读取路径通过测试；不安装可选GPU依赖时，纯CPU核心测试仍可运行。
- Ruff检查与格式检查通过；7个Shell脚本通过Bash语法检查；CLI帮助可直接运行。
- Python wheel可成功构建，包含默认配置资源。
- 已修复Miniconda base没有Torch时安装脚本在创建项目环境之前退出的问题；安装决策测试不实际下载CUDA wheel，仍需在服务器验证安装。

## 验证含义

上述合成数据、预设响应和微型随机模型仅测试程序，不代表比赛准确率或7B/8B模型表现；测试工具不会进入正式推理作为失败兜底。

未在用户AutoDL上加载实际7B/8B权重，未执行bitsandbytes CUDA量化内核，未测目标GPU峰值显存、速度或准确率。本地CPU测试不替代这些GPU检查。

## AutoDL上接着执行

```bash
bash scripts/install.sh
bash scripts/prepare.sh
bash scripts/smoke.sh
bash scripts/train_pilot.sh
bash scripts/evaluate.sh
```

`doctor.json`、`smoke/train/backward_probe.json`和`smoke/train/summary.json`分别确认环境、实际GPU反向传播和训练更新。只有在真实模型的验证/确认结果产生后，才能评价准确率；官方测试准确率还需要Kaggle评分。
