# 交付验证记录

日期：2026-10-25（北京时间）。

## 已完成的本地验证

- 本次修复在实际PyTorch 2.2.1下通过AutoDL模块49项测试，包含7项镜像环境复用测试；原框架39项测试通过。此前48项测试已在PyTorch 2.2.2和2.8.0两套CPU环境分别通过。
- 当前兼容测试实际使用PyTorch 2.2.1、NumPy 1.26.4、Transformers 4.45.2、PEFT 0.13.2、Accelerate 0.34.2；同时回归PyTorch 2.8.0、Transformers 4.56.2、PEFT 0.17.1、Accelerate 1.10.1。均在CPU上运行，不下载预训练权重。
- 已确认completion token概率与完整因果前向一致，温度一致；LoRA获得非零梯度并发生变化；基座参数保持不变。
- 已确认禁用LoRA的基座reference及warm-start reference适配器不随actor更新而变化。
- 已通过真实优化器、LoRA保存/加载、随机状态与日志恢复的两组微型训练测试：从第1组恢复后的最终适配器与连续训练2组逐张量完全相同。
- ijson 3.4.0流式读取路径通过测试；不安装可选GPU依赖时，纯CPU核心测试仍可运行。
- Ruff检查通过；11个Shell脚本通过Bash语法检查；CLI帮助可直接运行。
- Python wheel可成功构建，包含默认配置资源。
- 已修复Miniconda base没有Torch时安装脚本在创建项目环境之前退出的问题；安装决策测试不实际下载CUDA wheel，仍需在服务器验证安装。
- 新增离线wheelhouse生成/校验、离线安装分支及Dockerfile；没有在本地Mac上准备完整Linux CUDA包，也没有实际构建/运行Docker镜像。相关CPU测试不替代完整离线GPU安装验证。

- 已用pip dry-run确认兼容清单全部25个包存在Linux x86_64/Python3.10二进制安装包，并检查包元数据：固定清单加镜像Torch 2.2.1满足全部启用的依赖要求。未在本地下载这些Linux安装包或执行CUDA安装。
- 镜像入口对缺失/过旧/不可用Torch在安装前失败，不调用Torch下载；旧项目环境备份保留，匹配镜像的环境复用；原一键入口自动检测镜像并选择配置，显式modern配置仍可用。CI加入两套CPU张量依赖矩阵，远端CI结果待运行。

- 用户实机日志为2.2.1+cu118，网页标注为2.2.2。已统一安装入口、bootstrap验证和doctor的最低版本常量为2.2.1；回归覆盖2.2.1/2.2.2接受、2.2.0拒绝、CPU/无GPU拒绝，以及通过2.2.1完整安装控制流程而不下载Torch。CI的镜像兼容任务改为真实2.2.1。

## 验证含义

上述合成数据、预设响应和微型随机模型仅测试程序，不代表比赛准确率或7B/8B模型表现；测试工具不会进入正式推理作为失败兜底。

未在用户AutoDL上加载实际7B/8B权重，未执行bitsandbytes CUDA量化内核，未测目标GPU峰值显存、速度或准确率。本地CPU测试不替代这些GPU检查。

## AutoDL上接着执行

```bash
cd /root/autodl-tmp/RAG-Team6
git fetch origin
git switch feat/autodl-qwen-image
git pull --ff-only
cd autodl
bash run_qwen_image.sh
```

`doctor.json`、`smoke/train/backward_probe.json`和`smoke/train/summary.json`分别确认环境、实际GPU反向传播和训练更新。只有在真实模型的验证/确认结果产生后，才能评价准确率；官方测试准确率还需要Kaggle评分。
