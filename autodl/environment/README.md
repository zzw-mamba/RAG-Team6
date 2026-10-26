# 可复用环境：离线依赖包与容器

这个项目的本地Qwen/LLaMA推理、LoRA反向传播和GRPO更新使用PyTorch。环境打包会把它预装或准备好，不会消除这个依赖。数据处理、BM25检索和部分测试可单独不用Torch。

仓库包含环境配方、下载和离线安装工具，**不包含数GB的CUDA二进制包，也不包含已经构建好的Docker镜像**。第一次准备依赖或构建镜像仍需要网络。模型权重与数据继续独立存放。

## 1. 对当前AutoDL：准备一次离线包，之后复用

最新实际镜像环境为Linux x86_64、Python3.11.9；先前Miniconda base的Python3.10是另一个解释器。请在**相同Python主/次版本的Linux x86_64**机器上准备包；该机器无需GPU，但网络要能完成一次下载。不要直接在macOS上下载一套包拿去Linux安装。

网络正常的Linux机器执行：

```bash
cd RAG-Team6/autodl
bash scripts/download_env.sh
```

默认输出到`.cache/wheelhouse/`。下载脚本使用独立的`.cache/env-builder`环境，不在Miniconda base安装Torch。它先只从PyTorch官方CUDA索引取`torch==2.8.0+cu128`，随后从清华PyPI镜像下载其NVIDIA依赖和其余项目依赖，避免CUDA索引中的`pypi.nvidia.com`链接。

可选择其他依赖索引：

```bash
DEPENDENCY_INDEX_URL=https://pypi.org/simple bash scripts/download_env.sh
```

默认超时120秒、连接重试20次、单次命令内不完整下载恢复重试30次。已经完成的wheel会保留；失败后可重跑。**未完成文件不保证跨命令按字节续传**，增加重试也不能保证任何网络都可下载成功。

成功后目录包含：全部依赖的wheel、逐项固定版本的`resolved-requirements.txt`、目标系统/Python信息及SHA256清单`MANIFEST.json`。只有两轮下载都成功才生成完整清单；目标服务器会检查文件完整性和Python版本。

将整个目录打包搬到AutoDL：

```bash
tar -cf rag-env-linux-py311.tar -C .cache wheelhouse
```

在AutoDL将它解压到`/root/autodl-tmp/rag-env/`，然后执行：

```bash
mkdir -p /root/autodl-tmp/rag-env
tar -xf /path/to/rag-env-linux-py311.tar -C /root/autodl-tmp/rag-env
cd /root/autodl-tmp/RAG-Team6/autodl
RAG_ENV_PROFILE=modern WHEELHOUSE=/root/autodl-tmp/rag-env/wheelhouse bash run_all.sh
```

安装阶段通过`--no-index --find-links`只读取本地依赖，不访问包索引；不继承Miniconda base中的无关包。之后继续GPU检查、训练和评估。离线包**不包含Python解释器、系统驱动或模型**；目标机器仍需相同Python主/次版本和可用NVIDIA驱动。Python3.11/3.12服务器需分别在对应版本的Linux机器准备依赖包。

若先在这台AutoDL上把依赖包下载完成，也可直接使用：

```bash
bash scripts/download_env.sh
RAG_ENV_PROFILE=modern WHEELHOUSE="$PWD/.cache/wheelhouse" bash run_all.sh
```

下载和安装验证成功后，可以用AutoDL平台的保存镜像功能复用已经安装的环境，避免下一台实例重新安装。实际可用镜像选项以平台控制台为准。

## 2. 连Python和CUDA库一起打包：Docker

项目提供`Dockerfile`，基于`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`，预装项目依赖，并创建供阶段脚本使用的`.venv`。

在支持Docker的Linux主机上构建：

```bash
cd RAG-Team6/autodl
docker build -t team6-rag:torch2.8-cu128 .
```

构建好的镜像可保存后搬到其他Docker主机：

```bash
docker save team6-rag:torch2.8-cu128 -o /path/to/team6-rag-image.tar
docker load -i /path/to/team6-rag-image.tar
```

在安装了NVIDIA容器运行支持的主机上运行：

```bash
docker run --rm --gpus all --shm-size=8g \
  -v /root/Qwen2.5:/root/Qwen2.5:ro \
  -v /root/autodl-tmp/models:/root/autodl-tmp/models:ro \
  -v /root/autodl-tmp/data:/root/autodl-tmp/data:ro \
  -v /root/autodl-tmp/rag-team6-work:/root/autodl-tmp/rag-team6-work \
  team6-rag:torch2.8-cu128
```

默认跳过依赖安装，继续准备数据、GPU检查、训练、评估和全量预测。默认配置仍使用你给出的模型和数据路径。镜像不内置数据和模型，依靠挂载读取；宿主机NVIDIA驱动仍不可省略。

AutoDL普通实例本身就是容器，不能假定其中还具备启动Docker的权限。已有 Qwen 镜像的当前实例可先用下一节的直接复用入口；需要整体迁移时可用离线依赖包或平台保存镜像；Docker文件供可自行管理Docker的主机使用。这里只交付构建配置，没有在目标GPU构建/运行镜像。固定版本标签不是不可变镜像摘要，需要严格固定镜像内容时，应另记录构建结果的digest。

## 3. AutoDL.Art 的 Qwen2.5-7B 镜像能否直接用

核对日期：2026-10-25。[用户提供的 v1.0 镜像页面](https://www.autodl.art/i/QwenLM/Qwen2.5/Qwen2.5-7B-Instruct/1327/1.0)标注 **PyTorch 2.2.2、CUDA 11.8、23.43GB**，提供的是 Qwen 聊天 WebUI。页面给出的环境和模型存放位置分别是：

```text
/root/Qwen2.5/text-generation-webui/installer_files/env
/root/Qwen2.5/text-generation-webui/models/
```

用户实机运行日志确认镜像内实际为 **PyTorch 2.2.1+cu118**；网页标注不等于环境内版本。当前入口接受该实机版本，不会为了补丁版本差异重新下载Torch。

本分支按实际conda清单复用Python3.11.9、Torch2.2.1+cu118、NumPy1.26.4、PEFT0.8.2、Accelerate0.27.2、bitsandbytes0.43.1、safetensors0.4.3和HF Hub0.23.3。只在项目环境把Transformers升级到4.45.2、tokenizers升级到0.20.3，并补充ijson3.4.0。Transformers4.40.2对LLaMA 3.1的RoPE字段会报错，这是必要升级；不修改模型的RoPE语义来迁就旧库。兼容范围内的先前项目依赖也会保留，不逐项强制升级/降级。训练代码支持 CUDA PyTorch >=2.2.1，并适配旧版 LLaMA 的 `num_logits_to_keep` 接口。专用入口继承镜像已有 Torch，只安装完整列出的其他依赖，使用 `--no-deps` 阻止安装器解析下载 Torch/NVIDIA 运行库；不改镜像的 WebUI 环境。实际生效的Python、Torch路径、25个项目依赖及其版本会写入 `.cache/image-environment.json`，运行前再次检查兼容范围。

```bash
cd /root/autodl-tmp/RAG-Team6/autodl
bash run_qwen_image.sh
```

无需手动激活镜像环境。项目中的旧 `.venv` 先备份，新 `.venv` 继承镜像 Python 的包。镜像 Torch 不可用时会在安装前停止，不自动换成新 Torch；确认 Python 路径不同后可通过 `IMAGE_PYTHON=/actual/env/bin/python` 指定。`run_all.sh` 检测到这个镜像 Python 时也会自动选择兼容配置；通用新版 Torch 安装/离线包仍通过 `RAG_ENV_PROFILE=modern` 选择。

默认配置 `configs/autodl-qwen-image.json` 将 Qwen 路径指向镜像的 `models` 目录。该目录下须有唯一的目标 HF 模型；多模型时将配置指向具体模型目录。网页未核实实际权重格式，本入口要求完整原始 BF16/FP16 safetensors 和 tokenizer。LLaMA 权重仍从 `/root/autodl-tmp/models/Llama-3.1-8B-Instruct` 读取，数据从 `/root/autodl-tmp/data` 读取；它们需要事先放好。页面的“显存>=16G”是 Qwen 聊天示例的要求，本项目继续按用户的单卡48GB及以上配置并执行 GPU smoke。

完整命令、断点恢复和路径调整见 [项目镜像使用说明](../README.md#直接使用-autodlart-qwen25-镜像pytorch-221--cuda-118)。不需要同时运行聊天 WebUI 的 `python server.py`。

兼容依据：[Qwen 官方模型说明](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)、[Transformers 4.45.2 LLaMA 实现](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/llama/modeling_llama.py)、[PEFT 0.8.2 依赖](https://github.com/huggingface/peft/blob/v0.8.2/setup.py)、[旧Transformers的RoPE限制](https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/configuration_llama.py)、[bitsandbytes 0.43.3 CUDA 支持](https://huggingface.co/docs/bitsandbytes/v0.43.3/installation)。

## 4. 复现范围

`requirements-gpu.txt`固定直接依赖；具体准备成功的一套wheel还会固定全部解析出的间接依赖及其校验值。更改依赖要求后应使用新的wheelhouse目录。wheelhouse和镜像属于大体积运行产物，不提交GitHub。

依赖完整性、源选择和版本/平台拒绝逻辑有CPU测试；未在这台Mac上下载Linux/CUDA的完整依赖包，也未声明完成真实离线GPU安装。Docker构建和CUDA运行仍需目标主机验证。

官方说明：[pip离线安装](https://pip.pypa.io/en/latest/user_guide/#installing-from-local-packages)、[PyTorch发布版本与CUDA源](https://pytorch.org/get-started/previous-versions/#v280)、[AutoDL环境说明](https://api.autodl.com/docs/env/)。
