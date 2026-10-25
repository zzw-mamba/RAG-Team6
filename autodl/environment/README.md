# 可复用环境：离线依赖包与容器

这个项目的本地Qwen/LLaMA推理、LoRA反向传播和GRPO更新使用PyTorch。环境打包会把它预装或准备好，不会消除这个依赖。数据处理、BM25检索和部分测试可单独不用Torch。

仓库包含环境配方、下载和离线安装工具，**不包含数GB的CUDA二进制包，也不包含已经构建好的Docker镜像**。第一次准备依赖或构建镜像仍需要网络。模型权重与数据继续独立存放。

## 1. 对当前AutoDL：准备一次离线包，之后复用

当前报错的服务器是Linux x86_64、Python3.10。请在**相同Python主/次版本的Linux x86_64**机器上准备包；该机器无需GPU，但网络要能完成一次下载。不要直接在macOS上下载一套包拿去Linux安装。

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
tar -cf rag-env-linux-py310.tar -C .cache wheelhouse
```

在AutoDL将它解压到`/root/autodl-tmp/rag-env/`，然后执行：

```bash
mkdir -p /root/autodl-tmp/rag-env
tar -xf /path/to/rag-env-linux-py310.tar -C /root/autodl-tmp/rag-env
cd /root/autodl-tmp/RAG-Team6/autodl
WHEELHOUSE=/root/autodl-tmp/rag-env/wheelhouse bash run_all.sh
```

安装阶段通过`--no-index --find-links`只读取本地依赖，不访问包索引；不继承Miniconda base中的无关包。之后继续GPU检查、训练和评估。离线包**不包含Python解释器、系统驱动或模型**；目标机器仍需相同Python主/次版本和可用NVIDIA驱动。Python3.11/3.12服务器需分别在对应版本的Linux机器准备依赖包。

若先在这台AutoDL上把依赖包下载完成，也可直接使用：

```bash
bash scripts/download_env.sh
WHEELHOUSE="$PWD/.cache/wheelhouse" bash run_all.sh
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

AutoDL普通实例本身就是容器，不能假定其中还具备启动Docker的权限。当前实例优先用离线依赖包或平台保存镜像；Docker文件供可自行管理Docker的主机使用。这里只交付构建配置，没有在目标GPU构建/运行镜像。固定版本标签不是不可变镜像摘要，需要严格固定镜像内容时，应另记录构建结果的digest。

## 3. AutoDL.Art 的 Qwen2.5-7B 镜像能否直接用

核对日期：2026-10-25。[用户提供的 v1.0 镜像页面](https://www.autodl.art/i/QwenLM/Qwen2.5/Qwen2.5-7B-Instruct/1327/1.0)标注 **PyTorch 2.2.2、CUDA 11.8、23.43GB**，提供的是 Qwen 聊天 WebUI。页面给出的环境和模型存放位置分别是：

```text
/root/Qwen2.5/text-generation-webui/installer_files/env
/root/Qwen2.5/text-generation-webui/models/
```

所以它可以作为基础，但不能原样满足当前项目的完整 LoRA/GRPO 训练环境：项目检查要求 CUDA PyTorch >=2.6，仍需升级依赖；本页也未说明包含 LLaMA-3.1-8B-Instruct。页面的“显存>=16G”是该 Qwen 示例的要求，不是本项目两模型训练的显存保证。

如果你当前实例来自这个镜像，先查看实际环境（不会启动聊天服务）：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/Qwen2.5/text-generation-webui/installer_files/env
python -c 'import sys,torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
```

激活后能找到 Torch，可以解释 base Python 报“没有torch”；但若版本仍为2.2.2，激活本身不会解决项目版本要求，也不能消除升级时的下载需求。若希望免下载大体积Torch包，应选实际自带可用 CUDA PyTorch >=2.6 的环境（本项目打包配方用2.8.0/cu128），或使用上面的离线依赖包。

使用这个镜像的 Qwen 权重时，复制配置并指定其 models 目录：

```bash
cd /root/autodl-tmp/RAG-Team6/autodl
python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path('configs/autodl.json').read_text())
cfg['qwen_path'] = '/root/Qwen2.5/text-generation-webui/models'
Path('configs/autodl-webui.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
PY
```

该目录下必须只有一个符合架构要求的模型；多模型时应将 `qwen_path` 改为所需模型的具体目录。实际权重格式未由网页核实，需要有完整 HF config、tokenizer 和原始 BF16/FP16 权重；GGUF、AWQ、GPTQ 不能直接用于本项目当前的 NF4 加载入口。另将 LLaMA 权重和数据放到配置指定位置。准备好环境包后，可执行 `CONFIG=configs/autodl-webui.json WHEELHOUSE="$PWD/.cache/wheelhouse" bash run_all.sh`。

镜像页面的 `python server.py` 用来启动聊天界面；本项目使用自己的训练和推理入口，无需同时启动该服务。

## 4. 复现范围

`requirements-gpu.txt`固定直接依赖；具体准备成功的一套wheel还会固定全部解析出的间接依赖及其校验值。更改依赖要求后应使用新的wheelhouse目录。wheelhouse和镜像属于大体积运行产物，不提交GitHub。

依赖完整性、源选择和版本/平台拒绝逻辑有CPU测试；未在这台Mac上下载Linux/CUDA的完整依赖包，也未声明完成真实离线GPU安装。Docker构建和CUDA运行仍需目标主机验证。

官方说明：[pip离线安装](https://pip.pypa.io/en/latest/user_guide/#installing-from-local-packages)、[PyTorch发布版本与CUDA源](https://pytorch.org/get-started/previous-versions/#v280)、[AutoDL环境说明](https://api.autodl.com/docs/env/)。
