#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import sys,torch; assert sys.version_info >= (3,10), "Use Python >=3.10"; assert tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) >= (2,6), "Use PyTorch >=2.6"; assert torch.cuda.is_available(), "Choose an AutoDL PyTorch CUDA image"; print("Existing PyTorch:",torch.__version__,"CUDA:",torch.version.cuda)'
"$PYTHON" -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Keep the CUDA-compatible PyTorch provided by the AutoDL image. Do not reinstall CUDA.
python -m pip install -e '.[gpu]'
python -m unittest discover -s tests -v
python -m autodl_rag --config "${CONFIG:-configs/autodl.json}" doctor
echo 'Installed. For each new terminal: source .venv/bin/activate'
