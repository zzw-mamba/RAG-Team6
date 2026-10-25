"""Install a missing CUDA PyTorch into the selected project interpreter only."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import re
import subprocess
import sys

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
MIN_IMAGE_TORCH = (2, 2, 1)


def inspect_torch():
    try:
        torch = importlib.import_module("torch")
        return {
            "status": "present",
            "python": sys.executable,
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return {"status": "missing", "python": sys.executable}
        return {"status": "broken", "python": sys.executable, "error": str(exc)}
    except (ImportError, OSError, RuntimeError) as exc:
        return {"status": "broken", "python": sys.executable, "error": str(exc)}


def install_needed(state, minimum=(2, 6)):
    if state["status"] == "missing":
        return True
    if state["status"] != "present":
        raise RuntimeError(
            f"PyTorch import is broken: {state}. Check this environment before training."
        )
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", str(state["version"]))
    if not match:
        raise RuntimeError(f"Unrecognized torch version: {state['version']}")
    version = tuple(map(int, match.groups()))
    if not state["cuda_runtime"] or version < minimum:
        return True
    if not state["cuda_available"]:
        raise RuntimeError(
            "This interpreter has a CUDA PyTorch build, but CUDA is unavailable. "
            "Check nvidia-smi / GPU rental state / driver access before training. "
            f"Environment: {state}"
        )
    return False


def pip_command(executable, index, installed_extras):
    packages = ["torch==2.8.0"]
    # Shadow inherited incompatible optional packages in the venv, if present.
    if "torchvision" in installed_extras:
        packages.append("torchvision==0.23.0")
    if "torchaudio" in installed_extras:
        packages.append("torchaudio==2.8.0")
    return [executable, "-m", "pip", "install", "--upgrade", *packages, "--index-url", index]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", default=TORCH_INDEX)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--reuse-image",
        action="store_true",
        help="Accept CUDA torch >=2.2.1; never download or replace torch",
    )
    args = parser.parse_args(argv)
    state = inspect_torch()
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    minimum = MIN_IMAGE_TORCH if args.reuse_image else (2, 6)
    needed = install_needed(state, minimum)
    if needed and args.reuse_image:
        raise RuntimeError(
            "The Qwen image entry needs its existing working CUDA PyTorch >=2.2.1. "
            "Use the image Python; torch will not be downloaded by this entry."
        )
    if args.verify_only:
        if needed:
            raise RuntimeError("CUDA PyTorch >=2.6 is still missing in this interpreter")
        print("CUDA PyTorch ready.", flush=True)
        return
    if not needed:
        print("Keeping the existing working CUDA PyTorch.", flush=True)
        return
    if platform.system() != "Linux":
        raise RuntimeError("Run this CUDA installation on the AutoDL Linux server")
    extras = [name for name in ("torchvision", "torchaudio") if importlib.util.find_spec(name)]
    command = pip_command(sys.executable, args.index_url, extras)
    print(
        "Installing CUDA PyTorch into this project environment; download may be large.", flush=True
    )
    subprocess.run(command, check=True)
    # A CPU/old torch may already be imported above. Verify newly installed files
    # in a new process, rather than accidentally inspecting the old module cache.
    subprocess.run([sys.executable, "-m", "autodl_rag.bootstrap", "--verify-only"], check=True)


if __name__ == "__main__":
    main()
