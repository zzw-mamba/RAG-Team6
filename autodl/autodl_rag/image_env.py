"""Create a project venv inheriting the Qwen image's CUDA torch, never download it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .bootstrap import inspect_torch, install_needed
from .core import read_json, save_json

PROJECT = Path(__file__).resolve().parent.parent


def dependency_commands(python, requirements, index):
    # The file includes the complete non-Torch dependency closure. Disabling
    # resolution is deliberate: pip must never pull a new torch/NVIDIA runtime.
    return [
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            "--index-url",
            index,
            "pip>=25.2,<27",
        ],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--index-url",
            index,
            "-r",
            str(requirements),
        ],
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "-e",
            str(PROJECT),
        ],
    ]


def prepare_venv(project, identity):
    project = Path(project)
    venv = project / ".venv"
    marker = venv / ".qwen-image.json"
    matches = marker.is_file() and read_json(marker) == identity
    if venv.exists() and not matches:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = project / f".venv.backup-{stamp}-{os.getpid()}"
        venv.rename(backup)
        print(f"Preserved previous project environment: {backup}", flush=True)
    if not (venv / "bin/python").is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            check=True,
        )
        save_json(marker, identity)
    return venv / "bin/python"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", required=True)
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        raise RuntimeError("Use the Qwen image's Python >=3.10")
    state = inspect_torch()
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if install_needed(state, minimum=(2, 2, 2)):
        raise RuntimeError("Image CUDA torch >=2.2.2 missing; no torch download attempted")
    import torch

    identity = {
        "image_prefix": sys.prefix,
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
    }
    python = prepare_venv(PROJECT, identity)
    for command in dependency_commands(
        python, PROJECT / "environment/requirements-qwen-image.txt", args.index_url
    ):
        subprocess.run(command, check=True)
    # A stale venv or manual package installation must not shadow the image torch.
    code = (
        "import json,torch; from pathlib import Path; "
        "print(json.dumps({'torch_file':str(Path(torch.__file__).resolve()),"
        "'torch_version':torch.__version__,'cuda':torch.version.cuda}))"
    )
    actual = json.loads(subprocess.check_output([str(python), "-c", code], text=True))
    expected = {k: identity[k] for k in actual}
    if actual != expected:
        raise RuntimeError(f"Project torch differs from image: {actual}, expected {expected}")
    subprocess.run(
        [str(python), "-m", "autodl_rag.bootstrap", "--verify-only", "--reuse-image"],
        check=True,
    )
    print("Reused image torch/CUDA. Project dependencies ready.", flush=True)


if __name__ == "__main__":
    main()
