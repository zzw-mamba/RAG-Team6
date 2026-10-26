"""Create a project venv inheriting the Qwen image's CUDA torch, never download it."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .bootstrap import MIN_IMAGE_TORCH, inspect_torch, install_needed
from .core import digest_file, read_json, save_json

PROJECT = Path(__file__).resolve().parent.parent
REQUIREMENTS = PROJECT / "environment/requirements-qwen-image.txt"


def requirement_specs(requirements=REQUIREMENTS):
    from packaging.requirements import Requirement

    return [
        Requirement(line.strip())
        for line in Path(requirements).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def validate_packages(versions, requirements=REQUIREMENTS):
    from packaging.utils import canonicalize_name

    normalized = {canonicalize_name(k): v for k, v in versions.items()}
    errors = []
    for req in requirement_specs(requirements):
        installed = normalized.get(canonicalize_name(req.name))
        if installed is None or installed not in req.specifier:
            errors.append(f"{req}: installed {installed or 'missing'}")
    if errors:
        raise RuntimeError("Project dependencies do not match: " + "; ".join(errors))


def runtime_report():
    """Record effective packages inside the project interpreter."""
    import platform

    import torch

    versions = {}
    for req in requirement_specs():
        try:
            versions[req.name] = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            versions[req.name] = None
    validate_packages(versions)
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "attention_implementation": "sdpa",
        "packages": versions,
        "requirements_sha256": digest_file(REQUIREMENTS),
    }


def dependency_commands(python, requirements, index):
    # Keep satisfying project libraries: no --upgrade in the library command.
    # --no-deps forbids pip from pulling torch or NVIDIA runtime dependencies.
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
    if install_needed(state, minimum=MIN_IMAGE_TORCH):
        raise RuntimeError("Image CUDA torch >=2.2.1 missing; no torch download attempted")
    import torch

    identity = {
        "image_prefix": sys.prefix,
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
    }
    python = prepare_venv(PROJECT, identity)
    for command in dependency_commands(python, REQUIREMENTS, args.index_url):
        subprocess.run(command, check=True)
    # A stale venv or manual package installation must not shadow the image torch.
    code = (
        "import json; from autodl_rag.image_env import runtime_report; "
        "print(json.dumps(runtime_report()))"
    )
    actual = json.loads(subprocess.check_output([str(python), "-c", code], text=True))
    expected = {k: identity[k] for k in ("torch_file", "torch_version", "cuda")}
    if any(actual.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"Project torch differs from image: {actual}, expected {expected}")
    validate_packages(actual["packages"])
    report_path = PROJECT / ".cache/image-environment.json"
    save_json(report_path, actual)
    subprocess.run(
        [str(python), "-m", "autodl_rag.bootstrap", "--verify-only", "--reuse-image"],
        check=True,
    )
    print(f"Reused image torch/CUDA. Effective package report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
