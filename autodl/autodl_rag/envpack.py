"""Prepare and verify a local Linux/Python-specific CUDA dependency wheelhouse."""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from pathlib import Path

from .core import digest_file, read_json, save_json

REQUIREMENTS = Path(__file__).resolve().parent.parent / "environment/requirements-gpu.txt"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
DEPENDENCY_INDEX = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"


def target():
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": list(sys.version_info[:2]),
    }


def download_commands(directory, requirements=REQUIREMENTS, dependency_index=DEPENDENCY_INDEX):
    shared = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(directory),
        "--only-binary=:all:",
        "--retries",
        "20",
        "--timeout",
        "120",
        "--resume-retries",
        "30",
    ]
    # Fetch only torch from the CUDA index. Fetch its NVIDIA dependencies from
    # the general mirror, avoiding the problematic pypi.nvidia.com links there.
    torch = [*shared, "--no-deps", "--index-url", TORCH_INDEX, "torch==2.8.0+cu128"]
    dependencies = [
        *shared,
        "--find-links",
        str(directory),
        "--index-url",
        dependency_index,
        "-r",
        str(requirements),
    ]
    return torch, dependencies


def seal(directory, requirements=REQUIREMENTS):
    directory = Path(directory)
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        raise ValueError("No downloaded wheels to package")
    versions = {}
    for p in wheels:
        parts = p.name.split("-")
        if len(parts) < 5:
            raise ValueError(f"Invalid wheel filename: {p.name}")
        name = re.sub(r"[-_.]+", "-", parts[0]).lower()
        version = parts[1]
        if name in versions and versions[name] != version:
            raise ValueError(f"Multiple versions of {name}; use a new wheelhouse directory")
        versions[name] = version
    lock = directory / "resolved-requirements.txt"
    lock.write_text("".join(f"{name}=={version}\n" for name, version in sorted(versions.items())))
    manifest = {
        "target": target(),
        "requirements_sha256": digest_file(requirements),
        "resolved_sha256": digest_file(lock),
        "wheels": {p.name: {"bytes": p.stat().st_size, "sha256": digest_file(p)} for p in wheels},
        "contains_python_or_driver": False,
    }
    save_json(directory / "MANIFEST.json", manifest)
    return manifest


def verify(directory, requirements=REQUIREMENTS):
    directory = Path(directory)
    manifest = read_json(directory / "MANIFEST.json")
    if manifest["target"] != target():
        raise ValueError(
            f"Environment bundle targets {manifest['target']}, current interpreter is {target()}"
        )
    if manifest["requirements_sha256"] != digest_file(requirements):
        raise ValueError("Bundle was prepared with different dependency requirements")
    if manifest["resolved_sha256"] != digest_file(directory / "resolved-requirements.txt"):
        raise ValueError("Resolved dependency list checksum mismatch")
    if {p.name for p in directory.glob("*.whl")} != set(manifest["wheels"]):
        raise ValueError("Missing or unexpected wheels in bundle")
    for name, info in manifest["wheels"].items():
        if Path(name).name != name:
            raise ValueError("Invalid wheel path")
        p = directory / name
        if p.stat().st_size != info["bytes"] or digest_file(p) != info["sha256"]:
            raise ValueError(f"Corrupted wheel: {name}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "verify"])
    parser.add_argument("--directory", required=True)
    parser.add_argument("--dependency-index", default=DEPENDENCY_INDEX)
    args = parser.parse_args(argv)
    directory = Path(args.directory).resolve()
    if args.command == "verify":
        result = verify(directory)
        print(f"Verified {len(result['wheels'])} wheels for {result['target']}")
        return
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "Prepare this bundle on Linux x86_64 with the SAME Python major/minor as AutoDL (currently Python3.10); no GPU required"
        )
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "MANIFEST.json").exists():
        verify(directory)
        print("Complete matching bundle already exists; no download required.")
        return
    for command in download_commands(directory, dependency_index=args.dependency_index):
        subprocess.run(command, check=True)
    result = seal(directory)
    print(
        f"Bundle ready: {directory} ({len(result['wheels'])} wheels). Copy the whole directory to AutoDL."
    )


if __name__ == "__main__":
    main()
