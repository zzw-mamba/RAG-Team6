"""Capture reproducibility metadata, without collecting tokens, cookies, or all env vars."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

app = argparse.ArgumentParser()
app.add_argument("--output", type=Path, required=True)
args = app.parse_args()
result = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "gpu": [],
}
try:
    result["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    result["nvidia_smi"] = None
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
print(args.output)
