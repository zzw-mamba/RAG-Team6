"""Launch a pinned loopback vLLM service and publish a ready deployment manifest."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from autodl_rag.core import fingerprint, load_config, read_json, save_json
from autodl_rag.models import model_identity
from scripts.vllm_client import request_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8106)
    parser.add_argument("--memory-utilization", type=float, default=.65)
    parser.add_argument("--max-seqs", type=int, default=32)
    parser.add_argument("--ready-file")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 < args.memory_utilization < 1 or args.max_seqs < 1:
        parser.error("Invalid serving resource limits")
    version = importlib.metadata.version("vllm")
    if version != "0.11.2":
        raise RuntimeError("This adapter pins vLLM 0.11.2; install its separate environment")
    cfg = load_config(args.config)
    storage = Path(cfg["work_dir"]).parent / "vllm"
    storage.mkdir(parents=True, exist_ok=True)
    identity = model_identity(cfg["qwen_path"])
    model_config = read_json(Path(identity["path"]) / "config.json")
    if model_config.get("model_type") != "qwen2" or model_config.get("quantization_config"):
        raise ValueError("Expected original unquantized Qwen2.5 model files")
    generation = Path(identity["path"]) / "generation_config.json"
    eos = read_json(generation).get("eos_token_id", []) if generation.exists() else []
    if isinstance(eos, int):
        eos = [eos]
    command = [
        str(Path(sys.executable).with_name("vllm")), "serve", identity["path"],
        "--host", "127.0.0.1", "--port", str(args.port),
        "--served-model-name", "team6-qwen", "--dtype", "bfloat16",
        "--tensor-parallel-size", "1", "--max-model-len", "18000",
        "--max-num-seqs", str(args.max_seqs), "--max-num-batched-tokens", "32768",
        "--gpu-memory-utilization", str(args.memory_utilization),
        "--enable-prefix-caching", "--structured-outputs-config.backend", "xgrammar",
        "--generation-config", "vllm", "--seed", str(cfg["seed"]), "--disable-log-requests",
    ]
    if os.environ.get("VLLM_ENFORCE_EAGER") == "1":
        command.append("--enforce-eager")
    endpoint = f"http://127.0.0.1:{args.port}"
    manifest = {
        "status": "ready", "endpoint": endpoint, "served_model_name": "team6-qwen",
        "model_identity": identity, "dtype": "bfloat16", "quantization": None,
        "max_model_len": 18000, "eos_token_ids": eos,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "engine_environment": {name: os.environ.get(name, "")
                               for name in ("VLLM_USE_V1", "VLLM_ATTENTION_BACKEND")},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("vllm", "torch", "xgrammar", "transformers", "tokenizers")},
        "gpu_inventory": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines(),
        "command": command,
    }
    target = storage / "server.json"

    def acknowledge():
        if args.ready_file:
            save_json(args.ready_file, {"status": "ready", "deployment": fingerprint(manifest)})

    with (storage / "server.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if target.exists() and read_json(target) == manifest:
                request_json(endpoint, "/version", timeout=5)
                acknowledge()
                print("Matching managed vLLM service is already running", flush=True)
                return
            raise RuntimeError("A different managed vLLM deployment is running") from None
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError("Port is already used by another service; choose VLLM_PORT")
        save_json(target, {"status": "starting"})
        child = subprocess.Popen(command, start_new_session=True)
        save_json(storage / "server.pid.json", {
            "launcher": os.getpid(), "server": child.pid, "config": str(Path(args.config).resolve()),
        })

        def stop(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            deadline = time.monotonic() + 1200
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f"vLLM startup failed, exit code {child.returncode}; inspect server log")
                try:
                    models = request_json(endpoint, "/v1/models", timeout=3)["data"]
                    if any(m["id"] == "team6-qwen" and m.get("root") == identity["path"] for m in models):
                        break
                except (OSError, RuntimeError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("vLLM server did not become ready")
                time.sleep(2)
            save_json(target, manifest)
            acknowledge()
            print(f"Ready: {endpoint}; one BF16 Qwen, max sequences={args.max_seqs}", flush=True)
            code = child.wait()
            if code:
                raise RuntimeError(f"vLLM exited with {code}")
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            save_json(target, {"status": "stopped"})


if __name__ == "__main__":
    main()
