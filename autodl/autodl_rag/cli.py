from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import load_config, records, save_json


def main():
    parser = argparse.ArgumentParser(
        description="Team6 AutoDL: Qwen RAG + LLaMA trajectory-GRPO critic"
    )
    parser.add_argument("--config", default="configs/autodl.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor",
        help="Check CUDA, local models and raw data schemas without loading weights",
    )
    sub.add_parser(
        "prepare",
        help="Stream JSON, exclude test overlap and create disjoint grouped splits",
    )
    sub.add_parser("index", help="Build a training-context-only SQLite FTS5 corpus")
    run = sub.add_parser(
        "run", help="Run a baseline or trained critic with full trajectories"
    )
    run.add_argument("--split", choices=["tune", "confirmation", "test"], required=True)
    run.add_argument("--mode", choices=["rag", "critic", "rl"], required=True)
    run.add_argument("--adapter")
    run.add_argument("--output", required=True)
    run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all; use a separate output path for smoke",
    )
    run.add_argument("--resume", action="store_true")
    train = sub.add_parser(
        "train", help="Train only critic LoRA against real Qwen multi-turn outcomes"
    )
    train.add_argument("--run-dir", required=True)
    train.add_argument("--groups", type=int, default=64)
    train.add_argument("--save-every", type=int, default=16)
    train.add_argument(
        "--resume", help="Complete checkpoint directory from this same run"
    )
    train.add_argument(
        "--warm-start",
        help="Adapter for a NEW run; optimizer/RNG restart, frozen reference uses this adapter",
    )
    train.add_argument(
        "--probe-backward",
        action="store_true",
        help="Run a diagnostic backward without an optimizer update",
    )
    ev = sub.add_parser("evaluate")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--split", choices=["tune", "confirmation"], required=True)
    cmp = sub.add_parser("compare")
    for name in ("rag", "critic", "rl", "output"):
        cmp.add_argument("--" + name, required=True)
    cmp.add_argument(
        "--split", choices=["tune", "confirmation"], default="confirmation"
    )
    sel = sub.add_parser(
        "select", help="Rank already evaluated checkpoints using tune only"
    )
    sel.add_argument("--candidates", required=True)
    sel.add_argument("--output", required=True)
    exp = sub.add_parser(
        "export",
        help="Export exactly the full template; no model calls or label access",
    )
    exp.add_argument("--predictions", required=True)
    exp.add_argument("--output", required=True)
    demo = sub.add_parser("demo", help="Generate an offline HTML trajectory viewer")
    demo.add_argument("--predictions", required=True)
    demo.add_argument("--output", required=True)
    demo.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "doctor":
        from .models import doctor

        result = doctor(cfg)
        save_json(Path(cfg["work_dir"]) / "doctor.json", result)
    elif args.command == "prepare":
        from .data import prepare

        result = prepare(cfg)
    elif args.command == "index":
        from .retrieval import build_index

        result = build_index(cfg)
    elif args.command == "train":
        from .training import train

        result = train(
            cfg,
            args.run_dir,
            args.groups,
            args.save_every,
            args.resume,
            args.warm_start,
            args.probe_backward,
        )
    elif args.command == "run":
        from .experiment import run_dataset

        result = run_dataset(
            cfg,
            args.split,
            args.mode,
            args.output,
            args.adapter,
            args.limit,
            args.resume,
        )
    elif args.command == "evaluate":
        from .experiment import evaluate

        result = evaluate(cfg, args.predictions, args.split)
    elif args.command == "compare":
        from .experiment import compare

        result = compare(
            cfg,
            {name: getattr(args, name) for name in ("rag", "critic", "rl")},
            args.split,
            args.output,
        )
    elif args.command == "select":
        from .experiment import select

        result = select(cfg, args.candidates, args.output)
    elif args.command == "export":
        from .experiment import export_csv

        result = export_csv(cfg, args.predictions, args.output)
    elif args.command == "demo":
        import itertools

        data = list(itertools.islice(records(args.predictions), args.limit))
        if not data:
            raise ValueError("No trajectories for demo")
        # Escape script delimiters; all user/model strings render through textContent.
        payload = (
            json.dumps(data, ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )
        html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Team6 RAG过程展示</title>
<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#f6f8fa;color:#182431}select{padding:10px;width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px;border-radius:12px;line-height:1.6}h2{margin-top:30px}.note{color:#526273}</style>
<h1>Team6 RAG过程展示</h1><p class="note">真实运行轨迹回放。停止、拒绝修正和格式失败不代表答案正确。</p><select id="pick"></select><h2>问题</h2><pre id="question"></pre><h2>初始回答</h2><pre id="initial"></pre><details><summary>初答实际可见证据</summary><pre id="initial_documents"></pre></details><h2>批判与修正</h2><pre id="turns"></pre><h2>最终回答</h2><pre id="final"></pre><p id="status"></p>
<script>const rows=DATA;const pick=document.getElementById('pick');rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id+' · '+r.question;pick.append(o)});function show(){const r=rows[+pick.value];for(const k of ['question','initial','initial_documents','turns','final'])document.getElementById(k).textContent=typeof r[k]==='string'?r[k]:JSON.stringify(r[k],null,2);document.getElementById('status').textContent='停止原因：'+r.stop_reason+'；修正尝试：'+r.revisions+'；计入初答的耗时：'+r.logical_seconds.toFixed(2)+'秒'}pick.onchange=show;show();</script></html>""".replace(
            "DATA", payload
        )
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        result = {"html": str(target.resolve()), "rows": len(data)}
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
