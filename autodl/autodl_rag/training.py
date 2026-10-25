from __future__ import annotations

import importlib.metadata
import math
import random
import shutil
import time
from pathlib import Path

from .core import fingerprint, read_json, records, reward, save_json, write_line
from .data import prepared
from .models import LocalModel, model_identity, seed_everything
from .pipeline import Pipeline, source_identity


def advantages(rewards, floor):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / max(std, floor) for r in rewards]


def package_versions():
    return {
        p: importlib.metadata.version(p)
        for p in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
    }


def train(
    cfg,
    run_dir,
    groups=64,
    save_every=16,
    resume=None,
    warm_start=None,
    probe_backward=False,
):
    import torch

    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "Single-process trainer: do NOT launch with torchrun or accelerate launch"
        )
    if groups < 1 or save_every < 1:
        raise ValueError("groups/save_every must be positive")
    run_dir = Path(run_dir).resolve()
    folder = prepared(cfg)
    pool = list(records(folder / "train_pool.jsonl"))
    if groups > len(pool):
        raise ValueError(
            f"Requested {groups} groups but train_pool has {len(pool)} unique questions"
        )
    signature = {
        "config": cfg,
        "groups": groups,
        "warm_start": str(Path(warm_start).resolve()) if warm_start else None,
        "warm_start_files": {
            p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
            for p in Path(warm_start).glob("adapter*")
            if p.is_file()
        }
        if warm_start
        else None,
        "prepared": read_json(folder / "manifest.json"),
        "source": source_identity(),
        "qwen": model_identity(cfg["qwen_path"]),
        "critic": model_identity(cfg["llama_path"]),
        "reranker": model_identity(cfg["reranker_path"])
        if cfg.get("reranker_path")
        else None,
    }
    signature_hash = fingerprint(signature)
    manifest_path = run_dir / "manifest.json"
    if resume:
        resume = Path(resume).resolve()
        if (
            not manifest_path.exists()
            or read_json(manifest_path)["signature_hash"] != signature_hash
        ):
            raise ValueError(
                "Resume config/data/source/models/total groups differ from original run"
            )
        if (
            resume.parent != run_dir / "checkpoints"
            or not (resume / "trainer.pt").exists()
        ):
            raise ValueError(
                "--resume must be a complete checkpoint under this run-dir/checkpoints"
            )
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir}: choose a new run-dir or explicitly --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    qwen = LocalModel(cfg, "qwen")
    critic = LocalModel(cfg, "critic", trainable=True, warm_start=warm_start)
    if any(p.requires_grad for p in qwen.model.parameters()):
        raise RuntimeError("Qwen must remain frozen")
    params = [p for p in critic.model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable actor LoRA parameters")
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0)
    pipeline = Pipeline(cfg, mode="rl", qwen=qwen, critic=critic)
    start, updates, fixes, breaks, skipped = 0, 0, 0, 0, 0
    seed_everything(cfg["seed"])
    if resume:
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        set_peft_model_state_dict(
            critic.model,
            load_peft_weights(str(resume), device="cpu"),
            adapter_name="default",
        )
        state = torch.load(
            resume / "trainer.pt", map_location="cpu", weights_only=False
        )
        if state["signature_hash"] != signature_hash:
            raise ValueError("Checkpoint signature mismatch")
        optimizer.load_state_dict(state["optimizer"])
        start, updates, fixes, breaks, skipped = (
            state[k]
            for k in ("completed_groups", "updates", "fixes", "breaks", "skipped")
        )
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        # Archive uncheckpointed work, then replay from saved RNG/optimizer state.
        # Refuse old checkpoints when newer complete checkpoints exist.
        for complete in (run_dir / "checkpoints").glob("group-*/complete.json"):
            if complete.parent.name.endswith(".building"):
                continue
            if read_json(complete)["group"] > start:
                raise ValueError(
                    "A newer complete checkpoint exists; resume the latest one"
                )
        for name in ("progress.jsonl", "trajectories.jsonl"):
            log = run_dir / name
            if log.exists():
                from .experiment import repair_incomplete_tail

                repair_incomplete_tail(log)
                old = list(records(log))
                if any(r["group"] > start for r in old):
                    backup = run_dir / (name + f".before-resume-{time.time_ns()}")
                    shutil.copyfile(log, backup)
                    with log.open("w", encoding="utf-8") as f:
                        for r in old:
                            if r["group"] <= start:
                                write_line(f, r)
    packages = package_versions()
    if resume and read_json(manifest_path)["packages"] != packages:
        raise ValueError(
            "Resume package versions differ from the recorded training environment"
        )
    if not resume:
        save_json(
            manifest_path,
            {
                "signature_hash": signature_hash,
                "signature": signature,
                "packages": packages,
                "objective": "full up-to-two-turn critic trajectories; on-policy group-relative clipped objective with KL",
                "reference": "frozen warm-start adapter"
                if warm_start
                else "frozen original LLaMA (LoRA disabled)",
                "labels": "TRAIN only in reward; never in model payloads",
                "trained": False,
            },
        )
    initial_weights = {
        n: p.detach().float().cpu().clone()
        for n, p in critic.model.named_parameters()
        if p.requires_grad
    }

    def save_checkpoint(completed):
        dest = run_dir / "checkpoints" / f"group-{completed:06d}"
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite {dest}")
        staging = dest.with_name(dest.name + ".building")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        critic.save_adapter(staging)
        state = {
            "signature_hash": signature_hash,
            "completed_groups": completed,
            "updates": updates,
            "fixes": fixes,
            "breaks": breaks,
            "skipped": skipped,
            "optimizer": optimizer.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        torch.save(state, staging / "trainer.pt")
        save_json(
            staging / "complete.json",
            {"group": completed, "updates": updates, "signature_hash": signature_hash},
        )
        staging.rename(dest)
        save_json(
            run_dir / "latest.json",
            {"checkpoint": str(dest), "group": completed, "updates": updates},
        )

    probed = False
    with (
        (run_dir / "progress.jsonl").open("a", encoding="utf-8") as progress,
        (run_dir / "trajectories.jsonl").open("a", encoding="utf-8") as trajectories,
    ):
        for g in range(start, groups):
            row = pool[g]
            initial, _ = pipeline.initial(row)
            traces = []
            for branch in range(cfg["group_size"]):
                trace = pipeline.run(row, sample=True, initial=initial)
                parts = reward(trace, row["answers"], cfg)
                fixes += int(parts["initial_em"] == 0 and parts["final_em"] == 1)
                breaks += int(parts["initial_em"] == 1 and parts["final_em"] == 0)
                traces.append((trace, parts))
                # The reward log includes train labels, not the model requests.
                write_line(
                    trajectories,
                    {
                        "group": g + 1,
                        "branch": branch,
                        "train_id": row["id"],
                        "reward": parts,
                        "trace": {k: v for k, v in trace.items() if k != "_samples"},
                    },
                )
            values = [p["total"] for _, p in traces]
            adv = advantages(values, cfg["advantage_std_floor"])
            optimizer.zero_grad(set_to_none=True)
            if probe_backward and not probed:
                s = traces[0][0]["_samples"][0]
                loss = -critic.logprobs(s, grad=True).mean()
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError(
                        "Backward probe produced invalid/zero LoRA gradients"
                    )
                optimizer.zero_grad(set_to_none=True)
                save_json(
                    run_dir / "backward_probe.json",
                    {
                        "passed": True,
                        "gradient_norm": norm,
                        "diagnostic_only": True,
                        "optimizer_step_performed": False,
                    },
                )
                probed = True
            loss_value = 0.0
            if max(values) - min(values) > 1e-6:
                old_and_ref = []
                for trace, _ in traces:
                    pairs = []
                    for s in trace["_samples"]:
                        old = critic.logprobs(s).detach().cpu()
                        with critic.reference():
                            ref = critic.logprobs(s).detach().cpu()
                        pairs.append((old, ref))
                    old_and_ref.append(pairs)
                # Sequential trajectory/decision backward avoids storing four model graphs.
                for i, (trace, _) in enumerate(traces):
                    for j, s in enumerate(trace["_samples"]):
                        lp = critic.logprobs(s, grad=True)
                        old, ref = (p.to(lp.device) for p in old_and_ref[i][j])
                        ratio = (lp - old).exp()
                        delta = ref - lp
                        kl = delta.exp() - delta - 1
                        unclipped = ratio * adv[i]
                        clipped = (
                            ratio.clamp(
                                1 - cfg["clip_epsilon"], 1 + cfg["clip_epsilon"]
                            )
                            * adv[i]
                        )
                        objective = (
                            -torch.minimum(unclipped, clipped) + cfg["kl_beta"] * kl
                        ).mean()
                        loss = objective / (cfg["group_size"] * len(trace["_samples"]))
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                "Nonfinite GRPO loss; stop rather than saving corrupt weights"
                            )
                        loss.backward()
                        loss_value += float(loss.detach())
                norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                if not math.isfinite(norm):
                    raise FloatingPointError("Nonfinite actor gradients")
                scale = (
                    min((g + 1) / cfg["warmup_groups"], 1.0)
                    if cfg["warmup_groups"]
                    else 1.0
                )
                scale *= 1 - 0.8 * g / max(1, groups - 1)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cfg["learning_rate"] * scale
                optimizer.step()
                updates += 1
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            item = {
                "group": g + 1,
                "updates": updates,
                "rewards": values,
                "advantages": adv,
                "loss": loss_value,
                "skipped_equal_rewards": max(values) - min(values) <= 1e-6,
                "train_fixes": fixes,
                "train_breaks": breaks,
            }
            write_line(progress, item)
            print(
                f"Group {g + 1}/{groups} updates={updates} rewards={values} train fixes/breaks={fixes}/{breaks}",
                flush=True,
            )
            if (g + 1) % save_every == 0 or g + 1 == groups:
                save_checkpoint(g + 1)
    drift = math.sqrt(
        sum(
            float(((p.detach().float().cpu() - initial_weights[n]) ** 2).sum())
            for n, p in critic.model.named_parameters()
            if n in initial_weights
        )
    )
    result = {
        "complete": True,
        "groups": groups,
        "optimizer_updates": updates,
        "skipped_equal_reward_groups": skipped,
        "training_sampled_trajectory_fixes": fixes,
        "training_sampled_trajectory_breaks": breaks,
        "lora_l2_change_since_this_invocation": drift,
        "trained": updates > 0,
        "note": "Training counters are not validation metrics. If no updates occur, inspect raw critiques/rewards; do not claim RL improvement.",
    }
    save_json(run_dir / "summary.json", result)
    manifest = read_json(manifest_path)
    manifest.update(trained=updates > 0, complete=True, optimizer_updates=updates)
    save_json(manifest_path, manifest)
    return result
