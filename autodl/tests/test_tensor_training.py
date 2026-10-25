"""Real tiny LLaMA/PEFT tensor tests on CPU; no pretrained downloads or benchmark claims."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autodl_rag.core import read_json, save_json
from autodl_rag.models import LocalModel

HAS_TORCH = all(importlib.util.find_spec(p) for p in ("torch", "transformers", "peft"))


def tiny_critic(cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    local = LocalModel.__new__(LocalModel)
    local.model, local.device, local.dtype = model, "cpu", torch.float32
    local.cfg, local.reference_adapter = cfg, False

    class Tokenizer:
        def save_pretrained(self, path):
            save_json(Path(path) / "synthetic-tokenizer.json", {"fixture_only": True})

    local.tokenizer = Tokenizer()
    local._actor()
    return local


@unittest.skipUnless(
    HAS_TORCH, "Optional tensor tests need torch, transformers and peft"
)
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.cfg = {"temperature": 0.8}

    def test_tiny_qwen_generation_and_llama31_rope_forward(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

        common = dict(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        qwen = Qwen2ForCausalLM(Qwen2Config(**common)).eval()
        ids = torch.tensor([[1, 4, 5]])
        with torch.no_grad():
            output = qwen.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=2,
                min_new_tokens=2,
            )
        self.assertEqual(output.shape, (1, 5))
        torch.testing.assert_close(output[:, :3], ids)
        llama = LlamaForCausalLM(
            LlamaConfig(
                **common,
                rope_scaling={
                    "rope_type": "llama3",
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 128,
                },
            )
        ).eval()
        with torch.no_grad():
            logits = llama(input_ids=ids, use_cache=False).logits
        self.assertEqual(logits.shape, (1, 3, 32))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_completion_logprobs_match_full_causal_forward(self):
        import torch

        critic = tiny_critic(self.cfg)
        sample = {"prompt_ids": [1, 4, 5], "completion_ids": [6, 7, 2]}
        actual = critic.logprobs(sample)
        critic.model.eval()
        ids = torch.tensor([[1, 4, 5, 6, 7, 2]])
        with torch.no_grad():
            logits = critic.model(input_ids=ids, use_cache=False).logits[0, 2:-1] / 0.8
            wanted = (
                logits.log_softmax(-1)
                .gather(-1, torch.tensor([[6], [7], [2]]))
                .flatten()
            )
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-5)

    def test_real_lora_backward_updates_actor_only_and_reference_stays_fixed(self):
        import torch

        critic = tiny_critic(self.cfg)
        sample = {"prompt_ids": [1, 4, 5], "completion_ids": [6, 7, 2]}
        frozen = {
            n: p.detach().clone()
            for n, p in critic.model.named_parameters()
            if not p.requires_grad
        }
        with critic.reference():
            before = critic.logprobs(sample).clone()
        parameters = [p for p in critic.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        old = critic.logprobs(sample).detach()
        lp = critic.logprobs(sample, grad=True)
        loss = -(lp - old).exp().mean()
        loss.backward()
        self.assertTrue(
            any(
                p.grad is not None and float(p.grad.abs().sum()) > 0 for p in parameters
            )
        )
        self.assertTrue(
            all(
                p.grad is None
                for n, p in critic.model.named_parameters()
                if n in frozen
            )
        )
        optimizer.step()
        after = critic.logprobs(sample)
        self.assertFalse(torch.allclose(old, after))
        with critic.reference():
            reference_after = critic.logprobs(sample)
        torch.testing.assert_close(before, reference_after, rtol=0, atol=0)
        for n, p in critic.model.named_parameters():
            if n in frozen:
                torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)

    def test_warm_start_reference_adapter_stays_frozen_when_actor_changes(self):
        import torch
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

        sample = {"prompt_ids": [1, 4, 5], "completion_ids": [6, 7, 2]}
        source = tiny_critic(self.cfg)
        optim = torch.optim.AdamW(
            [p for p in source.model.parameters() if p.requires_grad], lr=0.01
        )
        (-source.logprobs(sample, grad=True).mean()).backward()
        optim.step()
        with tempfile.TemporaryDirectory() as tmp:
            source.save_adapter(tmp)
            torch.manual_seed(42)
            critic = tiny_critic(self.cfg)
            set_peft_model_state_dict(
                critic.model, load_peft_weights(tmp, device="cpu")
            )
            critic.model.load_adapter(tmp, adapter_name="reference", is_trainable=False)
            critic.reference_adapter = True
            critic._actor()
            with critic.reference():
                before = critic.logprobs(sample).clone()
            optim = torch.optim.AdamW(
                [p for p in critic.model.parameters() if p.requires_grad], lr=0.01
            )
            (-critic.logprobs(sample, grad=True).mean()).backward()
            optim.step()
            with critic.reference():
                after = critic.logprobs(sample)
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertFalse(torch.allclose(critic.logprobs(sample), before))
            self.assertTrue(
                all(
                    not p.requires_grad
                    for n, p in critic.model.named_parameters()
                    if ".reference." in n
                )
            )

    def test_full_trainer_checkpoint_resume_matches_uninterrupted_cpu_run(self):
        import torch
        from autodl_rag.data import prepare
        from autodl_rag.training import train
        from safetensors.torch import load_file
        from test_system import configuration

        class Qwen:
            def __init__(self):
                self.model = torch.nn.Linear(2, 2).requires_grad_(False)

        class Environment:
            """Deterministic synthetic rewards isolate actual optimizer/resume machinery."""

            def __init__(self, cfg, **kwargs):
                self.branch = 0

            def initial(self, row):
                return {}, True

            def run(self, row, **kwargs):
                branch = self.branch % 4
                self.branch += 1
                answer = row["answers"][0] if branch == 0 else "synthetic wrong answer"
                turn = {"status": "accepted" if branch == 0 else "critic_stop"}
                if branch == 2:
                    turn["status"] = "invalid_critique"
                return {
                    "initial": {"answer": "synthetic wrong answer"},
                    "final": {"answer": answer},
                    "revisions": int(branch in (0, 3)),
                    "turns": [turn],
                    "_samples": [
                        {"prompt_ids": [1, 4, 5], "completion_ids": [6 + branch, 2]}
                    ],
                }

        def local(cfg, role, **kwargs):
            return Qwen() if role == "qwen" else tiny_critic(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = configuration(root)
            cfg["learning_rate"] = 0.001
            prepare(cfg)
            run = root / "train"
            # All tensors/LoRA/checkpoint files are real. Only the environment is a fixture.
            with (
                patch("autodl_rag.training.LocalModel", side_effect=local),
                patch("autodl_rag.training.Pipeline", Environment),
                patch(
                    "autodl_rag.training.package_versions",
                    return_value={"torch": "cpu-fixture"},
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = train(cfg, run, groups=2, save_every=1, probe_backward=True)
                self.assertEqual(result["optimizer_updates"], 2)
                self.assertGreater(result["lora_l2_change_since_this_invocation"], 0)
                self.assertTrue(read_json(run / "backward_probe.json")["passed"])
                completed = run / "checkpoints" / "group-000002"
                wanted = load_file(completed / "adapter_model.safetensors")
                shutil.rmtree(completed)
                (run / "summary.json").unlink()
                # Keep later logs to verify that resume archives/rewinds uncheckpointed work.
                result = train(
                    cfg,
                    run,
                    groups=2,
                    save_every=1,
                    resume=run / "checkpoints" / "group-000001",
                )
                got = load_file(completed / "adapter_model.safetensors")
                self.assertEqual(result["optimizer_updates"], 2)
                self.assertTrue(list(run.glob("progress.jsonl.before-resume-*")))
                for name in wanted:
                    torch.testing.assert_close(wanted[name], got[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
