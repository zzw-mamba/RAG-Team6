"""Image reuse must never resolve/download torch or discard an existing environment."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autodl_rag.bootstrap import main as bootstrap
from autodl_rag.image_env import dependency_commands, main, prepare_venv
from autodl_rag.models import completion_logits_options


class ImageEnvironmentTests(unittest.TestCase):
    def test_reuse_mode_accepts_222_and_never_invokes_installer(self):
        working = {
            "status": "present",
            "version": "2.2.2+cu118",
            "cuda_runtime": "11.8",
            "cuda_available": True,
        }
        for state, ok in (
            (working, True),
            ({"status": "missing"}, False),
            ({**working, "version": "2.2.1+cu118"}, False),
            ({**working, "cuda_available": False}, False),
        ):
            with (
                patch("autodl_rag.bootstrap.inspect_torch", return_value=state),
                patch("autodl_rag.bootstrap.subprocess.run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if ok:
                    bootstrap(["--reuse-image", "--verify-only"])
                else:
                    with self.assertRaises(RuntimeError):
                        bootstrap(["--reuse-image"])
                run.assert_not_called()

    def test_dependency_install_cannot_resolve_torch_or_cuda(self):
        commands = dependency_commands(
            "/project/.venv/bin/python", "/req.txt", "https://mirror.test"
        )
        for command in commands:
            self.assertIn("--no-deps", command)
            self.assertEqual(command[:3], ["/project/.venv/bin/python", "-m", "pip"])
        self.assertIn("--no-index", commands[-1])
        requirements = Path(__file__).parents[1] / "environment/requirements-qwen-image.txt"
        names = [
            line.split("==")[0].lower()
            for line in requirements.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        self.assertFalse(any(name == "torch" or name.startswith("nvidia-") for name in names))
        self.assertIn("numpy", names)

    def test_missing_image_torch_fails_before_creating_or_installing(self):
        with (
            patch("autodl_rag.image_env.inspect_torch", return_value={"status": "missing"}),
            patch("autodl_rag.image_env.prepare_venv") as prepare,
            patch("autodl_rag.image_env.subprocess.run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "no torch download"):
                main(["--index-url", "https://mirror.test"])
            prepare.assert_not_called()
            run.assert_not_called()

    def test_unrelated_venv_is_preserved_and_matching_venv_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project / ".venv").mkdir()
            (project / ".venv/user-file").write_text("keep previous environment")

            def create(command, check):
                python = Path(command[-1]) / "bin/python"
                python.parent.mkdir(parents=True)
                python.touch()

            identity = {"image_prefix": "/image/env", "torch_version": "2.2.2+cu118"}
            with (
                patch("autodl_rag.image_env.subprocess.run", side_effect=create) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                python = prepare_venv(project, identity)
                self.assertTrue(python.is_file())
                self.assertIn("--system-site-packages", run.call_args.args[0])
                run.reset_mock()
                prepare_venv(project, identity)
                run.assert_not_called()
            backups = list(project.glob(".venv.backup-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "user-file").read_text(), "keep previous environment")

    def test_keyword_compatibility_handles_both_transformers_interfaces(self):
        class Old:
            def forward(self, input_ids, num_logits_to_keep=0):
                pass

        class New:
            def forward(self, input_ids, logits_to_keep=0):
                pass

        class Wrapper:
            def get_base_model(self):
                return Old()

        self.assertEqual(completion_logits_options(Old(), 8), {"num_logits_to_keep": 8})
        self.assertEqual(completion_logits_options(New(), 8), {"logits_to_keep": 8})
        self.assertEqual(completion_logits_options(Wrapper(), 8), {"num_logits_to_keep": 8})

    def test_image_entry_forwards_flags_and_chooses_image_config(self):
        with tempfile.TemporaryDirectory(prefix="qwen image ") as temp:
            project = Path(temp)
            shutil.copyfile(
                Path(__file__).parents[1] / "run_qwen_image.sh", project / "run_qwen_image.sh"
            )
            (project / "run_all.sh").write_text(
                'printf "%s\\n" "$RAG_ENV_PROFILE" "$CONFIG" "$@"\nexit 17\n'
            )
            env = {k: v for k, v in os.environ.items() if k not in ("CONFIG", "RAG_ENV_PROFILE")}
            result = subprocess.run(
                ["bash", str(project / "run_qwen_image.sh"), "--from", "train"],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 17)
            self.assertEqual(
                result.stdout.splitlines(),
                ["qwen-image", str(project / "configs/autodl-qwen-image.json"), "--from", "train"],
            )
            cfg = json.loads(
                (Path(__file__).parents[1] / "configs/autodl-qwen-image.json").read_text()
            )
            self.assertEqual(cfg["qwen_path"], "/root/Qwen2.5/text-generation-webui/models")


if __name__ == "__main__":
    unittest.main()
