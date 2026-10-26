"""Image reuse must never resolve/download torch or discard an existing environment."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from autodl_rag.bootstrap import main as bootstrap
from autodl_rag.image_env import (
    dependency_commands,
    main,
    prepare_venv,
    runtime_report,
    validate_packages,
)
from autodl_rag.models import completion_logits_options, doctor

HAS_PACKAGING = importlib.util.find_spec("packaging") is not None

# Relevant versions from the user's conda list; no paths, credentials or data.
IMAGE_PACKAGES = {
    "numpy": "1.26.4",
    "transformers": "4.40.2",
    "tokenizers": "0.19.1",
    "peft": "0.8.2",
    "accelerate": "0.27.2",
    "bitsandbytes": "0.43.1",
    "safetensors": "0.4.3",
    "sentencepiece": "0.2.0",
    "tqdm": "4.66.4",
    "huggingface-hub": "0.23.3",
    "packaging": "24.0",
    "psutil": "5.9.8",
    "PyYAML": "6.0.1",
    "filelock": "3.13.1",
    "fsspec": "2024.2.0",
    "regex": "2024.5.15",
    "requests": "2.32.3",
    "charset-normalizer": "3.3.2",
    "idna": "3.7",
    "urllib3": "2.2.1",
    "certifi": "2024.6.2",
    "typing-extensions": "4.9.0",
    "setuptools": "69.5.1",
    "wheel": "0.43.0",
    "pydantic": "2.7.3",
    "pydantic-core": "2.18.4",
    "annotated-types": "0.7.0",
}


class ImageEnvironmentTests(unittest.TestCase):
    def test_reuse_mode_accepts_actual_221_image_and_never_invokes_installer(self):
        working = {
            "status": "present",
            "version": "2.2.1+cu118",
            "cuda_runtime": "11.8",
            "cuda_available": True,
        }
        for state, ok in (
            (working, True),
            ({"status": "missing"}, False),
            ({**working, "version": "2.2.2+cu118"}, True),
            ({**working, "version": "2.2.0+cu118"}, False),
            ({**working, "cuda_runtime": None, "cuda_available": False}, False),
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

    def test_image_setup_accepts_221_and_verifies_inherited_torch(self):
        fake_torch = types.SimpleNamespace(
            __file__="/image/env/torch/__init__.py",
            __version__="2.2.1+cu118",
            version=types.SimpleNamespace(cuda="11.8"),
        )
        state = {
            "status": "present",
            "version": fake_torch.__version__,
            "cuda_runtime": "11.8",
            "cuda_available": True,
        }
        identity = {
            "torch_file": fake_torch.__file__,
            "torch_version": fake_torch.__version__,
            "cuda": "11.8",
            "packages": {
                **IMAGE_PACKAGES,
                "transformers": "4.45.2",
                "tokenizers": "0.20.3",
                "ijson": "3.4.0",
                "lm-format-enforcer": "0.11.3",
                "interegular": "0.3.3",
            },
        }
        with (
            patch("autodl_rag.image_env.inspect_torch", return_value=state),
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch(
                "autodl_rag.image_env.prepare_venv", return_value=Path("/project/.venv/bin/python")
            ),
            patch("autodl_rag.image_env.subprocess.run") as run,
            patch("autodl_rag.image_env.validate_packages"),
            patch("autodl_rag.image_env.save_json") as save,
            patch(
                "autodl_rag.image_env.subprocess.check_output", return_value=json.dumps(identity)
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(["--index-url", "https://mirror.test"])
        self.assertEqual(run.call_count, 4)
        self.assertIn("--reuse-image", run.call_args.args[0])
        self.assertEqual(save.call_args.args[0].name, "image-environment.json")

    def test_dependency_install_cannot_resolve_torch_or_cuda(self):
        commands = dependency_commands(
            "/project/.venv/bin/python", "/req.txt", "https://mirror.test"
        )
        for command in commands:
            self.assertIn("--no-deps", command)
            self.assertEqual(command[:3], ["/project/.venv/bin/python", "-m", "pip"])
        self.assertIn("--no-index", commands[-1])
        self.assertNotIn("--upgrade", commands[1])
        requirements = Path(__file__).parents[1] / "environment/requirements-qwen-image.txt"
        names = [
            line.split("==")[0].split(">=")[0].lower()
            for line in requirements.read_text().splitlines()
            if line and not line.startswith("#")
        ]
        self.assertFalse(any(name == "torch" or name.startswith("nvidia-") for name in names))
        self.assertIn("numpy", names)

    @unittest.skipUnless(HAS_PACKAGING, "Package version checks need optional packaging")
    def test_native_image_reuses_existing_libraries_and_adds_required_decoders(self):
        with self.assertRaises(RuntimeError) as failure:
            validate_packages(IMAGE_PACKAGES)
        self.assertIn("transformers", str(failure.exception))
        self.assertIn("tokenizers", str(failure.exception))
        self.assertIn("ijson", str(failure.exception))
        selected = {
            **IMAGE_PACKAGES,
            "transformers": "4.45.2",
            "tokenizers": "0.20.3",
            "ijson": "3.4.0",
            "lm-format-enforcer": "0.11.3",
            "interegular": "0.3.3",
        }
        validate_packages(selected)
        for name in ("peft", "accelerate", "bitsandbytes", "numpy", "huggingface-hub"):
            self.assertEqual(selected[name], IMAGE_PACKAGES[name])

    @unittest.skipUnless(HAS_PACKAGING, "Package version checks need optional packaging")
    def test_previous_project_versions_can_be_kept_without_downgrading(self):
        selected = {
            **IMAGE_PACKAGES,
            "transformers": "4.45.2",
            "tokenizers": "0.20.3",
            "ijson": "3.4.0",
            "lm-format-enforcer": "0.11.3",
            "interegular": "0.3.3",
            "peft": "0.13.2",
            "accelerate": "0.34.2",
            "bitsandbytes": "0.43.3",
            "huggingface-hub": "0.25.2",
            "safetensors": "0.4.5",
            "tqdm": "4.67.1",
        }
        validate_packages(selected)
        selected["numpy"] = "2.0.0"
        with self.assertRaisesRegex(RuntimeError, "numpy"):
            validate_packages(selected)

    @unittest.skipUnless(HAS_PACKAGING, "Package version checks need optional packaging")
    def test_effective_version_report_preserves_native_library_versions(self):
        versions = {
            **IMAGE_PACKAGES,
            "transformers": "4.45.2",
            "tokenizers": "0.20.3",
            "ijson": "3.4.0",
            "lm-format-enforcer": "0.11.3",
            "interegular": "0.3.3",
        }
        fake_torch = types.SimpleNamespace(
            __file__="/image/torch/__init__.py",
            __version__="2.2.1+cu118",
            version=types.SimpleNamespace(cuda="11.8"),
        )
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch(
                "autodl_rag.image_env.importlib.metadata.version", side_effect=versions.__getitem__
            ),
        ):
            report = runtime_report()
        self.assertEqual(report["packages"]["peft"], "0.8.2")
        self.assertEqual(report["packages"]["bitsandbytes"], "0.43.1")
        self.assertEqual(report["torch_version"], "2.2.1+cu118")
        self.assertEqual(report["attention_implementation"], "sdpa")
        self.assertEqual(len(report["requirements_sha256"]), 64)

    @unittest.skipUnless(HAS_PACKAGING, "Version checks need optional packaging")
    def test_doctor_rejects_original_transformers_before_loading_llama31(self):
        fake_torch = types.SimpleNamespace(
            __version__="2.2.1+cu118",
            version=types.SimpleNamespace(cuda="11.8"),
            cuda=types.SimpleNamespace(is_available=lambda: True),
        )
        versions = {**IMAGE_PACKAGES, "ijson": "3.4.0", "lm-format-enforcer": "0.11.3"}
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("importlib.metadata.version", side_effect=versions.__getitem__),
        ):
            with self.assertRaisesRegex(RuntimeError, "Transformers >=4.45.2"):
                doctor({})

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
