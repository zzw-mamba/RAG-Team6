"""Exercise the real Bash orchestrator with synthetic stage scripts, without GPUs."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rag launcher ")
        self.root = Path(self.temp.name)
        self.project = self.root / "project with spaces"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "configs").mkdir()
        (self.project / "configs/autodl.json").write_text("{}")
        source = Path(__file__).parent.parent / "run_all.sh"
        shutil.copyfile(source, self.project / "run_all.sh")
        self.trace = self.root / "trace.txt"
        self.logs = self.root / "logs with spaces"
        self.env = {**os.environ, "TRACE": str(self.trace), "LOG_DIR": str(self.logs)}
        for name in (
            "CONFIG",
            "RUN_NAME",
            "PILOT_GROUPS",
            "FAIL_STAGE",
            "RAG_ENV_PROFILE",
            "IMAGE_PYTHON",
        ):
            self.env.pop(name, None)
        # Removing IMAGE_PYTHON would expose the real host's /root default.
        # Ordinary stage fixtures have only the modern config. Tests that need
        # image detection explicitly replace this absent, fixture-local path.
        self.env["IMAGE_PYTHON"] = str(self.root / "absent image python")
        for stage, script in (
            ("install", "install.sh"),
            ("prepare", "prepare.sh"),
            ("smoke", "smoke.sh"),
            ("train", "train_pilot.sh"),
            ("evaluate", "evaluate.sh"),
            ("predict", "predict.sh"),
        ):
            (self.project / "scripts" / script).write_text(
                "#!/usr/bin/env bash\nset -eu\n"
                f'printf "%s\\n" "{stage}" >> "$TRACE"\n'
                'printf "config=%s\\nrun=%s\\n" "$CONFIG" "$RUN_NAME"\n'
                f'if [[ "${{FAIL_STAGE:-}}" == "{stage}" ]]; then echo "stage failure" >&2; exit 17; fi\n'
            )

    def tearDown(self):
        self.temp.cleanup()

    def run_launcher(self, *args):
        return subprocess.run(
            ["bash", str(self.project / "run_all.sh"), *args],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_all_six_stages_run_in_order_and_keep_logs(self):
        self.env["PILOT_GROUPS"] = "16"
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.trace.read_text().splitlines(),
            ["install", "prepare", "smoke", "train", "evaluate", "predict"],
        )
        self.assertEqual(len(list(self.logs.glob("*.log"))), 6)
        self.assertIn(
            str(self.project / "configs/autodl.json"), (self.logs / "1-install.log").read_text()
        )
        self.assertIn("run=pilot16", result.stdout)

    def test_failure_keeps_exit_code_and_never_starts_later_stages(self):
        self.env["FAIL_STAGE"] = "train"
        result = self.run_launcher()
        self.assertEqual(result.returncode, 17)
        self.assertEqual(
            self.trace.read_text().splitlines(), ["install", "prepare", "smoke", "train"]
        )
        self.assertIn("stage failure", (self.logs / "4-train.log").read_text())
        self.assertIn("--from train", result.stderr)
        self.assertFalse((self.logs / "5-evaluate.log").exists())

    def test_resume_and_skip_predict_only_run_evaluation(self):
        result = self.run_launcher("--from", "evaluate", "--skip-predict")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.trace.read_text().splitlines(), ["evaluate"])

    def test_existing_image_python_selects_compatible_config_automatically(self):
        image_python = self.root / "image python"
        image_python.write_text("#!/bin/sh\nexit 0\n")
        image_python.chmod(0o755)
        (self.project / "configs/autodl-qwen-image.json").write_text("{}")
        self.env["IMAGE_PYTHON"] = str(image_python)
        result = self.run_launcher("--from", "evaluate", "--skip-predict")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("autodl-qwen-image.json", result.stdout)
        self.env["RAG_ENV_PROFILE"] = "modern"
        result = self.run_launcher("--from", "evaluate", "--skip-predict")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("autodl-qwen-image.json", result.stdout)

    def test_stage_tests_ignore_an_existing_host_image_at_default_path(self):
        # Reproduce an AutoDL host on any OS without creating real /root files.
        host_python = self.root / "host image python"
        host_python.write_text("#!/bin/sh\nexit 0\n")
        host_python.chmod(0o755)
        launcher = self.project / "run_all.sh"
        launcher.write_text(
            launcher.read_text().replace(
                "/root/Qwen2.5/text-generation-webui/installer_files/env/bin/python",
                str(host_python),
            )
        )
        result = self.run_launcher("--from", "evaluate", "--skip-predict")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("configs/autodl.json", result.stdout)
        self.assertEqual(self.trace.read_text().splitlines(), ["evaluate"])

    def test_help_and_invalid_arguments_never_start_stages(self):
        self.assertEqual(self.run_launcher("--help").returncode, 0)
        self.assertEqual(self.run_launcher("--from", "wrong").returncode, 2)
        self.assertEqual(self.run_launcher("--from").returncode, 2)
        self.assertFalse(self.trace.exists())


if __name__ == "__main__":
    unittest.main()
