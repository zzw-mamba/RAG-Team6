from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from autodl_rag.bootstrap import inspect_torch, install_needed, main, pip_command


def cuda_state(version="2.8.0+cu128", runtime="12.8", available=True):
    return {
        "status": "present",
        "version": version,
        "cuda_runtime": runtime,
        "cuda_available": available,
    }


class BootstrapTests(unittest.TestCase):
    def test_missing_torch_is_installable_without_import_failure(self):
        error = ModuleNotFoundError("No module named 'torch'", name="torch")
        with patch("autodl_rag.bootstrap.importlib.import_module", side_effect=error):
            state = inspect_torch()
        self.assertEqual(state["status"], "missing")
        self.assertTrue(install_needed(state))

    def test_working_cuda_is_reused_without_pip(self):
        with (
            patch("autodl_rag.bootstrap.inspect_torch", return_value=cuda_state()),
            patch("autodl_rag.bootstrap.subprocess.run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main([])
        run.assert_not_called()

    def test_cpu_and_old_torch_require_cuda_installation(self):
        self.assertTrue(install_needed(cuda_state(runtime=None, available=False)))
        self.assertTrue(install_needed(cuda_state(version="2.5.1+cu124")))

    def test_unavailable_gpu_or_broken_import_fails_explicitly(self):
        for state in (
            cuda_state(available=False),
            {"status": "broken", "error": "missing shared library"},
        ):
            with self.assertRaises(RuntimeError):
                install_needed(state)

    def test_install_uses_project_interpreter_and_matching_optional_packages(self):
        command = pip_command(
            "/project/.venv/bin/python",
            "https://download.pytorch.org/whl/cu128",
            ["torchvision", "torchaudio"],
        )
        self.assertEqual(command[:4], ["/project/.venv/bin/python", "-m", "pip", "install"])
        self.assertIn("torch==2.8.0", command)
        self.assertIn("torchvision==0.23.0", command)
        self.assertIn("torchaudio==2.8.0", command)
        self.assertEqual(command[-1], "https://download.pytorch.org/whl/cu128")

    def test_missing_torch_install_is_verified_in_fresh_process(self):
        with (
            patch("autodl_rag.bootstrap.inspect_torch", return_value={"status": "missing"}),
            patch("autodl_rag.bootstrap.platform.system", return_value="Linux"),
            patch("autodl_rag.bootstrap.importlib.util.find_spec", return_value=None),
            patch("autodl_rag.bootstrap.subprocess.run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main([])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][0], sys.executable)
        self.assertEqual(
            run.call_args_list[1].args[0],
            [sys.executable, "-m", "autodl_rag.bootstrap", "--verify-only"],
        )


if __name__ == "__main__":
    unittest.main()
