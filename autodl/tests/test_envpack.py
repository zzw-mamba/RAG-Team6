from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autodl_rag.envpack import download_commands, seal, verify


class EnvironmentPackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.requirements = self.root / "requirements.txt"
        self.requirements.write_text("torch==2.8.0+cu128\n")
        self.wheel = self.root / "torch-2.8.0+cu128-cp310-cp310-linux_x86_64.whl"
        self.wheel.write_bytes(b"synthetic checksum fixture, never installed")

    def tearDown(self):
        self.temp.cleanup()

    def test_torch_and_dependencies_use_separate_indexes_and_resume_options(self):
        torch, deps = download_commands(self.root, self.requirements, "https://example.test/simple")
        self.assertIn("--no-deps", torch)
        self.assertIn("https://download.pytorch.org/whl/cu128", torch)
        self.assertIn("torch==2.8.0+cu128", torch)
        self.assertNotIn("--no-deps", deps)
        self.assertNotIn("https://download.pytorch.org/whl/cu128", deps)
        self.assertNotIn("--extra-index-url", deps)
        self.assertIn("https://example.test/simple", deps)
        self.assertIn("--find-links", deps)
        for command in (torch, deps):
            self.assertEqual(command[command.index("--resume-retries") + 1], "30")

    def test_checksums_accept_complete_bundle_and_reject_corruption(self):
        seal(self.root, self.requirements)
        self.assertIn(self.wheel.name, verify(self.root, self.requirements)["wheels"])
        self.wheel.write_bytes(b"incomplete download")
        with self.assertRaisesRegex(ValueError, "Corrupted wheel"):
            verify(self.root, self.requirements)

    def test_python_or_platform_mismatch_is_rejected(self):
        seal(self.root, self.requirements)
        with patch(
            "autodl_rag.envpack.target",
            return_value={"system": "Linux", "machine": "x86_64", "python": [3, 99]},
        ):
            with self.assertRaisesRegex(ValueError, "bundle targets"):
                verify(self.root, self.requirements)

    def test_missing_extra_or_changed_lock_files_are_rejected(self):
        seal(self.root, self.requirements)
        extra = self.root / "extra-1.0-py3-none-any.whl"
        extra.write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "unexpected wheels"):
            verify(self.root, self.requirements)
        extra.unlink()
        lock = self.root / "resolved-requirements.txt"
        original = lock.read_text()
        lock.write_text("torch==0.0\n")
        with self.assertRaisesRegex(ValueError, "dependency list checksum"):
            verify(self.root, self.requirements)
        lock.write_text(original)
        self.wheel.unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify(self.root, self.requirements)

    def test_ambiguous_versions_cannot_be_sealed(self):
        (self.root / "torch-2.7.0-cp310-cp310-linux_x86_64.whl").write_bytes(
            b"another synthetic version"
        )
        with self.assertRaisesRegex(ValueError, "Multiple versions"):
            seal(self.root, self.requirements)


if __name__ == "__main__":
    unittest.main()
