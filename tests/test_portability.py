import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import package_project
import run_laya_full as launcher


class PortabilityTests(unittest.TestCase):
    def launch(self, args, gpu_state="0, 2, 0\n1, 2, 0\n", previous=None):
        calls = []

        class FakeProcess:
            pid = 123
            returncode = 0
            def __init__(self, command, **kwargs):
                calls.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            if previous:
                (out / "launcher.json").write_text(json.dumps(previous))
            with patch.object(launcher, "OUT", out), patch.object(sys, "argv", ["run_laya_full.py"] + args), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch.object(launcher.subprocess, "check_output", return_value=gpu_state), \
                    patch.object(launcher.subprocess, "Popen", FakeProcess), contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
            return calls, json.loads((out / "launcher.json").read_text())

    def test_default_selects_only_one_idle_gpu(self):
        calls, meta = self.launch([])
        self.assertEqual(meta["gpus"], [0])
        self.assertEqual([c[1] for c in calls], ["0"])

    def test_explicit_gpu_order_defines_shards(self):
        calls, meta = self.launch(["--gpus", "1,0"])
        self.assertEqual(meta["gpus"], [1, 0])
        self.assertEqual([c[1] for c in calls], ["1", "0"])
        self.assertEqual(calls[1][0][-4:], ["--shard", "1", "--shards", "2"])

    def test_occupied_gpu_is_rejected_without_silent_repartition(self):
        with self.assertRaisesRegex(SystemExit, "occupied"):
            self.launch(["--gpus", "0,1"], "0, 2, 0\n1, 6000, 90\n")

    def test_resume_preserves_shard_mapping(self):
        previous = {"gpus": [1, 0], "started_unix": 1, "attempts": []}
        _, meta = self.launch([], previous=previous)
        self.assertEqual(meta["gpus"], [1, 0])
        with self.assertRaisesRegex(SystemExit, "original GPU/shard mapping"):
            self.launch(["--gpus", "0,1"], previous=previous)

    def test_duplicate_gpu_rejected(self):
        with self.assertRaisesRegex(SystemExit, "unique"):
            self.launch(["--gpus", "0,0"])

    def test_known_secret_rejected_without_echoing_value(self):
        secret = b"private-test-value-0123456789"
        with self.assertRaises(ValueError) as caught:
            package_project.reject_secrets(b"value=" + secret, "example.txt", [secret])
        self.assertNotIn(secret.decode(), str(caught.exception))

    def test_allowlist_does_not_pick_up_private_files_or_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("private")
            (root / "models").mkdir()
            (root / "models/weights.bin").write_bytes(b"large")
            (root / "results/phase1/predictions").mkdir(parents=True)
            (root / "results/phase1/predictions/cache.json").write_text("{}")
            names = {p.relative_to(root).as_posix() for p in package_project.selected_paths(root)}
            self.assertNotIn(".env", names)
            self.assertNotIn("models/weights.bin", names)
            self.assertNotIn("results/phase1/predictions/cache.json", names)


if __name__ == "__main__":
    unittest.main()
