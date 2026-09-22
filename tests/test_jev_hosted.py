import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_jev_rcta_hosted_full as runner
import jev_rcta_client as transport
from probe_jev_hosted import ENDPOINT
from report_jev_rcta_hosted import Replay
from resume_jev_rcta_hosted import RecoveryClient
import http.client


class HostedTests(unittest.TestCase):
    def exercise(self, fail=False, preflight_only=False):
        seen = []
        rows = [{"question_ID": "%s_%d" % (source, i), "source": source,
                 "history": [{"step": 0, "name": "Worker", "content": "x" * (i + 1)}],
                 "mistake_agent": "Worker", "mistake_step": 0}
                for source in ("a", "b", "c", "d", "e") for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe = root / "results/jev_rcta_hosted_probe"
            probe.mkdir(parents=True)
            (probe / "probe.json").write_text(json.dumps({"passed": True, "endpoint": ENDPOINT, "model": runner.ev.MODEL}))
            def prepare(out, manifest):
                for name in ("calls", "predictions", "errors"):
                    (out / name).mkdir(parents=True, exist_ok=True)
                return {}
            def case(row, out, key, stop):
                seen.append(row["question_ID"])
                if fail:
                    raise RuntimeError("preflight failure")
                return {"predicted_step": 0, "predicted_role": "Worker", "method": "test",
                        **{field: [0] for field in ("recall_candidates", "trace_seeds", "expanded_candidates", "final_candidates")}}
            argv = ["runner", "--workers", "2"] + (["--preflight-only"] if preflight_only else [])
            with patch.object(runner.ev, "ROOT", root), patch.object(runner, "load_full", return_value=(rows, {"revision": "test"})), \
                    patch.object(runner, "prepare", side_effect=prepare), patch.object(runner, "read_key", return_value="private-test-key"), \
                    patch.object(runner, "run_case", side_effect=case), patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(SystemExit):
                        runner.main()
                else:
                    runner.main()
            out = root / "results/jev_rcta_hosted_full"
            return seen, (out / "preflight.json").exists(), json.loads((out / "progress.json").read_text())

    def test_full_only_starts_after_all_five_preflight_cases(self):
        seen, passed, progress = self.exercise()
        self.assertEqual(seen[:5], [s + "_0" for s in "abcde"])
        self.assertEqual(len(seen), 10)
        self.assertTrue(passed)
        self.assertEqual(progress["status"], "complete")

    def test_failed_preflight_prevents_full_launch(self):
        seen, passed, progress = self.exercise(fail=True)
        self.assertEqual(len(seen), 1)
        self.assertFalse(passed)
        self.assertEqual(progress["status"], "error")

    def test_preflight_only_stops_after_five(self):
        seen, passed, progress = self.exercise(preflight_only=True)
        self.assertEqual(len(seen), 5)
        self.assertTrue(passed)
        self.assertEqual(progress["status"], "preflight_complete")

    def test_offline_replay_detects_modified_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            payload = {"model": runner.ev.MODEL, "state": {"history": "original evidence"}, "questions": {}}
            saved = {"endpoint": ENDPOINT, "request": payload,
                     "request_sha256": runner.ev.sha(runner.ev.dumps(payload).encode()),
                     "response": {"model": runner.ev.MODEL, "answers": {}}}
            (directory / "sample.json").write_text(json.dumps(saved))
            replay = Replay(directory)
            self.assertEqual(replay.call(payload["state"], {}, "sample"), saved["response"])
            with self.assertRaisesRegex(ValueError, "Replay request"):
                replay.call({"history": "changed"}, {}, "sample")

    def test_disconnect_recovery_preserves_request_and_quota_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = RecoveryClient("private", out / "calls", out, endpoint=ENDPOINT)
            reply = {"answers": {}}
            with patch.object(transport.Client, "call", side_effect=[http.client.RemoteDisconnected("gone"), reply]) as call, \
                    patch.object(client.stop_event, "wait", return_value=False):
                self.assertEqual(client.call({"original": "evidence"}, {"q": "unchanged"}, "tag"), reply)
                self.assertEqual(call.call_args_list[0], call.call_args_list[1])
            with patch.object(transport.Client, "call", side_effect=transport.BalanceExhausted("stop")) as call:
                with self.assertRaises(transport.BalanceExhausted):
                    client.call({}, {}, "tag")
                self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
