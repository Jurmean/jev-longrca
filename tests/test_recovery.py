from pathlib import Path
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_full as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        recovery.STOP.clear()

    def tearDown(self):
        recovery.STOP.clear()

    def test_retries_preserve_model_inputs(self):
        state, questions = {"history": "original evidence"}, {"root": "original question"}
        with tempfile.TemporaryDirectory() as directory:
            client = recovery.RecoveryClient("private-key", Path(directory) / "calls")
            with patch.object(recovery, "LOG", Path(directory) / "retries.jsonl"), \
                    patch.object(recovery.STOP, "wait", return_value=False) as sleep, \
                    patch.object(recovery.BASE_CLIENT, "call", side_effect=[RuntimeError("HTTP 403: Forbidden"), {"ok": True}]) as call:
                self.assertEqual(client.call(state, questions, "tag"), {"ok": True})
                self.assertEqual(call.call_args_list[0], call.call_args_list[1])
                self.assertEqual(call.call_args.args, (state, questions, "tag"))
                sleep.assert_called_once_with(15)

    def test_authentication_errors_do_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            client = recovery.RecoveryClient("private-key", Path(directory))
            with patch.object(recovery.BASE_CLIENT, "call", side_effect=RuntimeError("HTTP 401: Unauthorized")) as call:
                with self.assertRaises(RuntimeError):
                    client.call({}, {}, "tag")
                self.assertEqual(call.call_count, 1)

    def test_balance_exhaustion_stops_all_new_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results/full/predictions").mkdir(parents=True)
            error = recovery.urllib.error.HTTPError("https://api.typesafe.ai/v1/systemone", 402,
                "Payment Required", {}, io.BytesIO(b'{"error":"insufficient_balance"}'))
            with patch.object(recovery.ev, "ROOT", root), \
                    patch.object(recovery, "ORIGINAL_URLOPEN", side_effect=error) as opener:
                with self.assertRaises(recovery.BillingStop):
                    recovery.guarded_urlopen("https://api.typesafe.ai/v1/systemone")
                with self.assertRaises(recovery.BillingStop):
                    recovery.guarded_urlopen("https://api.typesafe.ai/v1/systemone")
                self.assertEqual(opener.call_count, 1)
                self.assertTrue((root / "results/full/balance_stop.json").exists())

    def test_rate_limits_are_not_balance_exhaustion(self):
        raw = b'{"error":"rate limit exceeded"}'
        error = recovery.urllib.error.HTTPError("https://api.typesafe.ai/v1/systemone", 429,
            "Too Many Requests", {}, io.BytesIO(raw))
        with patch.object(recovery, "ORIGINAL_URLOPEN", side_effect=error):
            with self.assertRaises(recovery.urllib.error.HTTPError) as raised:
                recovery.guarded_urlopen("https://api.typesafe.ai/v1/systemone")
            self.assertEqual(raised.exception.read(), raw)
            self.assertFalse(recovery.STOP.is_set())

    def test_transport_retries_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            client = recovery.RecoveryClient("private-key", Path(directory))
            with patch.object(recovery, "LOG", Path(directory) / "retries.jsonl"), \
                    patch.object(recovery.STOP, "wait", return_value=False), \
                    patch.object(recovery.BASE_CLIENT, "call", side_effect=RuntimeError("HTTP 503: no healthy upstream")) as call:
                with self.assertRaises(RuntimeError):
                    client.call({}, {}, "tag")
                self.assertEqual(call.call_count, 4)

    def test_only_known_transient_model_error_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            client = recovery.RecoveryClient("private-key", Path(directory))
            with patch.object(recovery, "LOG", Path(directory) / "retries.jsonl"), \
                    patch.object(recovery.STOP, "wait", return_value=False), \
                    patch.object(recovery.BASE_CLIENT, "call", side_effect=[RuntimeError("HTTP 400: Unknown model: jev-1.13.0"), {"ok": True}]) as call:
                self.assertEqual(client.call({}, {}, "tag"), {"ok": True})
                self.assertEqual(call.call_count, 2)
            with patch.object(recovery.BASE_CLIENT, "call", side_effect=RuntimeError("HTTP 400: invalid input")) as call:
                with self.assertRaises(RuntimeError):
                    client.call({}, {}, "tag")
                self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
