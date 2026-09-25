import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_jev_rcta_adaptive_mini as scheduler


class SchedulingTests(unittest.TestCase):
    def exercise(self, failure=None, preflight_only=False, connection_failure=False):
        rows = [{"question_ID": source + str(i), "source": source,
                 "history": [{"step": 0, "name": "Worker", "content": "x" * (i + 1)}],
                 "mistake_step": 0, "mistake_agent": "Worker"} for source in "abcde" for i in range(2)]
        seen = []
        with tempfile.TemporaryDirectory() as temp:
            def run(row, out, key, stop):
                qid = row["question_ID"]
                seen.append(qid)
                if qid == failure:
                    stop.set()
                    scheduler.runner.write_json(out / "balance_stop.json", {"status": "paused_insufficient_balance"})
                    raise scheduler.BalanceExhausted("quota")
                return {"predicted_step": None, "predicted_role": None, "method": "fixture",
                        "api_calls": 1, "input_tokens": 10, "decision_status": "abstained",
                        "decision_reason": "uncertain_final_choice", "recall_candidates": [0],
                        "expanded_candidates": [0], "final_candidates": [0]}
            with patch.object(scheduler.ev, "load_data", return_value=(rows, {"revision": "fixture"})), \
                 patch.object(scheduler.runner, "get_key", return_value="fixture-private"), \
                 patch.object(scheduler.runner, "preflight", side_effect=RuntimeError("connection") if connection_failure else None), \
                 patch.object(scheduler, "run_case", side_effect=run), \
                 patch.object(sys, "argv", ["scheduler", "--output", temp, "--workers", "1"] +
                              (["--preflight-only"] if preflight_only else [])), contextlib.redirect_stdout(io.StringIO()):
                if failure or connection_failure:
                    with self.assertRaises(SystemExit):
                        scheduler.main()
                else:
                    scheduler.main()
            return seen, json.loads((Path(temp) / "progress.json").read_text())

    def test_five_source_gate_and_abstentions_do_not_change_parameters(self):
        seen, progress = self.exercise()
        self.assertEqual(seen[:5], [s + "0" for s in "abcde"])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(progress["status"], "complete")
        self.assertEqual(progress["abstentions"], 10)

    def test_preflight_only_and_connection_failure(self):
        seen, progress = self.exercise(preflight_only=True)
        self.assertEqual(len(seen), 5)
        self.assertEqual(progress["status"], "preflight_complete")
        seen, progress = self.exercise(connection_failure=True)
        self.assertEqual(seen, [])
        self.assertEqual(progress["status"], "error")

    def test_quota_stops_preflight_or_main_queue(self):
        for failure, count in (("a0", 1), ("a1", 6)):
            with self.subTest(failure=failure):
                seen, progress = self.exercise(failure=failure)
                self.assertEqual(len(seen), count)
                self.assertEqual(progress["status"], "paused_insufficient_balance")


if __name__ == "__main__":
    unittest.main()
