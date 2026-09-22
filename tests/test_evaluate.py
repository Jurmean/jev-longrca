import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("evaluate", Path(__file__).resolve().parents[1] / "scripts/evaluate.py")
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.row = {"question_ID": "test", "source": "s", "history": [
            {"step": 0, "name": "Planner (-> Worker)", "role": "assistant", "content": "wrong plan"},
            {"step": 1, "name": "Worker", "role": "assistant", "content": "executed plan"},
            {"step": 2, "name": "Verifier", "role": "assistant", "content": "failure"}],
            "mistake_agent": "Planner", "mistake_step": 0, "mistake_reason": "SECRET_GOLD"}

    def test_independent_role_and_step_scoring(self):
        a = ev.score_row(self.row, {"predicted_role": " Worker ", "predicted_step": 0})
        self.assertFalse(a["role_correct"])
        self.assertTrue(a["step_exact"])
        b = ev.score_row(self.row, {"predicted_role": "  PLANNER (-> Worker) ", "predicted_step": 99})
        self.assertTrue(b["role_correct"])
        self.assertFalse(b["step_within_5"])
        self.assertIsNone(b["absolute_step_error"])
        self.assertFalse(ev.score_row(self.row, {"predicted_step": True})["valid_step"])

    def test_unknown_role_is_invalid(self):
        self.assertFalse(ev.score_row(self.row, {"predicted_role": "external", "predicted_step": 0})["valid_role"])

    def test_utf8_chunking_preserves_every_character(self):
        history = copy.deepcopy(self.row["history"])
        history[1]["content"] = "汉字🙂\\\n" * 20000
        batches = list(ev.chunks(history))
        for h in history:
            rebuilt = "".join(r["content"] for batch in batches for r in batch if r["step"] == h["step"])
            self.assertEqual(rebuilt, h["content"])
        self.assertGreater(len(batches), 1)

    def test_handoff_uses_recipient(self):
        self.assertEqual(ev.nearest_handoff(self.row["history"], 1), 0)

    def test_inference_only_observes_history(self):
        class FakeClient:
            def call(self, state, questions, tag):
                serialized = ev.dumps(state) + ev.dumps(questions)
                assert "SECRET_GOLD" not in serialized
                assert "mistake_" not in serialized
                return {"answers": {"root_step": {"choice": "0", "probabilities": {"0": 1}},
                                    "responsible_role": {"choice": "Verifier"}}}
        result = ev.predict(self.row["history"], FakeClient())
        self.assertEqual(result["predicted_role"], "Verifier")
        self.assertEqual(result["predicted_step"], 0)

    def test_source_weighted_mae_and_missing_outputs(self):
        rows = []
        for i, source in enumerate(["a", "a", "b"]):
            r = copy.deepcopy(self.row)
            r.update(question_ID=str(i), source=source)
            rows.append(r)
        summary = ev.summarize(rows, {"0": {"predicted_step": 2}, "2": {"predicted_step": 0}}, {"revision": "test"})
        self.assertAlmostEqual(summary["overall"]["source_weighted_valid_output_root_mae"], 4 / 3)
        self.assertAlmostEqual(summary["overall"]["step_exact"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
