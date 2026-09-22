"""Meaningful checks against the real tokenizer/upstream sequence builder; no GPU needed."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_laya as laya
try:
    from transformers import AutoTokenizer
    import torch
except ImportError:
    AutoTokenizer = None


@unittest.skipIf(AutoTokenizer is None, "Laya tests require the isolated inference environment")
class LayaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(str(laya.MODEL_DIR / "tokenizer"), local_files_only=True)
        cls.budget = laya.Budget(cls.tok)

    def test_lossless_unicode_recall_and_option_limits(self):
        history = [{"step": i, "name": "Worker", "role": "assistant",
                    "content": "原始日志🙂\n" * (5000 if i == 7 else 12)} for i in range(34)]
        shared = {"outcome": "failed"}
        chunks = list(laya.token_chunks(history, shared, self.budget))
        for h in history:
            parts = [r for batch in chunks for r in batch if r["step"] == h["step"]]
            self.assertEqual("".join(r["content"] for r in parts), h["content"])
            self.assertEqual(parts[0]["content_start"], 0)
            self.assertEqual(parts[-1]["content_end"], len(h["content"]))
        for batch in chunks:
            ids = sorted({r["step"] for r in batch})
            self.assertLessEqual(len(ids), 16)
            self.assertTrue(self.budget.fits(dict(shared, segment=batch), laya.ev.questions(ids)))

    def test_no_hidden_header_or_state_truncation(self):
        from rl_common import build_sequence
        from rl_agent_api import RLAgent
        state = {"history": "Evidence [MASK] must be visible. " * 100}
        for q in laya.ev.questions(list(range(16)), ["Planner", "Worker"]).values():
            header = self.budget.header(q)
            seq, markers = build_sequence(self.tok, state, RLAgent._to_internal(q), 8192, header + 16)
            tokens = self.budget.state_tokens(state)
            self.assertEqual(len(seq), len(tokens) + header + 4)
            self.assertEqual(seq[-len(tokens)-1:-1], tokens)
            self.assertEqual(len(markers), len(q["criteria"]))

    def test_reject_long_option_text(self):
        q = {"type": "choice", "instructions": "choose", "criteria": {"a": "long option " * 100, "b": "b"}}
        with self.assertRaisesRegex(ValueError, "48 tokens"):
            self.budget.header(q)

    def test_client_refuses_truncation_before_forward(self):
        class Agent:
            tok = self.tok
            cfg = {}
            def system_one(self, *args):
                raise AssertionError("Should not reach model")
        with tempfile.TemporaryDirectory() as tmp:
            client = laya.LocalClient(Agent(), Path(tmp))
            with self.assertRaisesRegex(ValueError, "Refusing silent"):
                client.call({"text": "evidence " * 20000}, laya.ev.questions([1, 2]), "oversized")

    def test_singleton_and_cache_are_not_model_calls(self):
        from rl_agent_api import RLAgent
        class Agent:
            tok = self.tok
            cfg = {}
            _to_internal = staticmethod(RLAgent._to_internal)
            def system_one(self, *args):
                raise AssertionError("Singleton does not require inference")
        with tempfile.TemporaryDirectory() as tmp:
            client = laya.LocalClient(Agent(), Path(tmp))
            a = client.call({"text": "test"}, laya.ev.questions([7]), "one")
            self.assertEqual(a["answers"]["root_step"]["choice"], "7")
            self.assertEqual(a["usage"]["input_tokens"], 0)
            self.assertEqual(a, client.call({"text": "test"}, laya.ev.questions([7]), "one"))
            with self.assertRaisesRegex(ValueError, "differs"):
                client.call({"text": "changed"}, laya.ev.questions([7]), "one")


if __name__ == "__main__":
    unittest.main()
