import copy
import json
import io
from pathlib import Path
import sys
import tempfile
import urllib.error
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate as ev
import jev_rcta as method
import evaluate_jev_rcta as runner
import jev_rcta_client as transport


class FakeClient:
    def __init__(self, relation="upstream", suspect="2"):
        self.relation, self.suspect = relation, suspect
        self.requests = []

    def call(self, state, questions, tag):
        self.requests.append((state, questions, tag))
        answers = {}
        for key, question in questions.items():
            options = question["criteria"]
            preferred = {"suspect": self.suspect, "anchor": "2", "relation": self.relation,
                         "root_step": "0", "responsible_role": "Verifier"}.get(key)
            selected = preferred if preferred in options else next(iter(options))
            answers[key] = {"choice": selected,
                            "probabilities": {s: float(s == selected) for s in options}}
        return {"answers": answers}


class RCTATests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"step": 0, "name": "Planner (-> Worker)", "content": "Use the incorrect API.", "role": "assistant"},
            {"step": 1, "name": "Other (-> Someone)", "content": "An unrelated handoff.", "role": "assistant"},
            {"step": 2, "name": "Worker", "content": "Applied the instructed API.", "role": "assistant"},
            {"step": 3, "name": "Verifier", "content": "Tests fail.", "role": "assistant"},
        ]

    def test_upstream_discovery_and_independent_role(self):
        with patch.dict(method.CONFIG, recall_k=1):
            result = method.predict(self.history, FakeClient())
        self.assertEqual(result["recall_candidates"], [2])
        self.assertIn(0, result["expanded_candidates"])
        self.assertEqual(result["predicted_step"], 0)
        self.assertEqual(result["predicted_role"], "Verifier")
        self.assertEqual(result["relation_hypotheses"][0]["upstream"], 0)
        self.assertTrue(result["relation_hypotheses"][0]["addressed"])

    def test_uncertain_or_repaired_never_hard_deletes_candidates(self):
        for relation in ("uncertain", "repaired", "local", "unrelated"):
            with patch.dict(method.CONFIG, recall_k=1):
                result = method.predict(self.history, FakeClient(relation=relation))
            self.assertEqual(result["expanded_candidates"], [0, 2])

    def test_no_suspects_falls_back_to_observed_anchor(self):
        result = method.predict(self.history, FakeClient(suspect="none"))
        self.assertTrue(result["anchor_fallback"])
        self.assertEqual(result["recall_candidates"], [2])

    def test_unknown_retains_candidates(self):
        result = method.predict(self.history, FakeClient(suspect="uncertain"))
        self.assertFalse(result["anchor_fallback"])
        self.assertEqual(len(result["recall_candidates"]), 3)

    def test_annotation_allowlist_and_no_mutation(self):
        history = copy.deepcopy(self.history)
        for h in history:
            h["mistake_reason"] = "SECRET_GOLD"
        before = copy.deepcopy(history)
        client = FakeClient()
        method.predict(history, client)
        self.assertNotIn("SECRET_GOLD", json.dumps(client.requests))
        self.assertNotIn("mistake_reason", json.dumps(client.requests))
        self.assertEqual(history, before)

    def test_singleton_questions_are_resolved_without_api(self):
        client = FakeClient()
        answers = method.ask(client, {}, {"only": method.choice("Pick", {"a": "only option"})}, "single")
        self.assertEqual(answers["only"]["choice"], "a")
        self.assertIsNone(answers["only"]["confidence"])
        self.assertFalse(client.requests)
        result = method.predict([{"step": 0, "name": "Worker", "content": "Failed task"}], client)
        self.assertEqual(result["predicted_step"], 0)
        self.assertEqual(result["predicted_role"], "Worker")
        self.assertTrue(all(len(q["criteria"]) >= 2 for _, qs, _ in client.requests for q in qs.values()))

    def test_lossless_fragments_and_budget(self):
        history = copy.deepcopy(self.history)
        history[2]["content"] = "汉字🙂\\\n\"" * 10000
        batches = list(method.segments(history))
        for h in history:
            self.assertEqual("".join(r["content"] for b in batches for r in b if r["step"] == h["step"]), h["content"])
        client = FakeClient()
        result = method.predict(history, client)
        self.assertGreater(result["segments"], 1)
        for state, questions, _ in client.requests:
            self.assertLessEqual(len(ev.dumps({"model": ev.MODEL, "state": state, "questions": questions}).encode()),
                                 method.CONFIG["request_bytes"])

    def test_unmatched_handoff_is_only_plan_context(self):
        self.assertEqual(method.handoff(self.history, 3), (1, False))

    def test_invalid_output_fails_and_budget_checked_before_call(self):
        client = FakeClient()
        with patch.dict(method.CONFIG, request_bytes=10):
            with self.assertRaisesRegex(ValueError, "budget"):
                method.predict(self.history, client)
        self.assertFalse(client.requests)
        with patch.object(client, "call", return_value={"answers": {}}):
            with self.assertRaisesRegex(ValueError, "Invalid Choice"):
                method.predict(self.history, client)

    def test_experiment_cannot_overwrite_other_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            runner.prepare(out, {"revision": "test", "files": []})
            with self.assertRaisesRegex(ValueError, "another protocol"):
                runner.prepare(out, {"revision": "changed", "files": []})

    def test_multiple_handoff_hops_are_bounded(self):
        history = [
            {"step": 0, "name": "Boss (-> Planner)", "content": "wrong goal"},
            {"step": 1, "name": "Planner (-> Worker)", "content": "wrong instruction"},
            {"step": 2, "name": "Worker", "content": "wrong implementation"},
        ]
        with patch.dict(method.CONFIG, recall_k=1):
            result = method.predict(history, FakeClient())
        self.assertEqual([(e["upstream"], e["candidate"]) for e in result["relation_hypotheses"]], [(1, 2), (0, 1)])
        self.assertEqual(result["expanded_candidates"], [0, 1, 2])

    def test_large_candidate_set_reduces_and_remains_valid(self):
        history = [{"step": i, "name": "Worker", "content": "log " + str(i)} for i in range(90)]
        with patch.dict(method.CONFIG, segment_steps=4):
            result = method.predict(history, FakeClient())
        self.assertGreater(len(result["recall_candidates"]), 16)
        self.assertLessEqual(len(result["trace_seeds"]), 16)
        self.assertLessEqual(len(result["final_candidates"]), 8)
        self.assertIn(result["predicted_step"], result["final_candidates"])

    def test_quota_exhaustion_stops_without_retry_and_persists(self):
        for status in (402, 429, 503):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                client = transport.Client("test-private-key", out / "calls", out)
                error = urllib.error.HTTPError("https://example.invalid", status, "error", {},
                                                io.BytesIO(b'{"error":"insufficient_balance"}'))
                with patch.object(transport.urllib.request, "urlopen", side_effect=error) as network, \
                        patch.object(transport.time, "sleep") as sleep:
                    with self.assertRaises(transport.BalanceExhausted):
                        client.call({}, {}, "first")
                    with self.assertRaises(transport.BalanceExhausted):
                        client.call({}, {}, "second")
                    self.assertEqual(network.call_count, 1)
                    sleep.assert_not_called()
                self.assertTrue((out / "balance_stop.json").exists())
                self.assertNotIn("test-private-key", (out / "balance_stop.json").read_text())

    def test_cache_reuses_exact_request_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = transport.Client("test-private-key", out / "calls", out)
            reply = {"model": ev.MODEL, "answers": {"a": {"choice": "x"}}}
            with patch.object(transport.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(reply).encode())) as network:
                client.call({"text": "a"}, {}, "first")
                client.call({"text": "a"}, {}, "first")
                self.assertEqual(network.call_count, 1)
                with self.assertRaisesRegex(ValueError, "Cached request"):
                    client.call({"text": "b"}, {}, "first")


if __name__ == "__main__":
    unittest.main()
