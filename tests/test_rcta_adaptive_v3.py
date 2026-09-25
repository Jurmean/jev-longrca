import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate as ev
import evaluate_jev_rcta_adaptive_v3 as runner
import jev_rcta_adaptive_v3 as method
from rcta_demo_v3 import HISTORY, ScriptedClient
from rcta_evidence import EvidenceIndex
from rcta_retrieval import Retrieval, reciprocal_rank_fusion


def certain(opts, selected, probability=.98):
    return {"choice": selected, "probabilities": {
        k: probability if k == selected else (1 - probability)/(len(opts) - 1) for k in opts}}


class RetrievalTests(unittest.TestCase):
    def test_short_view_retains_zero_overlap_evidence_without_clipping(self):
        history = [{"step": 0, "name": "W", "content": "alpha expected target"},
                   {"step": 1, "name": "W", "content": "执行了另一个动作🙂"}]
        retrieval = Retrieval(EvidenceIndex(history))
        packet = retrieval.packet(["alpha"], 4000)
        self.assertTrue(packet["view_complete"])
        self.assertEqual(packet["retrieval_strategy"], "all_eligible_original_text")
        self.assertEqual([r["content"] for r in packet["records"]], [h["content"] for h in history])

    def test_long_view_stays_bounded_and_does_not_claim_full_text_coverage(self):
        history = [{"step": i, "name": "W", "content": "alpha beta " * 400} for i in range(20)]
        packet = Retrieval(EvidenceIndex(history)).packet(["alpha"], 2500)
        self.assertFalse(packet["view_complete"])
        self.assertEqual(packet["retrieval_strategy"], "global_rrf_mmr_excerpts")
        self.assertLessEqual(len(ev.dumps(packet["records"]).encode()), 2500)

    def test_rrf_formula_and_duplicate_document_does_not_get_two_votes(self):
        scores = reciprocal_rank_fusion([["a", "a", "b"], ["b", "c"]], k=60)
        self.assertAlmostEqual(scores["a"], 1/61)
        self.assertAlmostEqual(scores["b"], 1/62 + 1/61)

    def test_mmr_penalizes_redundant_passages(self):
        history = [{"step": i, "name": "W", "content": content} for i, content in enumerate([
            "alpha target violation", "alpha target violation", "beta different implementation"])]
        index = EvidenceIndex(history)
        refs = list(index.spans)
        retrieval = Retrieval(index)
        ranked = list(retrieval.diverse(dict(zip(refs, [1, .98, .95]))))
        self.assertEqual(ranked[:2], [refs[0], refs[2]])

    def test_global_query_can_reach_early_evidence_far_from_late_failure(self):
        history = [{"step": i, "name": "W", "content": "routine unrelated logging"} for i in range(100)]
        history[3]["content"] = "alpha_rule introduced invalid rule_id conversion"
        history[99]["content"] = "alpha_rule failed"
        index = EvidenceIndex(history)
        retrieval = Retrieval(index)
        packet = retrieval.packet(["alpha_rule rule_id conversion"], 1700)
        self.assertEqual(packet["records"][0]["step"], 3)

    def test_feedback_changes_retrieval_and_excerpts_keep_utf8_provenance(self):
        history = [{"step": 0, "name": "W", "content": "alpha original plan 汉🙂"},
                   {"step": 1, "name": "W", "content": "beta later repair 汉🙂"}]
        index = EvidenceIndex(history)
        retrieval = Retrieval(index)
        self.assertEqual(index.spans[retrieval.ranked(["alpha"])[0]]["step"], 0)
        self.assertEqual(index.spans[retrieval.ranked(["beta"])[0]]["step"], 1)
        for r in index.spans:
            excerpt = retrieval.excerpt(r, "repair", 23)
            raw = history[excerpt["step"]]["content"].encode()[excerpt["start_byte"]:excerpt["end_byte"]]
            self.assertEqual(raw, excerpt["content"].encode())
            self.assertEqual(ev.sha(raw), excerpt["sha256"])


class PipelineTests(unittest.TestCase):
    def test_conditional_role_question_follows_root_selection_and_verification_is_fresh(self):
        c = ScriptedClient()
        result = method.predict(HISTORY, c)
        final = next(r for r in c.calls if r["tag"].endswith("_final"))["request"]
        verify = next(r for r in c.calls if r["tag"].endswith("_verify"))["request"]
        self.assertNotIn("responsible_role", final["questions"])
        self.assertIn("responsible_role", verify["questions"])
        self.assertEqual(verify["state"]["target_step"], result["predicted_step"])
        for forbidden in ("previous_labels", "probabilities", "final_answers", "responsible_role", "observed_anchor"):
            self.assertNotIn(forbidden, ev.dumps(verify["state"]))
        self.assertEqual(result["verified_step"], 1)
        self.assertEqual(result["predicted_role"], "Planner")

    def test_question_hint_controls_next_action(self):
        class RepairFirst(ScriptedClient):
            def call(self, state, qs, tag):
                r = super().call(state, qs, tag)
                if "assessment" in qs:
                    r["answers"]["need"] = certain(qs["need"]["criteria"], "repair")
                return r
        result = method.predict(HISTORY, RepairFirst())
        trace = [t for t in result["search_trace"] if "round" in t]
        self.assertEqual(trace[0]["action"], "inspect")
        self.assertEqual(trace[1]["action"], "repair")

    def test_moderate_upstream_probability_discovers_hypothesis_without_claiming_proof(self):
        class Moderate(ScriptedClient):
            def call(self, state, qs, tag):
                r = super().call(state, qs, tag)
                if tag.endswith("_upstream") and state["target_step"] == 2:
                    r["answers"]["relation"] = certain(qs["relation"]["criteria"], "upstream", .6)
                return r
        result = method.predict(HISTORY, Moderate())
        queued = [t for t in result["search_trace"] if t["action"] == "queue_upstream_hypothesis"]
        self.assertTrue(queued)
        self.assertFalse(queued[0]["verified_causal_edge"])
        self.assertEqual(result["focus_order"][:2], [2, 1])

    def test_iteration_uses_original_evidence_and_can_be_ablated(self):
        p = method.Pipeline(HISTORY, ScriptedClient())
        before = p.queries()
        p.remember(p.index.by_step[2][0])
        self.assertNotEqual(p.queries(), before)
        self.assertIn(HISTORY[2]["content"], p.queries())
        p = method.Pipeline(HISTORY, ScriptedClient(), replace(method.Config(), iterative_queries=0))
        before = p.queries()
        p.remember(p.index.by_step[2][0])
        self.assertEqual(p.queries(), before)

    def test_all_three_ablations_are_explicit_and_have_valid_outputs(self):
        for name in ("global_retrieval", "iterative_queries", "factored_verification"):
            with self.subTest(name=name):
                result = method.predict(HISTORY, ScriptedClient(), replace(method.Config(), **{name: 0}))
                self.assertEqual(result["method"], method.PROTOCOL)
                self.assertIn(result["decision_status"], ("supported", "best_effort", "abstained"))
                self.assertLessEqual(result["budget"]["logical_calls"], method.Config().max_calls)

    def test_budget_yields_to_final_calls_and_no_network_with_zero_search_allowance(self):
        c = ScriptedClient()
        result = method.predict(HISTORY, c, replace(method.Config(), max_calls=7))
        self.assertLessEqual(len(c.calls), 7)
        self.assertEqual([r["tag"].split('_', 1)[1] for r in c.calls][-2:], ["final", "verify"])
        self.assertIsNotNone(result["predicted_step"])
        c = ScriptedClient()
        result = method.predict(HISTORY, c, replace(method.Config(), max_calls=1))
        self.assertEqual(c.calls, [])
        self.assertIsNone(result["predicted_step"])

    def test_known_refutation_blocks_supported_and_point_output(self):
        class Refuted(ScriptedClient):
            def call(self, state, qs, tag):
                r = super().call(state, qs, tag)
                if "origin" in qs:
                    r["answers"]["origin"] = certain(qs["origin"]["criteria"], "no_violation")
                return r
        result = method.predict(HISTORY, Refuted())
        self.assertIsNone(result["predicted_step"])
        self.assertIsNone(result["verified_step"])

    def test_uncertain_fact_does_not_become_verified_by_repetition(self):
        class Uncertain(ScriptedClient):
            def call(self, state, qs, tag):
                r = super().call(state, qs, tag)
                if "outcome" in qs:
                    r["answers"]["outcome"] = certain(qs["outcome"]["criteria"], "unknown", .6)
                return r
        result = method.predict(HISTORY, Uncertain())
        self.assertEqual(result["decision_status"], "best_effort")
        self.assertIsNone(result["verified_step"])

    def test_gold_fields_excluded_and_replay_hashes_are_exact(self):
        history = copy.deepcopy(HISTORY)
        for h in history:
            h.update(mistake_step=999, mistake_reason="SECRET_GOLD")
        c = ScriptedClient()
        expected = method.predict(history, c)
        self.assertNotIn("SECRET_GOLD", ev.dumps(c.calls))
        class Replay:
            def __init__(self, test):
                self.records, self.test = iter(c.calls), test
            def call(self, state, qs, tag):
                r = next(self.records)
                self.test.assertEqual(r["tag"], tag)
                self.test.assertEqual(r["request_sha256"], ev.sha(ev.dumps({"model":ev.MODEL,"state":state,"questions":qs}).encode()))
                return r["response"]
        self.assertEqual(method.predict(history, Replay(self)), expected)


class RunnerTests(unittest.TestCase):
    def test_v3_offline_entry_freezes_v3_sources_and_never_reads_key(self):
        with tempfile.TemporaryDirectory() as out, patch.object(runner.runner.common, "get_key") as key, \
             patch.object(runner.runner, "Client") as client, patch.object(sys, "argv", ["v3", "--demo", "--output", out]), \
             patch("sys.stdout", new_callable=io.StringIO):
            runner.main()
            key.assert_not_called()
            client.assert_not_called()
            config = json.loads((Path(out) / "config.json").read_text())
            self.assertEqual(config["protocol"], method.PROTOCOL)
            self.assertIn("rcta_retrieval.py", config["source_sha256"])
            self.assertIn("jev_rcta_adaptive_v3.py", config["source_sha256"])
            self.assertEqual(json.loads((Path(out)/"demo.json").read_text())["api_calls"], 0)


if __name__ == "__main__":
    unittest.main()
