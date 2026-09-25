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
import evaluate_jev_rcta_adaptive_v2 as runner
import jev_rcta_adaptive_v2 as method
from rcta_demo import HISTORY, ScriptedClient
from rcta_directory import Directory
from rcta_evidence import EvidenceIndex


def answer(options, selected, probability=.98):
    return {"choice": selected, "probabilities": {
        k: probability if k == selected else (1 - probability) / (len(options) - 1) for k in options}}


class DirectoryTests(unittest.TestCase):
    def test_all_segments_reachable_and_utf8_preview_is_original_text(self):
        history = [{"step": i, "name": "Worker", "content": ('汉🙂\\\"' * 200 + ' requirement missing date ')*3}
                   for i in range(40)]
        index = EvidenceIndex(history)
        directory = Directory(index)
        self.assertTrue(directory.audit()["all_segments_reachable"])
        for key in directory.nodes[directory.root]["children"]:
            for excerpt in directory.preview(key)["excerpts"]:
                raw = history[excerpt["step"]]["content"].encode()[excerpt["start_byte"]:excerpt["end_byte"]]
                self.assertEqual(raw, excerpt["content"].encode())
                self.assertEqual(ev.sha(raw), excerpt["sha256"])

    def test_unknown_route_preserves_siblings_without_activating_all_nodes(self):
        history = [{"step": i, "name": "Worker", "content": 'x' * 1800} for i in range(90)]
        p = method.Pipeline(history, ScriptedClient())
        segment = p.next_segment()
        self.assertTrue(p.routed[0]["uncertain"])
        self.assertTrue(p.frontier)
        self.assertEqual(p.nodes, {})
        p.read_segment(segment)
        self.assertTrue(p.leads)
        self.assertEqual(p.nodes, {})
        self.assertEqual(len(p.cards), 1)


class PipelineTests(unittest.TestCase):
    def test_upstream_focus_and_evidence_supported_output(self):
        c = ScriptedClient()
        p = method.predict(HISTORY, c)
        self.assertEqual(p["focus_order"], [2, 1])
        self.assertEqual(p["predicted_step"], 1)
        self.assertEqual(p["verified_step"], 1)
        self.assertEqual(p["predicted_role"], "Planner")
        self.assertEqual(p["decision_status"], "supported")
        self.assertEqual(p["evidence_refs"][0]["step"], 1)
        self.assertEqual([x["tag"].split('_', 1)[1] for x in c.calls][-2:], ["final", "verify"])

    def test_uncertain_prediction_is_not_misreported_as_verified(self):
        class Uncertain(ScriptedClient):
            def call(self, state, qs, tag):
                result = super().call(state, qs, tag)
                if "root_step" in qs:
                    result["answers"]["root_step"] = answer(qs["root_step"]["criteria"], "1", .55)
                if "support" in qs:
                    result["answers"]["support"] = answer(qs["support"]["criteria"], "unknown", .6)
                return result
        p = method.predict(HISTORY, Uncertain())
        self.assertEqual(p["predicted_step"], 1)
        self.assertIsNone(p["verified_step"])
        self.assertEqual(p["decision_status"], "best_effort")
        self.assertNotIn("evidence_refs", p)

    def test_missing_citation_does_not_claim_verification_and_refutation_blocks_output(self):
        for refute in (False, True):
            class Unverified(ScriptedClient):
                def call(self, state, qs, tag):
                    result = super().call(state, qs, tag)
                    if "support" in qs:
                        result["answers"]["citation"] = answer(qs["citation"]["criteria"], "none")
                        if refute:
                            result["answers"]["support"] = answer(qs["support"]["criteria"], "refuted")
                    return result
            p = method.predict(HISTORY, Unverified())
            self.assertIsNone(p["verified_step"])
            self.assertEqual(p["predicted_step"], None if refute else 1)

    def test_search_call_budget_preserves_final_choice_and_verification(self):
        c = ScriptedClient()
        p = method.predict(HISTORY, c, replace(method.Config(), max_calls=6))
        self.assertEqual(p["stop_reason"], "reserved_final_calls")
        self.assertEqual(len(c.calls), 6)
        self.assertEqual([x["tag"].split('_', 1)[1] for x in c.calls][-2:], ["final", "verify"])
        self.assertEqual(p["predicted_step"], 1)
        self.assertEqual(p["decision_status"], "best_effort")

    def test_actual_usage_adapts_token_estimate_and_search_yields_to_final(self):
        class Costly(ScriptedClient):
            def call(self, state, qs, tag):
                result = super().call(state, qs, tag)
                result["usage"]["input_tokens"] = 4000
                return result
        c = Costly()
        p = method.predict(HISTORY, c, replace(method.Config(), max_input_tokens=45000))
        self.assertEqual(p["stop_reason"], "reserved_final_tokens")
        self.assertEqual([x["tag"].split('_', 1)[1] for x in c.calls][-2:], ["final", "verify"])
        self.assertLessEqual(p["budget"]["input_tokens"], 45000)
        self.assertIsNotNone(p["predicted_step"])

    def test_byte_reserve_and_tiny_budget_never_invent_output(self):
        class AvailableRoot(ScriptedClient):
            def call(self, state, qs, tag):
                result = super().call(state, qs, tag)
                if "root_step" in qs:
                    selected = next(k for k in qs["root_step"]["criteria"] if k.isdigit())
                    result["answers"]["root_step"] = answer(qs["root_step"]["criteria"], selected)
                return result
        c = AvailableRoot()
        p = method.predict(HISTORY, c, replace(method.Config(), max_request_bytes=52000))
        self.assertEqual(p["stop_reason"], "reserved_final_bytes")
        self.assertEqual([x["tag"].split('_', 1)[1] for x in c.calls][-2:], ["final", "verify"])
        self.assertLessEqual(p["budget"]["request_bytes"], 52000)
        c = ScriptedClient()
        p = method.predict(HISTORY, c, replace(method.Config(), max_calls=1))
        self.assertEqual(c.calls, [])
        self.assertIsNone(p["predicted_step"])

    def test_ambiguous_relations_finish_one_candidate_before_opening_another(self):
        class Ambiguous(ScriptedClient):
            def call(self, state, qs, tag):
                result = super().call(state, qs, tag)
                if "relation" in qs:
                    result["answers"]["relation"] = answer(qs["relation"]["criteria"], "unknown")
                return result
        p = method.predict(HISTORY, Ambiguous(), replace(method.Config(), max_rounds=8))
        trace = [t for t in p["search_trace"] if "round" in t]
        self.assertEqual([t["action"] for t in trace[:4]], ["inspect", "upstream", "repair", "challenge"])
        self.assertEqual({t["step"] for t in trace[:4]}, {2})
        self.assertIsNone(p["verified_step"])
        node = next(n for n in p["search_nodes"] if n["step"] == 2)
        self.assertEqual(node["check_coverage"]["repair"]["relation"], "unknown")

    def test_long_trajectory_needs_no_exhaustive_scan_to_reach_finalization(self):
        history = copy.deepcopy(HISTORY)
        history += [{"step": i, "name": "Worker", "content": "unrelated status " * 200} for i in range(5, 100)]
        c = ScriptedClient()
        p = method.predict(history, c)
        self.assertLess(p["scanned_segments"], p["segments"])
        self.assertFalse(p["coverage"]["full_scan"])
        self.assertLessEqual(len(c.calls), 13)
        self.assertEqual(p["predicted_step"], 1)
        self.assertTrue(p["coverage"]["pending_regions"])

    def test_final_compact_cards_retain_candidates_without_nonshrinking_tournament(self):
        history = [{"step": i, "name": "Worker", "content": "x" * 1900} for i in range(30)]
        c = ScriptedClient()
        pipeline = method.Pipeline(history, c)
        for ref in pipeline.index.spans:
            pipeline.add_node(ref)
            pipeline.nodes[pipeline.index.spans[ref]["step"]]["done"].append("inspect")
        result = pipeline.finalize()
        self.assertEqual(result["predicted_step"], 1)
        self.assertEqual(len(c.calls[0]["request"]["state"]["candidates"]), 30)
        self.assertTrue(all(len(ev.dumps(x["request"]).encode()) <= method.Config().request_bytes for x in c.calls))
        self.assertFalse(any(x["tag"].endswith("_reduce") for x in c.calls))

    def test_gold_fields_never_enter_requests_and_exact_replay_is_stable(self):
        history = copy.deepcopy(HISTORY)
        for h in history:
            h.update(mistake_reason="SECRET_GOLD", mistake_step=999)
        c = ScriptedClient()
        expected = method.predict(history, c)
        self.assertNotIn("SECRET_GOLD", ev.dumps(c.calls))
        class Replay:
            def __init__(self, test):
                self.calls, self.test = iter(c.calls), test
            def call(self, state, qs, tag):
                saved = next(self.calls)
                self.test.assertEqual(tag, saved["tag"])
                self.test.assertEqual(ev.sha(ev.dumps({"model": ev.MODEL, "state": state, "questions": qs}).encode()),
                                      saved["request_sha256"])
                return saved["response"]
        self.assertEqual(method.predict(history, Replay(self)), expected)


class RunnerTests(unittest.TestCase):
    def test_demo_is_offline_and_does_not_read_key(self):
        with tempfile.TemporaryDirectory() as out, patch.object(runner.common, "get_key") as key, \
             patch.object(runner, "Client") as client, patch.object(sys, "argv", ["runner", "--demo", "--output", out]), \
             patch("sys.stdout", new_callable=io.StringIO):
            runner.main()
            key.assert_not_called()
            client.assert_not_called()
            p = json.loads((Path(out) / "demo.json").read_text())
            self.assertEqual(p["api_calls"], 0)
            self.assertTrue(p["synthetic"])

    def test_stop_marker_prevents_resume(self):
        with tempfile.TemporaryDirectory() as out:
            (Path(out) / "user_stop.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "Stopped run"):
                runner.prepare(Path(out), {}, method.Config(), [], "live", None)

    def test_all_abstention_smoke_gate_prevents_full_batch(self):
        rows = [{"question_ID": s + str(i), "source": s, "history": HISTORY,
                 "mistake_step": 1, "mistake_agent": "Planner"} for s in "abcde" for i in range(2)]
        p = {"predicted_step": None, "predicted_role": None, "verified_step": None, "verified_role": None,
             "method": method.PROTOCOL, "decision_status": "abstained", "budget": {"input_tokens": 0, "output_tokens": 0}}
        with tempfile.TemporaryDirectory() as out, \
             patch.object(runner.ev, "load_data", return_value=(rows, {"revision": "fixture"})), \
             patch.object(runner.common, "get_key", return_value="fixture"), patch.object(runner.common, "preflight"), \
             patch.object(runner, "Client"), patch.object(runner.method, "predict", side_effect=lambda *a: copy.deepcopy(p)) as predict, \
             patch.object(sys, "argv", ["runner", "--live", "--mini", "--output", out]), \
             patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                runner.main()
            self.assertEqual(predict.call_count, 5)
            result = json.loads((Path(out) / "progress.json").read_text())
            self.assertEqual(result["status"], "stopped_no_usable_predictions")
            self.assertEqual(result["requested_n"], 10)


if __name__ == "__main__":
    unittest.main()
