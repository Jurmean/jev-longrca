import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate as ev
import evaluate_jev_rcta_adaptive as runner
import jev_rcta_adaptive as method
import jev_rcta_client as transport
from rcta_choice_policy import ChoicePolicy, adaptive_set, distribution, fit_calibration
from rcta_demo import HISTORY, ScriptedClient
from rcta_evidence import EvidenceIndex


def certain(options, choice):
    return {"choice": choice, "confidence": 0.99,
            "probabilities": {key: float(key == choice) for key in options}}


class PolicyTests(unittest.TestCase):
    def test_candidate_count_adapts_and_preserves_ties(self):
        peaked = {"choice": "a", "probabilities": {"a": .94, "b": .03, "c": .02, "d": .01}}
        flat = {"choice": "a", "probabilities": dict.fromkeys("abcd", .25)}
        self.assertEqual(adaptive_set(peaked), ["a"])
        self.assertEqual(adaptive_set(flat), list("abcd"))
        self.assertEqual(adaptive_set({"probabilities": {"a": .6, "b": .2, "c": .2}}, .7), list("abc"))
        self.assertEqual(adaptive_set({"probabilities": {"a": 1, "b": 0}}, 1), ["a", "b"])

    def test_unknown_mass_is_not_renormalized_away(self):
        answer = {"probabilities": {"unknown": .95, "s0": .03, "s1": .02}}
        self.assertEqual(adaptive_set(answer), ["unknown"])

    def test_probability_mismatch_is_visible_and_never_forces_reported_choice(self):
        answer = distribution({"choice": "a", "confidence": .4, "probabilities": {"a": .1, "b": .9}}, {"a": "", "b": ""})
        self.assertEqual(answer["choice"], "b")
        self.assertEqual(answer["reported_choice"], "a")
        self.assertTrue(answer["warnings"])
        for probs in ({"a": 1}, {"a": 0, "b": 0}, {"a": float("nan"), "b": .5},
                      {"a": -1, "b": 2}, {"a": True, "b": 0}):
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                distribution({"choice": "a", "probabilities": probs}, {"a": "", "b": ""})

    def test_calibration_is_typed_and_rejects_trajectory_leakage(self):
        records = [{"case_id": "dev-%d" % i, "kind": "location", "truth": "a",
                    "answer": {"choice": "a", "probabilities": {"a": .8, "b": .2}}} for i in range(20)]
        result = fit_calibration(records)
        self.assertAlmostEqual(result["question_types"]["location"]["mass"], .8)
        policy = ChoicePolicy(calibration=result)
        policy.check_holdout(["test-1"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            policy.check_holdout(["dev-0"])
        with self.assertRaisesRegex(ValueError, "one question"):
            fit_calibration(records + [records[0]])
        self.assertEqual(fit_calibration(records[:1])["question_types"]["location"]["mass"], 1)


class EvidenceTests(unittest.TestCase):
    def test_utf8_control_characters_and_exact_citations(self):
        history = [{"step": 0, "name": "Worker", "content": '汉🙂\x00"\\\n' * 1200,
                    "mistake_reason": "SECRET_GOLD"}]
        index = EvidenceIndex(history)
        self.assertEqual(index.audit()["primary_text_coverage"], "lossless")
        for ref in index.spans:
            r = index.read(ref)
            self.assertEqual(r["content"].encode(), history[0]["content"].encode()[r["start_byte"]:r["end_byte"]])
            self.assertEqual(ev.sha(r["content"].encode()), r["sha256"])
            self.assertLess(len(ev.dumps(r).encode()), 5000)
            self.assertNotIn("SECRET_GOLD", ev.dumps(r))
        middle = list(index.spans)[len(index.spans) // 2]
        self.assertEqual(index.packet([middle], 5000)["records"][0]["id"], middle)

    def test_retrieval_finds_nonadjacent_handoff_and_late_repair(self):
        history = [{"step": i, "name": "Other", "content": "unrelated status"} for i in range(30)]
        history[0].update(name="Planner (-> Worker)", content="orders.csv drop date")
        history[12].update(name="Worker", content="orders.csv drop date executed")
        history[29].update(name="Reviewer", content="orders.csv date repaired and validation passed")
        index = EvidenceIndex(history)
        self.assertEqual(index.read(index.retrieve(12, "upstream")[0])["step"], 0)
        self.assertEqual(index.read(index.retrieve(12, "repair")[0])["step"], 29)
        first = index.retrieve(12, "repair")[0]
        self.assertNotIn(first, index.retrieve(12, "repair", seen=[first]))


class PipelineTests(unittest.TestCase):
    def test_multiround_upstream_backtrack_and_cited_result(self):
        client = ScriptedClient()
        result = method.predict(HISTORY, client)
        self.assertEqual(result["predicted_step"], 1)
        self.assertEqual(result["predicted_role"], "Planner")
        self.assertEqual(result["recall_candidates"], [2])
        self.assertIn(1, result["expanded_candidates"])
        self.assertTrue(any(e["action"] == "backtrack" and e["step"] == 2 for e in result["search_trace"]))
        self.assertTrue(any(e["from"] == 2 and e["to"] == 1 for e in result["relation_hypotheses"]))
        self.assertEqual(result["evidence_refs"][0]["step"], 1)
        self.assertEqual(client.calls[-1]["request"]["state"]["proposed_root_step"], 1)
        self.assertTrue(all(len(ev.dumps(c["request"]).encode()) <= method.Config().request_bytes for c in client.calls))

    def test_only_history_allowlist_enters_inference_and_role_is_independent(self):
        history = copy.deepcopy(HISTORY)
        for h in history:
            h.update(mistake_step=99, mistake_reason="SECRET_GOLD", mistake_agent="SECRET_GOLD")
        before = copy.deepcopy(history)
        class IndependentRole(ScriptedClient):
            def call(self, state, qs, tag):
                response = super().call(state, qs, tag)
                if "responsible_role" in qs:
                    opts = qs["responsible_role"]["criteria"]
                    key = next(k for k, v in opts.items() if v == "Reviewer")
                    response["answers"]["responsible_role"] = certain(opts, key)
                return response
        client = IndependentRole()
        result = method.predict(history, client)
        self.assertEqual(result["predicted_step"], 1)
        self.assertEqual(result["predicted_role"], "Reviewer")
        self.assertNotIn("SECRET_GOLD", ev.dumps(client.calls))
        self.assertEqual(history, before)

    def test_flat_final_distribution_abstains(self):
        class Uncertain(ScriptedClient):
            def call(self, state, qs, tag):
                response = super().call(state, qs, tag)
                if "root_step" in qs:
                    opts = qs["root_step"]["criteria"]
                    response["answers"]["root_step"] = {"choice": next(iter(opts)), "confidence": 0,
                                                          "probabilities": dict.fromkeys(opts, 1 / len(opts))}
                return response
        result = method.predict(HISTORY, Uncertain())
        self.assertIsNone(result["predicted_step"])
        self.assertEqual(result["decision_status"], "abstained")

    def test_no_direct_citation_cannot_be_reported_as_supported(self):
        class Uncited(ScriptedClient):
            def call(self, state, qs, tag):
                response = super().call(state, qs, tag)
                if tag.endswith("_verify"):
                    response["answers"]["citation"] = certain(qs["citation"]["criteria"], "none")
                return response
        result = method.predict(HISTORY, Uncited())
        self.assertIsNone(result["predicted_step"])
        self.assertNotIn("evidence_refs", result)

    def test_uncertain_local_evidence_triggers_fresh_span_expansion(self):
        history = [{"step": 0, "name": "User", "content": "Check the artifact."},
                   {"step": 1, "name": "Worker", "content": "unresolved data " * 1600 + "NEEDLE"}]
        class NeedLocal(ScriptedClient):
            def call(self, state, qs, tag):
                response = super().call(state, qs, tag)
                if "assessment" in qs:
                    response["answers"]["assessment"] = certain(qs["assessment"]["criteria"], "unknown")
                    response["answers"]["need"] = certain(qs["need"]["criteria"], "local")
                return response
        pipeline = method.Pipeline(history, NeedLocal())
        pipeline.add_node(pipeline.index.by_step[1][0])
        actions, found = [], False
        for _ in range(20):
            _, neg_step, action, state, questions, packet = pipeline.choose_action()
            actions.append(action)
            answers = pipeline.session.ask(state, questions, action)
            pipeline.apply(pipeline.nodes[-neg_step], action, packet, answers)
            if any("NEEDLE" in r["content"] and r["step"] == 1 for r in packet["records"]):
                found = True
                break
        self.assertTrue(found)
        self.assertIn("expand", actions)

    def test_repaired_candidate_is_sidelined_and_cannot_be_final(self):
        class Repaired(ScriptedClient):
            def call(self, state, qs, tag):
                response = super().call(state, qs, tag)
                if tag.endswith("_repair"):
                    response["answers"]["relation"] = certain(qs["relation"]["criteria"], "repaired")
                return response
        result = method.predict(HISTORY, Repaired())
        self.assertIsNone(result["predicted_step"])
        self.assertTrue(any(e["action"] == "backtrack" and e["cause"] == "repair" for e in result["search_trace"]))

    def test_deferred_segments_can_be_reopened(self):
        class Deferred(ScriptedClient):
            def call(self, state, qs, tag):
                response = super().call(state, qs, tag)
                if "relevance" in qs:
                    response["answers"]["relevance"] = certain(qs["relevance"]["criteria"], "irrelevant")
                return response
        result = method.predict(HISTORY, Deferred(), replace(method.Config(), segment_records=1))
        self.assertTrue(any(e["action"] == "reopen_segment" for e in result["search_trace"]))
        self.assertEqual(result["predicted_step"], 1)

    def test_budget_stop_has_no_fabricated_answer_or_extra_call(self):
        for config in (replace(method.Config(), max_calls=1), replace(method.Config(), max_request_bytes=1),
                       replace(method.Config(), request_bytes=100)):
            client = ScriptedClient()
            result = method.predict(HISTORY, client, config)
            self.assertIsNone(result["predicted_step"])
            self.assertEqual(result["decision_status"], "abstained")
            self.assertEqual(len(client.calls), 0)
        class Expensive(ScriptedClient):
            def call(self, state, qs, tag):
                response = super().call(state, qs, tag)
                response["usage"]["input_tokens"] = 1000
                return response
        client = Expensive()
        result = method.predict(HISTORY, client, replace(method.Config(), max_input_tokens=500))
        self.assertEqual(len(client.calls), 1)
        self.assertIsNone(result["predicted_step"])

    def test_unresolved_context_tournament_never_forces_top_k(self):
        history = [{"step": i, "name": "Worker", "content": "x" * 1900} for i in range(30)]
        class Flat:
            def call(self, state, qs, tag):
                return {"answers": {key: {"choice": next(iter(q["criteria"])),
                        "probabilities": dict.fromkeys(q["criteria"], 1 / len(q["criteria"]))} for key, q in qs.items()}}
        pipeline = method.Pipeline(history, Flat(), replace(method.Config(), request_bytes=22000))
        for ref in pipeline.index.spans:
            pipeline.add_node(ref)
        result = pipeline.finalize()
        self.assertEqual(result["decision_reason"], "final_context_ambiguity")
        self.assertEqual(len(pipeline.final_candidates), 30)

    def test_exact_offline_replay_is_deterministic(self):
        client = ScriptedClient()
        expected = method.predict(HISTORY, client)
        class Replay:
            def __init__(self, test):
                self.records = iter(client.calls)
                self.test = test
            def call(self, state, qs, tag):
                record = next(self.records)
                self.test.assertEqual(tag, record["tag"])
                payload = {"model": ev.MODEL, "state": state, "questions": qs}
                self.test.assertEqual(ev.sha(ev.dumps(payload).encode()), record["request_sha256"])
                return record["response"]
        self.assertEqual(method.predict(HISTORY, Replay(self)), expected)

    def test_transport_quota_stop_prevents_any_further_network_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            client = transport.Client("fixture-private", out / "calls", out)
            error = urllib.error.HTTPError("https://example.invalid", 402, "quota", {}, io.BytesIO(b"insufficient_balance"))
            with patch.object(transport.urllib.request, "urlopen", side_effect=error) as network:
                for _ in range(2):
                    with self.assertRaises(transport.BalanceExhausted):
                        method.predict(HISTORY, client)
                self.assertEqual(network.call_count, 1)
            self.assertTrue((out / "balance_stop.json").exists())


class RunnerTests(unittest.TestCase):
    def test_output_protocol_and_mode_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            runner.prepare(out, {"revision": "test"}, method.Config(), ["sample"], "demo")
            for mode, config in (("live", method.Config()), ("demo", replace(method.Config(), max_calls=5))):
                with self.assertRaisesRegex(ValueError, "another protocol"):
                    runner.prepare(out, {"revision": "test"}, config, ["sample"], mode)

    def test_offline_modes_never_read_key_or_call_transport(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, "get_key") as key, \
             patch.object(runner, "Client") as client, patch.object(sys, "argv", [
                "runner", "--demo", "--output", temp]), patch("sys.stdout", new_callable=io.StringIO):
            runner.main()
            key.assert_not_called()
            client.assert_not_called()
            result = json.loads((Path(temp) / "demo.json").read_text())
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["api_calls"], 0)

    def test_live_entry_preflight_exact_cache_resume_and_scoring(self):
        row = {"question_ID": "fixture-1", "source": "fixture", "history": HISTORY,
               "mistake_step": 1, "mistake_agent": "Planner", "mistake_reason": "SECRET_GOLD"}
        fixture = ScriptedClient()
        def respond(request, timeout):
            payload = json.loads(request.data)
            qs = payload["questions"]
            if "check" in qs:
                result = {"model": ev.MODEL, "answers": {"check": certain(qs["check"]["criteria"], "seven")}}
            else:
                if "event" in qs:
                    phase = "scan"
                elif "assessment" in qs:
                    phase = "inspect"
                elif "root_step" in qs:
                    phase = "final"
                elif "support" in qs:
                    phase = "verify"
                else:
                    opts = qs["relation"]["criteria"]
                    phase = "upstream" if "upstream" in opts else "repair" if "repaired" in opts else "challenge"
                result = fixture.call(payload["state"], qs, "fixture_" + phase)
            return io.BytesIO(json.dumps(result).encode())
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(runner.ev, "load_data", return_value=([row], {"revision": "fixture"})), \
             patch.object(runner, "get_key", return_value="fixture-private"), \
             patch.object(sys, "argv", ["runner", "--live", "--case-id", "fixture-1", "--output", temp]), \
             patch.object(transport.urllib.request, "urlopen", side_effect=respond) as network, \
             patch("sys.stdout", new_callable=io.StringIO):
            runner.main()
            calls = network.call_count
            self.assertGreater(calls, 1)
            runner.main()
            self.assertEqual(network.call_count, calls)
            summary = json.loads((Path(temp) / "summary.json").read_text())
            self.assertEqual(summary["overall"]["step_exact"], 1)
            self.assertEqual(summary["abstentions"], 0)
            self.assertNotIn("estimated_input_cost_usd", summary)
            self.assertNotIn("SECRET_GOLD", ev.dumps(fixture.calls))

    def test_failed_live_preflight_stops_before_trajectory(self):
        row = {"question_ID": "fixture-1", "source": "fixture", "history": HISTORY}
        bad = {"model": ev.MODEL, "answers": {"check": certain({"three": "", "seven": ""}, "three")}}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(runner.ev, "load_data", return_value=([row], {"revision": "fixture"})), \
             patch.object(runner, "get_key", return_value="fixture-private"), \
             patch.object(sys, "argv", ["runner", "--live", "--case-id", "fixture-1", "--output", temp]), \
             patch.object(transport.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(bad).encode())) as network, \
             patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                runner.main()
            self.assertEqual(network.call_count, 1)
            progress = json.loads((Path(temp) / "progress.json").read_text())
            self.assertEqual(progress["failed_id"], "_preflight")
            self.assertEqual(progress["completed_ids"], [])


if __name__ == "__main__":
    unittest.main()
