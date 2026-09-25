"""Literature-guided RCA: global diverse retrieval, feedback questions and fresh verification.

RRF/MMR, IRCoT and Decomposed Prompting inspire the retrieval/control flow.
CoVe inspires separating verification context from previous judgments. This is
an untrained Choice adaptation, not a reproduction of these papers' results.
"""
from collections import deque
from dataclasses import dataclass, fields

import evaluate as ev
import jev_rcta_adaptive as v1
import jev_rcta_adaptive_v2 as v2
from rcta_retrieval import Retrieval

PROTOCOL = "jev-rcta-adaptive-v3.1"
NEEDS = {"upstream": "Earlier requirements, instructions or a handoff",
         "local": "Actual action, parameters or local implementation",
         "repair": "Later correction or validation outcome", "challenge": "Evidence for a competing explanation",
         "none": "Current evidence suffices for a provisional comparison", "unknown": "Missing evidence is unclear"}
QUERY_TERMS = {
    "upstream": "requirement instruction requested constraint must should plan delegate handoff 要求 指令 约束",
    "local": "implementation edit write execute parameters arguments change 实现 参数 执行",
    "repair": "repair fixed test validation persisted failed 修复 验证 失败",
    "challenge": "contradiction expected actual comparison mismatch alternative conflict 冲突 对比",
}


@dataclass(frozen=True)
class Config(v2.Config):
    max_hops: int = 4
    retrieval_view_bytes: int = 9000
    rrf_k: int = 60
    mmr_percent: int = 70
    global_retrieval: int = 1
    iterative_queries: int = 1
    factored_verification: int = 1

    def validate(self):
        v2.Config(**{f.name: getattr(self, f.name) for f in fields(v2.Config)}).validate()
        for name in ("max_hops", "retrieval_view_bytes", "rrf_k", "mmr_percent"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("Invalid v3 configuration: " + name)
        if self.mmr_percent > 100:
            raise ValueError("MMR percentage must be <= 100")
        for name in ("global_retrieval", "iterative_queries", "factored_verification"):
            if type(getattr(self, name)) is not int or getattr(self, name) not in (0, 1):
                raise ValueError("Ablations must be 0 or 1: " + name)


class Pipeline(v2.Pipeline):
    def __init__(self, history, client, config=None, calibration=None):
        super().__init__(history, client, config or Config(), calibration)
        self.retriever = Retrieval(self.index, self.config.rrf_k, self.config.mmr_percent / 100)
        self.anchors, self.retrieval_trace, self.pending_causes = [], [], deque()
        self.presented_steps, self.opened_segments = set(), set()
        self.need = "upstream"
        self.seed_answers = None

    def remember(self, ref):
        if ref in self.index.spans and ref not in self.anchors:
            self.anchors.append(ref)

    def queries(self, need=None):
        texts = [self.index.history[0]["content"], self.index.history[-1]["content"]]
        if self.config.iterative_queries:
            texts += [QUERY_TERMS.get(need or self.need, QUERY_TERMS["challenge"])]
            texts += [self.retriever.text[r] for r in self.anchors[-2:]]
        return texts

    def global_view(self, kind):
        packet = self.retriever.packet(self.queries(), self.config.retrieval_view_bytes,
                                       excluded_steps=self.attempted if kind == "lead" else ())
        self.presented_steps.update(r["step"] for r in packet["records"])
        self.retrieval_trace.append({"kind": kind, "query_anchor_refs": self.anchors[-2:], "need": self.need,
                                     "view_refs": [r["source_ref"] for r in packet["records"]],
                                     "ranked_spans": packet["ranked_spans"], "query_count": packet["query_count"],
                                     "strategy": packet["retrieval_strategy"], "view_complete": packet["view_complete"]})
        return packet

    def seed(self):
        packet = self.global_view("seed")
        opts = {r["source_ref"]: "Open original step %d at this source excerpt" % r["step"] for r in packet["records"]}
        if not opts:
            return
        qs = {
            "symptom": v1.question("Which supplied ORIGINAL excerpt best describes an observable task failure, "
                "requirement/action conflict or incorrect result? This is an evidence anchor, not yet root attribution.",
                dict(opts, unknown="No observable failure can be identified from these excerpts")),
            "need": v1.question("Which evidence type would best distinguish a bad original instruction, "
                "a deviation during execution, and a mistaken interpretation of results?", NEEDS),
        }
        answers = self.session.ask(dict(self.shared, evidence_directory=packet), qs, "seed")
        self.seed_answers = answers
        self.need = answers["need"]["choice"]
        chosen = answers["symptom"]["choice"]
        if chosen in opts:
            self.remember(chosen)
            self.shared["observed_anchor"] = self.index.packet([chosen], 3000)

    def pick_lead(self):
        while self.pending_causes:
            ref = self.pending_causes.popleft()
            if self.index.spans[ref]["step"] not in self.attempted:
                self.add_node(ref, source="upstream_hypothesis")
                return self.index.spans[ref]["step"]
        packet = self.global_view("lead")
        opts = {r["source_ref"]: "Inspect original step %d containing this excerpt" % r["step"] for r in packet["records"]}
        if not opts:
            return None
        qs = {"location": v1.question("Choose the most useful candidate to inspect next for the FIRST decisive "
                "unrepaired error. Distinguish an original task requirement from an agent introducing a bad decision, "
                "and an error's origin from later reports of it. Other unshown regions remain possible.", dict(opts, **v1.UNKNOWN))}
        answer = self.session.ask(dict(self.shared, retrieved_evidence=packet), qs, "lead")["location"]
        ordered, selected = self.ordered_options(answer, opts, "location")
        self.recalled.update(self.index.spans[r]["step"] for r in selected if r in opts)
        chosen = answer["choice"] if answer["choice"] in opts else ordered[0]
        if answer["probabilities"][chosen] <= 0:
            return None
        self.trace.append({"action": "global_lead", "answer": answer, "mass_set": selected,
                           "chosen_ref": chosen, "uncertain_probe": answer["choice"] not in opts})
        self.add_node(chosen, source="global_retrieval")
        self.remember(chosen)
        return self.index.spans[chosen]["step"]

    def node_packet(self, node, action):
        if not self.config.global_retrieval:
            return super().node_packet(node, action)
        step = node["step"]
        own = node["refs"] + self.index.by_step[step]
        if action == "expand":
            own = [r for r in self.index.by_step[step] if r not in node["seen_local"]] + own
        own_packet = self.index.packet(own, self.config.evidence_bytes // 2)
        seen = set(node["seen"].get(action, []))
        eligible = [r for r, s in self.index.spans.items() if s["step"] != step and r not in seen
                    and (action != "upstream" or s["step"] < step)
                    and (action != "repair" or s["step"] > step)]
        target = self.retriever.text[node["refs"][0]]
        queries = [self.index.history[0]["content"], target]
        if self.config.iterative_queries:
            queries += [QUERY_TERMS.get(action, QUERY_TERMS["local"])]
            queries += [self.retriever.text[r] for r in self.anchors[-1:]]
        ranked = self.retriever.ranked(queries, eligible)
        extra = self.index.packet(ranked, self.config.evidence_bytes // 2)
        records = list({r["id"]: r for r in own_packet["records"] + extra["records"]}.values())
        return {"records": records, "new_external_refs": [r["id"] for r in extra["records"]],
                "external_remaining": extra["remaining_spans"],
                "local_remaining": len(set(self.index.by_step[step]) - set(node["seen_local"]) -
                                       {r["id"] for r in own_packet["records"]})}

    def action_questions(self, action, records):
        qs = super().action_questions(action, records)
        if action in ("inspect", "expand"):
            qs["need"] = v1.question("Which missing evidence most distinguishes bad requirements, an execution "
                "deviation, a repaired mistake, and an incorrect interpretation for the TARGET?", NEEDS)
        else:
            qs["next_need"] = v1.question("Based on the evidence shown and checks already recorded, which "
                "other evidence type is most useful for resolving this TARGET's responsibility?", NEEDS)
        return qs

    def do_action(self, node, action):
        answers = super().do_action(node, action)
        for ref in self.inspected_refs:
            for segment in self.index.segments:
                if ref in segment["refs"]:
                    self.opened_segments.add(segment["id"])
                    break
        citation = answers["citation"]["choice"]
        if citation in self.index.spans:
            self.remember(citation)
        if "next_need" in answers:
            node["labels"]["need"] = answers["next_need"]
        self.need = node["labels"]["need"]["choice"]
        if action == "upstream" and citation in self.index.spans and self.index.spans[citation]["step"] < node["step"]:
            relation = answers["relation"]
            possible = "upstream" in self.policy.select(relation, "upstream_relation")
            if possible and relation["probabilities"]["upstream"] >= relation["probabilities"]["local"]:
                self.pending_causes.append(citation)
                self.trace.append({"action": "queue_upstream_hypothesis", "from": node["step"],
                                   "evidence_ref": citation, "verified_causal_edge": False})
        return answers

    def search(self):
        if not self.config.global_retrieval:
            return super().search()
        self.seed()
        for hop in range(self.config.max_hops):
            step = self.pick_lead()
            if step is None:
                self.stop_reason = "no_new_retrieval_lead"
                return
            self.attempted.add(step)
            self.focus_order.append(step)
            node = self.nodes[step]
            self.do_action(node, "inspect")
            for _ in range(2):
                if node["status"] == "sidelined":
                    break
                done = set(node.get("attempted_checks", []))
                hint = node["labels"]["need"]["choice"]
                if hint == "local" and set(self.index.by_step[step]) - set(node["seen_local"]):
                    action = "expand"
                elif hint in ("upstream", "repair", "challenge") and hint not in done:
                    action = hint
                else:
                    action = next((a for a in ("upstream", "repair", "challenge") if a not in done), None)
                if action is None:
                    break
                self.do_action(node, action)
                if self.pending_causes:
                    break
            self.trace.append({"action": "retrieval_feedback", "hop": hop, "need": self.need,
                               "source_refs": self.anchors[-2:]})
        self.stop_reason = "hop_budget"

    def finalize(self):
        if not self.config.factored_verification:
            return super().finalize()
        ids = sorted(s for s, n in self.nodes.items() if n["status"] == "open" and "inspect" in n["done"])
        self.final_candidates = ids
        if not ids:
            return {"decision_status": "abstained", "decision_reason": "no_inspected_viable_candidate"}
        qs = {"root_step": v1.question("Choose the best point prediction for the earliest decisive unrepaired "
            "error among inspected candidates. Original requirements are context, not automatically a mistake. "
            "Use outside if the root is missing and unknown if no candidate has relevant evidence.",
            dict({str(s): "Original step %d" % s for s in ids}, **v1.UNKNOWN))}
        for width in (1800, 900, 300, 80):
            state = dict(self.shared, candidates=[self.final_card(s, width) for s in ids], coverage=self.coverage())
            if self.session.size(state, qs) <= min(self.config.request_bytes, self.config.final_request_bytes):
                break
        root = self.session.ask(state, qs, "final")["root_step"]
        result = {"decision_status": "abstained", "decision_reason": "no_final_candidate",
                  "final_answers": {"root_step": root}, "final_set": self.policy.select(root, "root_step")}
        if root["choice"] in v1.UNKNOWN:
            return result
        step, role = int(root["choice"]), None
        node = self.nodes[step]
        # Fresh original evidence and a neutral target identifier. No prior
        # labels, probabilities, draft role or drafted causal explanation.
        refs = node["refs"][:1]
        for kind in ("upstream", "repair", "challenge"):
            packet = self.node_packet(dict(node, seen={}), kind)
            refs += packet["new_external_refs"][:1]
        packet = self.index.packet(list(dict.fromkeys(refs)), self.config.evidence_bytes)
        fresh_state = {"target_step": step, "original_evidence": packet,
                       "task_and_outcome": self.shared["task_and_outcome"],
                       "notice": "Assess only original evidence. Unshown context remains unknown."}
        result.update(predicted_step=step, predicted_role=None, tentative_step=step, tentative_role=None,
                      decision_status="best_effort", decision_reason="point_prediction_unverified",
                      candidate_evidence_refs=[self.index.spans[r["id"]] for r in packet["records"] if r["step"] == step])
        cites = dict(v1.pointers(packet["records"]), none="No direct supporting fragment", unknown="Insufficient evidence")
        try:
            check = self.session.ask(fresh_state, {
                "origin": v1.question("Compare the TARGET action/decision to earlier requirements and instructions. "
                    "Which account is supported by the shown original text?", {
                        "local": "TARGET introduces the relevant violation rather than following an earlier wrong decision",
                        "upstream": "An earlier decision introduces the violation that TARGET follows",
                        "no_violation": "TARGET does not violate the relevant requirement", "unknown": "Cannot distinguish"}),
                "outcome": v1.question("What does later original evidence establish about the SAME alleged error?", {
                    "persists": "It remains and contributes to the final failure", "repaired": "Its successful repair is verified",
                    "unlinked": "The observed failure is unrelated", "unknown": "No decisive outcome evidence"}),
                "citation": v1.question("Which fragment directly establishes the TARGET's erroneous decision itself?", cites),
                "responsible_role": v1.question("For this TARGET attribution, which recorded role made the relevant "
                    "erroneous decision? Follow original authorship/delegation evidence; do not automatically blame "
                    "a tool or a reporter of the error.", dict({"r%d" % i: name for i, name in enumerate(self.names)},
                                                              unknown="Responsibility is unresolved")),
                "responsibility_citation": v1.question("Which original fragment establishes who made the relevant decision?", cites),
            }, "verify")
        except v1.BudgetStop as error:
            result.update(decision_reason="verification_budget_exhausted", verification_stop=str(error))
            return result
        result["verification"] = check
        role_answer = check["responsible_role"]
        if role_answer["choice"] != "unknown":
            role = self.names[int(role_answer["choice"][1:])]
        result.update(predicted_role=role, tentative_role=role)
        if (self.confident(check["origin"], "no_violation") or self.confident(check["outcome"], "repaired")
                or self.confident(check["outcome"], "unlinked")):
            result.update(predicted_step=None, predicted_role=None, decision_status="abstained",
                          decision_reason="independent_evidence_refuted")
            return result
        citation = check["citation"]["choice"]
        direct = {r["id"] for r in packet["records"] if r["step"] == step}
        if (self.confident(root, root["choice"]) and self.confident(check["origin"], "local")
                and self.confident(check["outcome"], "persists") and citation in direct):
            role_cited = check["responsibility_citation"]["choice"] in {r["id"] for r in packet["records"]}
            result.update(decision_status="supported", decision_reason="fresh_origin_and_outcome_checked",
                          verified_step=step, verified_role=role if role is not None and role_cited
                          and self.confident(role_answer, role_answer["choice"]) else None,
                          evidence_refs=[self.index.spans[citation]])
        return result

    def coverage(self):
        result = super().coverage()
        if hasattr(self, "retrieval_trace"):
            result.update(global_retrieval_rounds=len(self.retrieval_trace),
                          excerpt_steps_presented=len(self.presented_steps),
                          action_segments_read=len(self.opened_segments))
        return result

    def run(self):
        result = super().run()
        result.update(method=PROTOCOL, retrieval_trace=self.retrieval_trace, query_anchor_refs=self.anchors,
                      seed_answers=self.seed_answers, presented_steps=sorted(self.presented_steps),
                      research_adaptation="RRF/MMR + structured IRCoT + conditional role attribution + fresh verification")
        return result


def predict(history, client, config=None, calibration=None):
    return Pipeline(history, client, config, calibration).run()
