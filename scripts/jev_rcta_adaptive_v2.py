"""Budgeted, lazy-routing Adaptive RCTA with candidate-focused evidence checks.

Point predictions and evidence-supported predictions are separate outputs.
Defaults are engineering heuristics, not calibrated probabilities or a claim of
global causal verification. The previous v1 implementation remains reproducible.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math

import evaluate as ev
import jev_rcta_adaptive as v1
from rcta_directory import Directory

PROTOCOL = "jev-rcta-adaptive-v2"


@dataclass(frozen=True)
class Config(v1.Config):
    segment_bytes: int = 12000
    segment_records: int = 24
    evidence_bytes: int = 8000
    request_bytes: int = 26000
    max_calls: int = 24
    max_request_bytes: int = 240000
    max_input_tokens: int = 80000
    max_rounds: int = 12
    directory_fanout: int = 6
    reserve_calls: int = 2
    reserve_request_bytes: int = 40000
    reserve_input_tokens: int = 24000
    final_request_bytes: int = 20000

    def validate(self):
        super().validate()
        if not 2 <= self.directory_fanout <= 12:
            raise ValueError("Directory fanout must be between 2 and 12")
        if self.reserve_calls < 2:
            raise ValueError("Final choice and verification require two reserved calls")
        if self.reserve_request_bytes < 2 * min(self.final_request_bytes, self.request_bytes):
            raise ValueError("Final byte reserve must cover both bounded final requests")


class Session(v1.Session):
    def estimate_tokens(self, size):
        # The service reports usage after the response. This is a conservative
        # adaptive estimate, not a tokenizer or a hard billing guarantee.
        ratio = max([0.5] + [e["input_tokens"] / max(e["request_bytes"], 1) * 1.25 for e in self.events])
        return math.ceil(size * ratio) + 512

    def ask(self, state, questions, kind):
        size = self.size(state, questions)
        estimate = self.estimate_tokens(size)
        final = kind in ("final", "verify")
        c = self.config
        if final and size > min(c.final_request_bytes, c.request_bytes):
            raise v1.BudgetStop("final_request_context_limit")
        if not final:
            if self.calls + 1 + c.reserve_calls > c.max_calls:
                raise v1.BudgetStop("reserved_final_calls")
            if self.request_bytes + size + c.reserve_request_bytes > c.max_request_bytes:
                raise v1.BudgetStop("reserved_final_bytes")
            if self.input_tokens + estimate + c.reserve_input_tokens > c.max_input_tokens:
                raise v1.BudgetStop("reserved_final_tokens")
        elif self.input_tokens + estimate > c.max_input_tokens:
            raise v1.BudgetStop("estimated_input_token_budget")
        before = self.input_tokens
        answers = super().ask(state, questions, kind)
        self.events[-1].update(input_tokens=self.input_tokens - before, estimated_input_tokens=estimate)
        return answers


class Pipeline(v1.Pipeline):
    def __init__(self, history, client, config=None, calibration=None):
        super().__init__(history, client, config or Config(), calibration)
        self.session = Session(client, self.config)
        refs = self.index.by_step[0][:1] + self.index.by_step[len(history) - 1][-1:]
        self.shared["task_and_outcome"] = self.index.packet(refs, 6000)
        self.directory = Directory(self.index, self.config.directory_fanout)
        self.frontier, self.leads = deque([self.directory.root]), deque()
        self.routed, self.attempted, self.inspected_refs = [], set(), set()
        self.focus_order, self.final_candidates = [], []
        self.rounds = 0

    def ordered_options(self, answer, concrete, kind):
        selected = self.policy.select(answer, kind)
        ranked = sorted(concrete, key=lambda k: (-answer["probabilities"][k], k))
        # Mass sets prioritize siblings of ONE question. Other options remain
        # reachable; uncertain branches are not instantiated as evidence nodes.
        return [k for k in ranked if k in selected] + [k for k in ranked if k not in selected], selected

    def next_segment(self):
        while self.frontier:
            key = self.frontier.popleft()
            node = self.directory.nodes[key]
            if not node["children"]:
                return self.index.segments[node["start_segment"]]
            previews = [self.directory.preview(k) for k in node["children"]]
            questions = {"branch": v1.question(
                "Which region should be read next to distinguish the failure's initiating cause? "
                "These are incomplete directory excerpts. Choose unknown when the view cannot rank regions.",
                dict({p["id"]: "Original steps %d..%d" % tuple(p["step_range"]) for p in previews},
                     unknown="The sampled excerpts do not distinguish these regions"))}
            try:
                answer = self.session.ask(dict(self.shared, directory=previews), questions, "route")["branch"]
            except v1.BudgetStop:
                self.frontier.appendleft(key)
                raise
            ordered, selected = self.ordered_options(answer, node["children"], "branch")
            self.frontier.extendleft(reversed(ordered))
            self.routed.append({"node": key, "answer": answer, "mass_set": selected,
                                "queued_children": ordered, "uncertain": "unknown" in selected})
            self.trace.append({"action": "route", "node": key, "queued_children": ordered,
                               "notice": "Queue priority only; no probability multiplication across questions"})
        return None

    def read_segment(self, segment):
        records = [self.index.read(r) for r in segment["refs"]]
        qs = {
            "event": v1.question("Which activity is most relevant in this segment?", {
                "instruction": "Planning or instruction", "execution": "Execution", "observation": "Observation",
                "repair": "Repair attempt", "mixed": "Mixed or unclear"}),
            "phenomenon": v1.question("Which observable failure-related phenomenon occurs here?", {
                "constraint": "Requirement/action conflict", "execution": "Execution failure",
                "state": "State mismatch", "unsupported": "Unsupported conclusion", "none": "No anomaly",
                "unknown": "Insufficient evidence"}),
            "location": v1.question("Which original fragment is the best lead for tracing the decisive error? "
                "An observed failure may be an effect; it need not be the root.", dict(v1.pointers(records), **v1.UNKNOWN)),
        }
        try:
            answers = self.session.ask(dict(self.shared, segment_id=segment["id"], evidence=records), qs, "scan")
        except v1.BudgetStop:
            position = next(i for i, s in enumerate(self.index.segments) if s["id"] == segment["id"])
            self.frontier.appendleft("range_%d_%d" % (position, position + 1))
            raise
        ordered, selected = self.ordered_options(answers["location"], segment["refs"], "location")
        self.cards.append({"id": segment["id"], "refs": segment["refs"], "labels": answers,
                           "selected_refs": [r for r in selected if r in segment["refs"]],
                           "uncertain": any(s in selected for s in v1.UNKNOWN)})
        self.recalled.update(self.index.spans[r]["step"] for r in selected if r in segment["refs"])
        # Store low-ranked alternatives as leads, not dozens of active nodes.
        self.leads.extend(ordered)

    def next_candidate(self):
        while True:
            while self.leads:
                ref = self.leads.popleft()
                step = self.index.spans[ref]["step"]
                if step not in self.attempted:
                    self.add_node(ref)
                    return step
            segment = self.next_segment()
            if segment is None:
                return None
            self.read_segment(segment)

    def do_action(self, node, action):
        if self.rounds >= self.config.max_rounds:
            raise v1.BudgetStop("round_budget")
        packet = self.node_packet(node, action)
        state = dict(self.shared, target_step=node["step"], evidence=packet["records"],
                     evidence_view={k: v for k, v in packet.items() if k != "records"},
                     previous_labels=node["labels"])
        answers = self.session.ask(state, self.action_questions(action, packet["records"]), action)
        self.rounds += 1
        self.inspected_refs.update(r["id"] for r in packet["records"])
        self.trace.append({"round": self.rounds - 1, "action": action, "step": node["step"],
                           "refs": [r["id"] for r in packet["records"]], "criterion": "finish_current_candidate"})
        super().apply(node, action, packet, answers)
        if action in ("upstream", "repair", "challenge"):
            # One attempted relation check is NOT proof that all context was read
            # or that an unknown relation was resolved. Keep that distinction.
            node.setdefault("attempted_checks", []).append(action)
            node.setdefault("check_coverage", {})[action] = {
                "remaining_spans": packet["external_remaining"],
                "relation": answers["relation"]["choice"], "evidence_ref": answers["citation"]["choice"]}
        return answers

    def search(self):
        focus = None
        while self.rounds < self.config.max_rounds:
            step = focus if focus is not None else self.next_candidate()
            focus = None
            if step is None:
                self.stop_reason = "frontier_exhausted"
                return
            if step in self.attempted:
                continue
            self.attempted.add(step)
            self.focus_order.append(step)
            node = self.nodes[step]
            self.do_action(node, "inspect")
            if node["status"] == "sidelined":
                continue
            if node["labels"]["need"]["choice"] == "local" and set(self.index.by_step[step]) - set(node["seen_local"]):
                self.do_action(node, "expand")
            for action in ("upstream", "repair", "challenge"):
                if node["status"] == "sidelined":
                    break
                self.do_action(node, action)
                if node["status"] == "redirected":
                    # An evidenced earlier cause can take over the current branch.
                    candidates = [n for n in self.nodes.values() if step in n["parents"] and n["step"] not in self.attempted]
                    if candidates:
                        focus = min(candidates, key=lambda n: n["step"])["step"]
                    break
            labels = node["labels"]
            if node["status"] == "open" and all(k in labels and self.confident(labels[k], value)
                    for k, value in (("assessment", "introduces"), ("upstream", "local"),
                                     ("repair", "persists"), ("challenge", "supports"))):
                self.stop_reason = "candidate_evidence_sufficient"
                return
            # After closing a candidate, explore another region before spending
            # the remaining budget on every lead in the same leaf.
            if focus is None and self.frontier:
                segment = self.next_segment()
                if segment is not None:
                    previous = self.leads
                    self.leads = deque()
                    self.read_segment(segment)
                    self.leads.extend(previous)
        self.stop_reason = "round_budget"

    def final_card(self, step, excerpt_bytes):
        node = self.nodes[step]
        ref = node["refs"][0]
        return {"step": step, "labels": {k: {"choice": a["choice"], "probability": a["probabilities"][a["choice"]]}
                                         for k, a in node["labels"].items()},
                "attempted_checks": node.get("attempted_checks", []),
                "evidence": self.directory.excerpt(ref, excerpt_bytes),
                "notice": "Labels are hypotheses; this candidate view is an excerpt."}

    def finalize(self):
        ids = sorted(s for s, n in self.nodes.items() if n["status"] == "open" and "inspect" in n["done"])
        self.final_candidates = ids
        if not ids:
            return {"decision_status": "abstained", "decision_reason": "no_inspected_viable_candidate"}
        qs = self.final_questions(ids, with_roles=True)
        qs["root_step"] = v1.question(
            "Make a point prediction: which inspected candidate best explains the initiating, decisive, unrepaired error? "
            "Rank the available evidence even if uncertain; verification is reported separately. "
            "Use unknown only if none has relevant evidence; outside if evidence identifies a missing root.",
            dict({str(s): "Original step %d" % s for s in ids}, **v1.UNKNOWN))
        for width in (1800, 900, 300, 80):
            state = dict(self.shared, candidates=[self.final_card(s, width) for s in ids],
                         coverage=self.coverage(), output_policy="point_prediction_with_separate_verification")
            if self.session.size(state, qs) <= min(self.config.request_bytes, self.config.final_request_bytes):
                break
        answers = self.session.ask(state, qs, "final")
        root, role = answers["root_step"], answers["responsible_role"]
        result = {"decision_status": "abstained", "decision_reason": "no_final_candidate",
                  "final_answers": answers, "final_set": self.policy.select(root, "root_step")}
        if root["choice"] in v1.UNKNOWN:
            return result
        step = int(root["choice"])
        node = self.nodes[step]
        packet = self.candidate_view(step)["evidence"]
        tentative_role = self.names[int(role["choice"][1:])] if role["choice"] != "unknown" else None
        result.update(tentative_step=step, tentative_role=tentative_role,
                      predicted_step=step, predicted_role=tentative_role,
                      decision_status="best_effort", decision_reason="point_prediction_unverified",
                      candidate_evidence_refs=[self.index.spans[r["id"]] for r in packet["records"] if r["step"] == step])
        try:
            check = self.session.ask(dict(self.shared, proposed_root_step=step, evidence=packet,
                                          previous_labels=node["labels"], coverage=self.coverage()), {
                "support": v1.question("Independently check the proposed root against the ORIGINAL evidence. "
                    "Is it an initiating, decisive, unrepaired error? Labels are not proof. Unseen text remains unknown.", {
                        "supported": "Shown original evidence supports the attribution",
                        "refuted": "Shown evidence contradicts it", "unknown": "Evidence is insufficient"}),
                "citation": v1.question("Which original fragment directly establishes the proposed error itself?",
                    dict(v1.pointers(packet["records"]), none="No direct support", unknown="Insufficient evidence"))}, "verify")
        except v1.BudgetStop as error:
            result["decision_reason"] = "verification_budget_exhausted"
            result["verification_stop"] = str(error)
            return result
        result["verification"] = check
        if self.confident(check["support"], "refuted"):
            result.update(predicted_step=None, predicted_role=None, decision_status="abstained",
                          decision_reason="independent_evidence_refuted")
            return result
        citation = check["citation"]["choice"]
        direct = {r["id"] for r in packet["records"] if r["step"] == step}
        attempted = all(a in node.get("attempted_checks", []) for a in ("upstream", "repair", "challenge"))
        # Supported status additionally requires relation evidence, not merely
        # marking three checks attempted or reaching the end of a retrieved page.
        relations = all(k in node["labels"] and self.confident(node["labels"][k], label)
                        for k, label in (("assessment", "introduces"), ("upstream", "local"),
                                         ("repair", "persists"), ("challenge", "supports")))
        if (self.confident(root, root["choice"]) and self.confident(check["support"], "supported")
                and citation in direct and attempted and relations):
            result.update(decision_status="supported", decision_reason="retrieved_evidence_checked",
                          verified_step=step, verified_role=tentative_role if role["choice"] != "unknown"
                          and self.confident(role, role["choice"]) else None,
                          evidence_refs=[self.index.spans[citation]])
        return result

    def coverage(self):
        return {"total_segments": len(self.index.segments), "read_segments": len(self.cards),
                "routing_questions": len(self.routed), "pending_regions": len(self.frontier),
                "pending_leads": len(self.leads), "action_spans_read": len(self.inspected_refs),
                "full_scan": len(self.cards) == len(self.index.segments),
                "verification_scope": "retrieved_original_evidence; not exhaustive global causality"}

    def run(self):
        try:
            self.search()
        except v1.BudgetStop as error:
            self.stop_reason = str(error)
        # Search reaching its allowance does not consume or skip finalization.
        try:
            result = self.finalize()
        except v1.BudgetStop as error:
            result = {"decision_status": "abstained", "decision_reason": str(error)}
        return dict({"predicted_step": None, "predicted_role": None, "verified_step": None, "verified_role": None},
                    **result, method=PROTOCOL, config=asdict(self.config), model=ev.MODEL,
                    output_policy="point_prediction_and_separate_evidence_supported_subset",
                    history_sha256=self.index.history_sha256, segments=len(self.index.segments),
                    scanned_segments=len(self.cards), stop_reason=self.stop_reason, coverage=self.coverage(),
                    calibration_mode="question_level_calibrated_sets" if self.policy.calibration else "uncalibrated_heuristic",
                    recall_candidates=sorted(self.recalled), expanded_candidates=sorted(self.expanded),
                    final_candidates=self.final_candidates, evidence_cards=self.cards, routing=self.routed,
                    search_nodes=list(self.nodes.values()), search_trace=self.trace, focus_order=self.focus_order,
                    relation_hypotheses=self.edges, choice_events=self.session.events,
                    response_warnings=self.session.warnings,
                    budget={"logical_calls": self.session.calls, "request_bytes": self.session.request_bytes,
                            "input_tokens": self.session.input_tokens, "output_tokens": self.session.output_tokens,
                            "note": "Final calls/bytes/estimated tokens reserved. Actual token usage is post-response; "
                                    "estimates are not hard billing guarantees. Retries may add network attempts."})


def predict(history, client, config=None, calibration=None):
    return Pipeline(history, client, config, calibration).run()
