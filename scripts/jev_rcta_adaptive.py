"""Adaptive JEV-RCTA: typed evidence, variable-width sets and noisy question search.

Literature-inspired engineering prototype, not a reproduction of EC2, CBM or
Self-RAG. The question scheduler is a transparent cost-aware heuristic. Model
labels are hypotheses, never new evidence. Only history enters predict().
"""
from dataclasses import asdict, dataclass
import math

import evaluate as ev
from rcta_choice_policy import ChoicePolicy, distribution
from rcta_evidence import EvidenceIndex

PROTOCOL = "jev-rcta-adaptive-v1"
UNKNOWN = {"outside": "None of these candidates; search elsewhere.",
           "unknown": "The supplied evidence is insufficient to decide."}


@dataclass(frozen=True)
class Config:
    span_bytes: int = 2200
    segment_bytes: int = 32000
    segment_records: int = 64
    evidence_bytes: int = 10000
    request_bytes: int = 62000
    max_calls: int = 64
    max_request_bytes: int = 1600000
    max_input_tokens: int = 300000
    max_rounds: int = 24
    selection_mass: float = 0.9
    decision_threshold: float = 0.8

    def validate(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid configuration: " + key)
            if key not in ("selection_mass", "decision_threshold") and type(value) is not int:
                raise ValueError("Integer budget required: " + key)
        if self.selection_mass > 1 or self.decision_threshold > 1:
            raise ValueError("Probability thresholds must be <= 1")
        if self.evidence_bytes < self.span_bytes * 2 + 800 or self.segment_records > 250:
            raise ValueError("Evidence budget or segment option count is invalid")


def question(instructions, options):
    return {"type": "choice", "instructions": ev.RULES + instructions, "criteria": options}


def pointers(records):
    return {r["id"]: "Original step %d, UTF-8 bytes [%d, %d)" %
            (r["step"], r["start_byte"], r["end_byte"]) for r in records}


class BudgetStop(RuntimeError):
    pass


class Session:
    def __init__(self, client, config):
        self.client, self.config = client, config
        self.calls, self.request_bytes, self.input_tokens, self.output_tokens = 0, 0, 0, 0
        self.events, self.warnings = [], []

    def size(self, state, questions):
        return len(ev.dumps({"model": ev.MODEL, "state": state, "questions": questions}).encode())

    def ask(self, state, questions, kind):
        if any(not 2 <= len(q["criteria"]) <= 255 for q in questions.values()):
            raise ValueError("Choice must contain 2..255 options")
        size = self.size(state, questions)
        if size > self.config.request_bytes:
            raise BudgetStop("request_context_limit")
        if self.calls >= self.config.max_calls:
            raise BudgetStop("call_budget")
        if self.request_bytes + size > self.config.max_request_bytes:
            raise BudgetStop("request_byte_budget")
        if self.input_tokens >= self.config.max_input_tokens:
            raise BudgetStop("input_token_budget")
        tag = "%03d_%s" % (self.calls, kind)
        self.calls += 1
        self.request_bytes += size
        response = self.client.call(state, questions, tag)
        usage = response.get("usage", {})
        for key in ("input_tokens", "output_tokens"):
            value = usage.get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError("Invalid API token usage")
            setattr(self, key, getattr(self, key) + value)
        answers = {key: distribution(response.get("answers", {}).get(key), q["criteria"])
                   for key, q in questions.items()}
        for key, answer in answers.items():
            self.warnings.extend({"tag": tag, "question": key, "warning": w} for w in answer["warnings"])
        self.events.append({"id": len(self.events), "kind": kind, "tag": tag,
                            "request_bytes": size, "answers": answers})
        return answers

    def exhausted(self, reserve=0):
        return (self.calls + reserve >= self.config.max_calls or
                self.input_tokens >= self.config.max_input_tokens or
                self.request_bytes + reserve * self.config.request_bytes >= self.config.max_request_bytes)


class Pipeline:
    def __init__(self, history, client, config=None, calibration=None):
        self.config = config or Config()
        self.config.validate()
        self.policy = ChoicePolicy(self.config.selection_mass, calibration)
        self.index = EvidenceIndex(history, self.config.span_bytes,
                                   self.config.segment_bytes, self.config.segment_records)
        self.names = ev.roles(self.index.history)
        if not self.names:
            raise ValueError("No recorded roles")
        self.session = Session(client, self.config)
        context_refs = self.index.by_step[0][:2] + [self.index.by_step[s][-1]
                       for s in range(max(0, len(history) - 2), len(history))]
        self.shared = {"task_and_outcome": self.index.packet(context_refs, 11000),
                       "known_outcome": "This completed trajectory failed; no external evaluator reason is supplied.",
                       "label_notice": "Labels and branches are fallible hypotheses. Original evidence takes precedence."}
        self.cards, self.nodes, self.trace, self.edges = [], {}, [], []
        self.recalled, self.expanded, self.deferred = set(), set(), []
        self.stop_reason = "frontier_exhausted"

    def confident(self, answer, label):
        return answer["choice"] == label and answer["probabilities"][label] >= self.config.decision_threshold

    def refs_selected(self, answer, refs, kind):
        chosen = self.policy.select(answer, kind)
        result = [r for r in chosen if r in refs]
        # A missing winner is not grounds to silently discard the whole segment.
        if not result and "unknown" in chosen:
            result = list(refs)
        return result

    def add_node(self, ref, parent=None, source="recall"):
        step = self.index.spans[ref]["step"]
        if step not in self.nodes:
            self.nodes[step] = {"step": step, "refs": [], "seen_local": [], "status": "open",
                                "labels": {}, "done": [], "pages": {}, "seen": {},
                                "parents": [], "source": source, "support_refs": []}
        node = self.nodes[step]
        if ref not in node["refs"]:
            node["refs"].append(ref)
            if node["status"] == "sidelined":
                node["status"], node["done"] = "open", []
                self.trace.append({"action": "reopen", "step": step, "evidence_ref": ref})
        if parent is not None and parent not in node["parents"]:
            node["parents"].append(parent)
        self.expanded.add(step)

    def scan(self):
        for segment in self.index.segments:
            if self.session.exhausted(reserve=3):
                self.stop_reason = "budget_during_scan"
                break
            records = [self.index.read(r) for r in segment["refs"]]
            questions = {
                "event": question("Which activity is most relevant in this segment?", {
                    "instruction": "Planning or instruction", "execution": "Action or tool execution",
                    "observation": "Result or check", "repair": "Repair attempt", "mixed": "Mixed or unclear"}),
                "phenomenon": question("What observable failure-related phenomenon is supported here?", {
                    "constraint": "A requirement conflicts with an action", "execution": "An execution fails",
                    "state": "State or artifact mismatch", "unsupported": "Conclusion lacks evidence",
                    "none": "No visible anomaly", "unknown": "Insufficient evidence"}),
                "relevance": question("How does this segment relate to the final failure?", {
                    "cause": "May introduce a decisive error", "effect": "May propagate or expose an error",
                    "repair": "May repair a relevant error", "irrelevant": "Supported to be unrelated",
                    "unknown": "Relationship is unclear"}),
                "location": question("Select the original fragment most useful for locating the decisive error. "
                    "An observation is a lead, not necessarily the root.", dict(pointers(records), **UNKNOWN)),
            }
            answers = self.session.ask(dict(self.shared, segment_id=segment["id"], evidence=records), questions, "scan")
            selected = self.refs_selected(answers["location"], segment["refs"], "location")
            deferred = self.confident(answers["relevance"], "irrelevant") or self.confident(answers["location"], "outside")
            card = {"id": segment["id"], "refs": segment["refs"], "labels": answers,
                    "selected_refs": selected, "deferred": deferred}
            self.cards.append(card)
            if deferred:
                self.deferred.append(card)
            else:
                for ref in selected:
                    self.add_node(ref)
                    self.recalled.add(self.index.spans[ref]["step"])

    def node_packet(self, node, action):
        step = node["step"]
        # Prefer the exact selected fragment, including middle fragments of a huge step.
        own = node["refs"] + self.index.by_step[step]
        if action == "expand":
            fresh = [r for r in self.index.by_step[step] if r not in node["seen_local"]]
            own = fresh + node["refs"]
        own_packet = self.index.packet(own, self.config.evidence_bytes // 2)
        if action in ("inspect", "expand"):
            neighbor = [r for s in (step - 1, step + 1) if s in self.index.by_step
                        for r in self.index.by_step[s]]
            extra = self.index.packet(neighbor, self.config.evidence_bytes // 2)
        else:
            refs = self.index.retrieve(step, action, node["seen"].get(action, []))
            extra = self.index.packet(refs, self.config.evidence_bytes // 2)
        records = list({r["id"]: r for r in own_packet["records"] + extra["records"]}.values())
        return {"records": records, "new_external_refs": [r["id"] for r in extra["records"]],
                "external_remaining": extra["remaining_spans"],
                "local_remaining": len(set(self.index.by_step[step]) - set(node["seen_local"]) -
                                       {r["id"] for r in own_packet["records"]})}

    def action_questions(self, action, records):
        cites = dict(pointers(records), none="No fragment supports the requested relation", unknown="Cannot identify support")
        if action in ("inspect", "expand"):
            return {
                "assessment": question("Does the TARGET step introduce the decisive error, given the actual text?", {
                    "introduces": "Introduces a relevant error", "propagates": "Carries an earlier error forward",
                    "refutes": "Evidence contradicts this step being the root", "unknown": "Insufficient evidence"}),
                "need": question("Which missing evidence would best distinguish the competing explanations?", {
                    "upstream": "Earlier requirements or instructions", "repair": "Later repairs or validation",
                    "local": "More of this step", "challenge": "Counterevidence elsewhere", "none": "Nothing obvious missing"}),
                "citation": question("Which fragment most directly bears on the TARGET step's possible error?", cites)}
        if action == "upstream":
            options = {"upstream": "An earlier instruction introduces the same error and is followed",
                       "local": "Earlier context does not introduce the error; the target introduces it",
                       "unrelated": "Retrieved earlier context is unrelated", "unknown": "Insufficient evidence"}
            prompt = "Compare the TARGET step with earlier evidence. Recipient matching alone proves no causal relation. "
            cite_prompt = "Which earlier fragment directly introduces the same error followed by the TARGET step?"
        elif action == "repair":
            options = {"repaired": "Explicit later evidence verifies this error was successfully repaired",
                       "persists": "Explicit later evidence shows this same error persists",
                       "unknown": "No decisive repair outcome is available"}
            prompt = "Check whether the TARGET error was successfully repaired before final failure. An attempted fix is not proof. "
            cite_prompt = "Which later fragment directly establishes repair or persistence of the TARGET error?"
        else:
            options = {"supports": "Evidence supports the TARGET as an unrepaired initiating error",
                       "refutes": "Evidence contradicts the TARGET being an unrepaired initiating error",
                       "unknown": "Evidence does not resolve the hypothesis"}
            prompt = "Actively test the TARGET root hypothesis against counterevidence. Do not assume it is correct. "
            cite_prompt = "Which fragment directly supports or contradicts the TARGET root hypothesis?"
        return {"relation": question(prompt, options), "citation": question(cite_prompt, cites)}

    def available(self, node):
        if node["status"] == "sidelined":
            return []
        if "inspect" not in node["done"]:
            return [("inspect", 4 if node["source"] == "upstream" else 2)]
        hint = node["labels"].get("need", {}).get("choice")
        candidates = []
        assessment = node["labels"]["assessment"]
        ambiguous = max(assessment["probabilities"].values()) < self.config.decision_threshold
        if hint == "local" or assessment["choice"] == "unknown" or ambiguous:
            if set(self.index.by_step[node["step"]]) - set(node["seen_local"]):
                candidates.append(("expand", 4))
        for action in ("upstream", "repair", "challenge"):
            if action not in node["done"]:
                priority = 5 if hint == action else 2
                if action == "upstream" and node["labels"]["assessment"]["choice"] == "propagates":
                    priority = 7
                candidates.append((action, priority / (1 + node["pages"].get(action, 0))))
        return candidates

    def choose_action(self):
        options = []
        for step, node in sorted(self.nodes.items()):
            for action, priority in self.available(node):
                packet = self.node_packet(node, action)
                state = dict(self.shared, target_step=step, evidence=packet["records"],
                             evidence_view={k: v for k, v in packet.items() if k != "records"},
                             previous_labels=node["labels"])
                questions = self.action_questions(action, packet["records"])
                cost = self.session.size(state, questions)
                options.append((priority / max(cost, 1), -step, action, state, questions, packet))
        if not options:
            return None
        return max(options, key=lambda x: (x[0], x[1], x[2]))

    def apply(self, node, action, packet, answers):
        step = node["step"]
        node["seen_local"] = sorted(set(node["seen_local"]) | {
            r["id"] for r in packet["records"] if r["step"] == step})
        node["seen"][action] = sorted(set(node["seen"].get(action, [])) | set(packet["new_external_refs"]))
        node["pages"][action] = node["pages"].get(action, 0) + 1
        citation = answers["citation"]["choice"]
        visible = {r["id"]: r for r in packet["records"]}
        valid_citation = citation in visible
        if valid_citation:
            node["support_refs"] = list(dict.fromkeys(node["support_refs"] + [citation]))
        if action in ("inspect", "expand"):
            node["labels"].update(answers)
            if "inspect" not in node["done"]:
                node["done"].append("inspect")
            if self.confident(answers["assessment"], "refutes") and valid_citation:
                node["status"] = "sidelined"
        else:
            relation = answers["relation"]
            node["labels"][action] = relation
            temporal_citation = valid_citation and (action == "challenge" or
                action == "upstream" and visible[citation]["step"] < step or
                action == "repair" and visible[citation]["step"] > step)
            if (relation["choice"] != "unknown" and temporal_citation or packet["external_remaining"] == 0):
                if action not in node["done"]:
                    node["done"].append(action)
            if action == "upstream":
                upstream_refs = [r for r in visible if visible[r]["step"] < step]
                selected = self.refs_selected(answers["citation"], upstream_refs, "citation")
                if self.confident(relation, "upstream") and valid_citation and citation in upstream_refs:
                    for ref in selected:
                        self.add_node(ref, parent=step, source="upstream")
                        self.edges.append({"from": step, "to": self.index.spans[ref]["step"],
                                           "evidence_ref": ref, "relation": relation,
                                           "verified_causal_edge": False})
                    node["status"] = "redirected"
            if action == "repair" and self.confident(relation, "repaired") and valid_citation and visible[citation]["step"] > step:
                node["status"] = "sidelined"
            if action == "challenge" and self.confident(relation, "refutes") and valid_citation:
                node["status"] = "sidelined"
        if node["status"] == "sidelined":
            self.trace.append({"action": "backtrack", "step": step, "cause": action, "evidence_ref": citation})

    def reopen_segment(self):
        if not self.deferred:
            return False
        card = self.deferred.pop(0)
        # Revisit a deferred segment only when the current frontier is exhausted.
        for ref in card["selected_refs"] or card["refs"]:
            self.add_node(ref, source="reopened_segment")
        self.trace.append({"action": "reopen_segment", "segment": card["id"]})
        return True

    def search(self):
        for round_id in range(self.config.max_rounds):
            if self.session.exhausted(reserve=2):
                self.stop_reason = "search_budget"
                return
            option = self.choose_action()
            while option is None and self.reopen_segment():
                option = self.choose_action()
            if option is None:
                self.stop_reason = "frontier_exhausted"
                return
            priority, neg_step, action, state, questions, packet = option
            answers = self.session.ask(state, questions, action)
            self.trace.append({"round": round_id, "action": action, "step": -neg_step,
                               "criterion": "heuristic_discrimination_per_request_byte", "priority": priority,
                               "refs": [r["id"] for r in packet["records"]],
                               "event_id": self.session.events[-1]["id"]})
            self.apply(self.nodes[-neg_step], action, packet, answers)
            remaining = [n for n in self.nodes.values() if n["status"] != "sidelined"]
            if len(remaining) == 1:
                labels = remaining[0]["labels"]
                checks = {"assessment": "introduces", "upstream": "local",
                          "repair": "persists", "challenge": "supports"}
                if all(k in labels and self.confident(labels[k], v) for k, v in checks.items()):
                    self.stop_reason = "evidence_sufficient"
                    return
        self.stop_reason = "round_budget"

    def candidate_view(self, step):
        node = self.nodes[step]
        refs = node["refs"][:1] + node["support_refs"] + node["refs"][1:]
        packet = self.index.packet(refs, self.config.evidence_bytes)
        return {"step": step, "labels": node["labels"], "evidence": packet}

    def final_questions(self, ids, with_roles=False):
        result = {"root_step": question("Choose the earliest decisive unrepaired root among the candidate steps. "
            "Use unknown if evidence cannot distinguish them; outside if the root is missing.",
            dict({str(s): "Original step %d" % s for s in ids}, **UNKNOWN))}
        if with_roles:
            result["responsible_role"] = question("Select the responsible decision-making role independently of "
                "the selected step's emitter. Choose unknown if responsibility is not established.",
                dict({"r%d" % i: name for i, name in enumerate(self.names)}, unknown="Responsibility is unclear"))
        return result

    def finalize(self):
        candidates = sorted(s for s, n in self.nodes.items() if n["status"] != "sidelined")
        self.final_candidates = candidates
        if not candidates:
            return {"decision_status": "abstained", "decision_reason": "no_supported_candidates"}
        # Context-limited tournaments keep adaptive sets; if no reduction is
        # justified, abstain instead of silently replacing mass selection by top-k.
        while True:
            groups, batch = [], []
            for step in candidates:
                test = batch + [step]
                state = dict(self.shared, candidates=[self.candidate_view(s) for s in test])
                qs = self.final_questions(test, with_roles=True)
                if batch and (len(test) > 250 or self.session.size(state, qs) > self.config.request_bytes):
                    groups.append(batch)
                    batch = [step]
                else:
                    batch = test
            if batch:
                groups.append(batch)
            if len(groups) == 1:
                break
            retained = set()
            for group in groups:
                state = dict(self.shared, candidates=[self.candidate_view(s) for s in group])
                ans = self.session.ask(state, self.final_questions(group), "reduce")["root_step"]
                selected = self.policy.select(ans, "root_step")
                if any(key in selected for key in UNKNOWN):
                    retained.update(group)
                retained.update(int(s) for s in selected if s not in UNKNOWN)
            if len(retained) >= len(candidates):
                return {"decision_status": "abstained", "decision_reason": "final_context_ambiguity"}
            candidates = sorted(retained)
            self.final_candidates = candidates
        state = dict(self.shared, candidates=[self.candidate_view(s) for s in candidates])
        answers = self.session.ask(state, self.final_questions(candidates, with_roles=True), "final")
        root, role = answers["root_step"], answers["responsible_role"]
        result = {"decision_status": "abstained", "decision_reason": "uncertain_final_choice",
                  "final_answers": answers, "final_set": self.policy.select(root, "root_step")}
        if root["choice"] in UNKNOWN:
            return result
        step = int(root["choice"])
        packet = self.candidate_view(step)["evidence"]
        check = self.session.ask(dict(self.shared, proposed_root_step=step, evidence=packet), {
            "support": question("Independently check the proposed root against original evidence, including possible "
                "earlier causes and later repairs. Is it an initiating, decisive, unrepaired error?", {
                    "supported": "Original evidence supports this root attribution",
                    "refuted": "Evidence contradicts this attribution", "unknown": "Insufficient evidence"}),
            "citation": question("Select the fragment that directly establishes the proposed root error itself.",
                dict(pointers(packet["records"]), none="No direct support", unknown="Insufficient evidence"))}, "verify")
        citation = check["citation"]["choice"]
        direct = {r["id"] for r in packet["records"] if r["step"] == step}
        role_resolved = role["choice"] != "unknown" and self.confident(role, role["choice"])
        result.update(verification=check, tentative_step=step,
                      tentative_role=self.names[int(role["choice"][1:])] if role["choice"] != "unknown" else None)
        inspected = "inspect" in self.nodes[step]["done"]
        checked = all(k in self.nodes[step]["done"] for k in ("upstream", "repair", "challenge"))
        complete_scan = len(self.cards) == len(self.index.segments)
        if (self.confident(check["support"], "supported") and self.confident(root, root["choice"])
                and citation in direct and inspected and checked and complete_scan):
            result.update(predicted_step=step, predicted_role=result["tentative_role"] if role_resolved else None,
                          decision_status="supported", decision_reason="evidence_checked",
                          evidence_refs=[self.index.spans[citation]])
        elif not complete_scan or not inspected or not checked:
            result["decision_reason"] = "incomplete_evidence_checks"
        return result

    def run(self):
        self.final_candidates = []
        result = {}
        try:
            self.scan()
            self.search()
            result = self.finalize()
        except BudgetStop as error:
            self.stop_reason = str(error)
            result = {"decision_status": "abstained", "decision_reason": str(error)}
        return dict({"predicted_step": None, "predicted_role": None}, **result,
                    method=PROTOCOL, config=asdict(self.config), model=ev.MODEL,
                    history_sha256=self.index.history_sha256, segments=len(self.index.segments),
                    scanned_segments=len(self.cards), stop_reason=self.stop_reason,
                    calibration_mode="question_level_calibrated_sets" if self.policy.calibration else "uncalibrated_heuristic",
                    recall_candidates=sorted(self.recalled), expanded_candidates=sorted(self.expanded),
                    final_candidates=self.final_candidates, evidence_cards=self.cards,
                    search_nodes=list(self.nodes.values()), search_trace=self.trace, relation_hypotheses=self.edges,
                    choice_events=self.session.events, response_warnings=self.session.warnings,
                    budget={"logical_calls": self.session.calls, "request_bytes": self.session.request_bytes,
                            "input_tokens": self.session.input_tokens, "output_tokens": self.session.output_tokens,
                            "note": "Includes replayed calls; transport retries can add attempts. "
                                    "Input-token cap is checked after each response; bytes are prechecked."})


def predict(history, client, config=None, calibration=None):
    return Pipeline(history, client, config, calibration).run()
