"""Experimental RCTA-inspired attribution using only JEV Choice questions.

Evidence cards are extracts, not generated summaries. Relation labels are model
hypotheses, not verified causal edges. No reference annotations enter predict().
"""
import math
import re

import evaluate as ev

CONFIG = {
    "protocol": "jev-rcta-choice-v2", "model": ev.MODEL,
    "singleton_policy": "deterministic_without_api",
    "segment_bytes": 24000, "piece_bytes": 8000, "segment_steps": 60,
    "overlap_steps": 3, "overlap_bytes": 600, "recall_k": 3,
    "group_size": 8, "trace_candidates": 16, "trace_depth": 2,
    "card_bytes": 7000, "evidence_bytes": 1800,
    "request_bytes": 62000, "shared_context": dict(ev.CONFIG),
}
RELATIONS = {
    "upstream": "The instruction already introduces the same decisive error; the candidate follows it without repair.",
    "local": "The candidate introduces a decisive error absent from the instruction or departs from it.",
    "repaired": "The apparent error is demonstrably repaired before the final failure.",
    "unrelated": "The supplied instruction and candidate do not support the same failure mechanism.",
    "uncertain": "The supplied original evidence is insufficient to distinguish these explanations.",
}


def choice(instructions, criteria):
    return {"type": "choice", "instructions": ev.RULES + instructions, "criteria": criteria}


def ask(client, state, questions, tag):
    active = {k: q for k, q in questions.items() if len(q["criteria"]) != 1}
    answers = {k: {"choice": next(iter(q["criteria"])),
                   "probabilities": {next(iter(q["criteria"])): 1.0}, "confidence": None,
                   "source": "deterministic_single_option"}
               for k, q in questions.items() if len(q["criteria"]) == 1}
    payload = {"model": ev.MODEL, "state": state, "questions": active}
    size = len(ev.dumps(payload).encode())
    if size > CONFIG["request_bytes"]:
        raise ValueError("Request exceeds conservative UTF-8 budget: %d" % size)
    for question in questions.values():
        if not 1 <= len(question["criteria"]) <= 255:
            raise ValueError("Choice option count outside 1..255")
    if active:
        answers.update(client.call(state, active, tag).get("answers", {}))
    for key, question in questions.items():
        answer = answers.get(key, {})
        if answer.get("choice") not in question["criteria"]:
            raise ValueError("Invalid Choice response for " + key)
    return answers


def ranked(answer, ids, k):
    """Rank only within one question; never compare probabilities across calls."""
    selected = answer["choice"]
    probabilities = answer.get("probabilities", {})
    def probability(step):
        value = float(probabilities.get(str(step), 0))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid probability")
        return value
    ordered = sorted(set(ids), key=lambda s: (-probability(s), s))
    if selected.isdigit() and int(selected) in ordered:
        ordered.remove(int(selected))
        ordered.insert(0, int(selected))
    return ordered[:k]


def segments(history):
    """Every source character occurs in a primary segment, including huge steps."""
    batch = []
    for h in history:
        parts = ev.split_text(h["content"], CONFIG["piece_bytes"])
        for i, part in enumerate(parts):
            item = dict(ev.record(h), content=part)
            if len(parts) > 1:
                item["part"] = "%d/%d" % (i + 1, len(parts))
            if batch and (len(ev.dumps(batch + [item]).encode()) > CONFIG["segment_bytes"]
                          or len(batch) >= CONFIG["segment_steps"]):
                yield batch
                batch = []
            batch.append(item)
    if batch:
        yield batch


def handoff(history, step):
    """Keep explicit recipient matches distinct from generic plan context."""
    target = ev.normalize_role(history[step].get("name", ""))
    nearest = None
    for h in reversed(history[:step]):
        match = re.search(r"\(\s*->\s*(.*?)\)", h.get("name", ""))
        if match:
            if nearest is None:
                nearest = h["step"]
            if ev.normalize_role(match.group(1)) == target:
                return h["step"], True
    return nearest, False


def evidence(history, step):
    return {"candidate": ev.record(history[step], CONFIG["evidence_bytes"]),
            "neighbors": [ev.record(history[s], 350) for s in (step - 1, step + 1)
                          if 0 <= s < len(history)]}


def outline(cards):
    """Budget each card equally; preserve IDs and explicit omission markers."""
    if not cards:
        return []
    budget = CONFIG["card_bytes"] // len(cards)
    result = []
    for card in cards:
        item = {k: v for k, v in card.items() if k != "extract"}
        remaining = budget - len(ev.dumps(item).encode()) - 32
        if remaining >= 80:
            item["extract"] = ev.clip(card["extract"], remaining)
        else:
            item["extract_omitted"] = True
        result.append(item)
    if len(ev.dumps(result).encode()) > CONFIG["card_bytes"]:
        # For unusually many fragments, a complete ID-only outline remains available
        # in saved cards. Do not silently truncate or pretend the view is complete.
        return {"omitted": True, "segments": len(cards),
                "reason": "Complete card index exceeds outline budget; see saved evidence_cards."}
    return result


def reduce_candidates(history, candidates, shared, client, tag, limit):
    candidates = sorted(set(candidates))
    level = 0
    while len(candidates) > limit:
        retained = set()
        for offset in range(0, len(candidates), CONFIG["group_size"]):
            group = candidates[offset:offset + CONFIG["group_size"]]
            if len(group) <= CONFIG["recall_k"]:
                retained.update(group)
                continue
            state = dict(shared, candidate_evidence=[evidence(history, s) for s in group])
            answers = ask(client, state, ev.questions(group), "%s_%d_%d" % (tag, level, offset))
            retained.update(ranked(answers["root_step"], group, CONFIG["recall_k"]))
        candidates = sorted(retained)
        level += 1
    return candidates


def predict(history, client):
    if not history or [h["step"] for h in history] != list(range(len(history))):
        raise ValueError("Expected non-empty contiguous 0-based history")
    names = ev.roles(history)
    if not names:
        raise ValueError("No recorded workflow roles")
    shared = ev.context(history)
    cards, recall = [], set()
    for index, batch in enumerate(segments(history)):
        ids = sorted({h["step"] for h in batch})
        criteria = {str(s): "Original step %d" % s for s in ids}
        questions = {
            "suspect": choice("Select the strongest candidate error in this segment. "
                              "Choose none if it provides no supported candidate; uncertain if evidence is insufficient.",
                              dict(criteria, none="No supported candidate in this segment",
                                   uncertain="Insufficient evidence; retain alternatives")),
            "anchor": choice("Select the step most informative about task progress or the final failure "
                             "for an extractive evidence card. It need not be erroneous.", criteria),
        }
        start = ids[0]
        overlap = [ev.record(h, CONFIG["overlap_bytes"])
                   for h in history[max(0, start - CONFIG["overlap_steps"]):start]]
        answers = ask(client, dict(shared, segment=batch, overlap=overlap), questions, "recall_%03d" % index)
        selected = answers["suspect"]["choice"]
        retained = [] if selected == "none" else ranked(answers["suspect"], ids, CONFIG["recall_k"])
        recall.update(retained)
        anchor = int(answers["anchor"]["choice"])
        # Extract the actual viewed fragment, not a different part of a huge record.
        extract = "\n".join(h["content"] for h in batch if h["step"] == anchor)
        cards.append({"segment": index, "range": [ids[0], ids[-1]], "anchor": anchor,
                      "role": history[anchor].get("name", ""),
                      "suspect": selected, "candidates": retained, "extract": ev.clip(extract, 1200)})
    fallback = not recall
    if fallback:
        recall.update(card["anchor"] for card in cards)
    shared = dict(shared, extractive_outline=outline(cards),
                  outline_note="Cards contain original excerpts and fallible Choice selections, not verified conclusions.")
    seeds = reduce_candidates(history, recall, shared, client, "seed", CONFIG["trace_candidates"])
    expanded, edges, seen = set(seeds), [], set()
    frontier = list(seeds)
    for depth in range(CONFIG["trace_depth"]):
        next_frontier = set()
        for step in frontier:
            upstream, addressed = handoff(history, step)
            if upstream is None or (upstream, step) in seen:
                continue
            seen.add((upstream, step))
            expanded.add(upstream)
            state = dict(shared, candidate_evidence=evidence(history, step),
                         earlier_instruction=ev.record(history[upstream], 5000),
                         instruction_addressed_to_candidate=addressed)
            questions = {"relation": choice(
                "Compare the candidate action and the earlier instruction using original evidence. "
                "For repaired, require explicit later evidence of repair. An unmatched handoff is only plan context. "
                "Do not infer causation from temporal order or recipient matching alone.", RELATIONS)}
            answer = ask(client, state, questions, "trace_%d_%d_%d" % (depth, upstream, step))["relation"]
            edges.append({"upstream": upstream, "candidate": step, "addressed": addressed,
                          "relation": answer["choice"], "confidence": answer.get("confidence")})
            if answer["choice"] == "upstream":
                next_frontier.add(upstream)
        frontier = sorted(next_frontier)
    # Relations guide the final judgment; even 'repaired' never deletes a candidate.
    shared = dict(shared, relation_hypotheses=edges,
                  relation_note="Unverified model classifications; original evidence takes precedence.")
    finalists = reduce_candidates(history, expanded, shared, client, "final_reduce", CONFIG["group_size"])
    answers = ask(client, dict(shared, candidate_evidence=[evidence(history, s) for s in finalists]),
                  ev.questions(finalists, names), "final")
    return {"predicted_step": int(answers["root_step"]["choice"]),
            "predicted_role": answers["responsible_role"]["choice"],
            "step_confidence": answers["root_step"].get("confidence"),
            "role_confidence": answers["responsible_role"].get("confidence"),
            "method": CONFIG["protocol"], "segments": len(cards), "anchor_fallback": fallback,
            "recall_candidates": sorted(recall), "trace_seeds": seeds,
            "expanded_candidates": sorted(expanded), "final_candidates": finalists,
            "evidence_cards": cards, "relation_hypotheses": edges}
