"""Choice validation and adaptive sets; no training or network access.

Default mass thresholds are heuristics, not calibrated correctness probabilities.
Optional split-calibration uses a conservative cumulative-rank score. Guarantees
for exchangeable single questions do not extend to the adaptive search policy.
"""
import argparse
import collections
import json
import math
from pathlib import Path


def distribution(answer, criteria):
    if not isinstance(answer, dict) or answer.get("choice") not in criteria:
        raise ValueError("Missing or invalid Choice answer")
    raw = answer.get("probabilities")
    if not isinstance(raw, dict) or set(raw) != set(criteria):
        raise ValueError("Choice probabilities must cover exactly the requested options")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or not 0 <= p <= 1 for p in raw.values()):
        raise ValueError("Invalid Choice probability")
    total = sum(raw.values())
    if total <= 0 or abs(total - 1) > 0.025:
        raise ValueError("Choice probabilities do not sum to one")
    confidence = answer.get("confidence")
    if confidence is not None and (isinstance(confidence, bool) or
            not isinstance(confidence, (int, float)) or
            not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Invalid Choice confidence")
    p = {k: v / total for k, v in raw.items()}
    top = sorted(p, key=lambda k: (-p[k], k))[0]
    return {"choice": top, "reported_choice": answer["choice"], "probabilities": p,
            "confidence": confidence,
            "entropy": -sum(v * math.log(v) for v in p.values() if v) / math.log(len(p))
                       if len(p) > 1 else 0.0,
            "warnings": (["reported_choice_not_probability_maximum"]
                         if p[answer["choice"]] < p[top] - 1e-9 else [])}


def adaptive_set(answer, mass=0.9):
    """Include the crossing option and all ties; never apply a fixed top-k cap."""
    if not math.isfinite(mass) or not 0 < mass <= 1:
        raise ValueError("Selection mass must be in (0, 1]")
    p = answer["probabilities"]
    ordered = sorted(p, key=lambda k: (-p[k], k))
    if mass == 1:
        return ordered
    selected, cumulative, boundary = [], 0.0, None
    for key in ordered:
        if cumulative >= mass - 1e-12 and boundary != p[key]:
            break
        selected.append(key)
        cumulative += p[key]
        boundary = p[key]
    return selected


class ChoicePolicy:
    def __init__(self, mass=0.9, calibration=None):
        if not math.isfinite(mass) or not 0 < mass <= 1:
            raise ValueError("Selection mass must be in (0, 1]")
        self.mass = mass
        self.calibration = calibration or {}
        if calibration:
            if calibration.get("format") != "rcta-choice-calibration-v1":
                raise ValueError("Unknown calibration format")
            if not calibration.get("case_ids") or not calibration.get("question_types"):
                raise ValueError("Calibration requires provenance and question types")
            for item in calibration["question_types"].values():
                if not isinstance(item, dict) or not 0 < item.get("mass", 0) <= 1:
                    raise ValueError("Invalid calibration threshold")

    def select(self, answer, kind):
        threshold = self.calibration.get("question_types", {}).get(kind, {}).get("mass", self.mass)
        return adaptive_set(answer, threshold)

    def check_holdout(self, case_ids):
        overlap = set(case_ids) & set(self.calibration.get("case_ids", []))
        if overlap:
            raise ValueError("Calibration and evaluation trajectories overlap")


def fit_calibration(records, alpha=0.1):
    """One labelled question per (trajectory, question type); no test labels.

    Input: case_id, kind, answer (raw Choice), truth (correct option).
    Uses a finite-sample quantile of cumulative rank mass; includes all ties.
    Different question kinds are calibrated separately. Caller must ensure the
    held-out questions follow the same sampling policy as deployment questions.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    grouped, seen = collections.defaultdict(list), set()
    for row in records:
        case, kind = row["case_id"], row["kind"]
        if not isinstance(case, str) or not case or not isinstance(kind, str) or not kind:
            raise ValueError("Non-empty case_id and kind required")
        if (case, kind) in seen:
            raise ValueError("Use one question per trajectory and kind for calibration")
        seen.add((case, kind))
        answer = distribution(row["answer"], row["answer"]["probabilities"])
        p, truth = answer["probabilities"], row["truth"]
        if truth not in p:
            raise ValueError("Calibration truth is not a requested option")
        score = min(1.0, sum(v for v in p.values() if v >= p[truth]))
        grouped[kind].append((score, answer, truth))
    if not grouped:
        raise ValueError("No calibration records")
    result = {"format": "rcta-choice-calibration-v1", "alpha": alpha,
              "case_ids": sorted({case for case, _ in seen}), "question_types": {},
              "scope": "Conservative question-level sets; no tree-level coverage guarantee. "
                       "Requires exchangeability and the same question sampling policy."}
    for kind, items in sorted(grouped.items()):
        n = len(items)
        rank = math.ceil((n + 1) * (1 - alpha))
        mass = sorted(x[0] for x in items)[rank - 1] if rank <= n else 1.0
        result["question_types"][kind] = {
            "n": n, "mass": mass,
            "top1_accuracy": sum(a["choice"] == y for _, a, y in items) / n,
            "mean_max_probability": sum(max(a["probabilities"].values()) for _, a, _ in items) / n,
            "brier": sum(sum((v - (key == y)) ** 2 for key, v in a["probabilities"].items())
                         for _, a, y in items) / n,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Offline calibration from held-out labelled Choice questions")
    parser.add_argument("--input", required=True, help="JSONL with case_id, kind, answer, truth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    result = fit_calibration(records, args.alpha)
    path = Path(args.output)
    if path.exists():
        raise ValueError("Choose a new calibration output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Saved calibration for %d question types; no API calls." % len(result["question_types"]))


if __name__ == "__main__":
    main()
