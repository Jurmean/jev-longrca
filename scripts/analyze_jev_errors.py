"""Analyze frozen JEV Full predictions locally; never performs inference/API calls."""
import collections
import csv
import json
from pathlib import Path

from evaluate import normalize_role, top_steps

ROOT = Path(__file__).resolve().parents[1]


def stage(d):
    p, gold = d["prediction"], d["reference_step"]
    if d["step_exact"]:
        return "correct"
    if d["method"] == "full_context":
        return "direct_wrong"
    if gold not in p["expanded_candidates"]:
        return "recall_miss"
    if gold not in p["final_candidates"]:
        return "reduction_drop"
    return "final_wrong"


def main():
    summary = json.loads((ROOT / "results/full/summary.json").read_text())
    details = summary["details"]
    segmented = [d for d in details if d["method"] == "segmented_choice"]
    direct = [d for d in details if d["method"] == "full_context"]
    errors = [d for d in details if not d["step_exact"]]
    counts = collections.Counter(stage(d) for d in details)
    stats = {
        "scope": "Frozen JEV Full; benchmark references, zero-based step IDs; no new inference",
        "protocol": summary["protocol"],
        "dataset_revision": summary["dataset_revision"],
        "n": len(details),
        "root_errors": len(errors),
        "stage_counts": dict(counts),
        "gold_unavailable_before_final": counts["recall_miss"] + counts["reduction_drop"],
        "gold_available_but_wrong": counts["final_wrong"] + counts["direct_wrong"],
        "segmented_funnel": {
            "n": len(segmented),
            **{key: sum(d["reference_step"] in d["prediction"][key] for d in segmented)
               for key in ("recall_candidates", "expanded_candidates", "final_candidates")},
            "correct": sum(d["step_exact"] for d in segmented),
        },
        "direct": {"n": len(direct), "correct": sum(d["step_exact"] for d in direct)},
        "root_error_direction": {
            "early": sum(d["prediction"]["predicted_step"] < d["reference_step"] for d in errors),
            "late": sum(d["prediction"]["predicted_step"] > d["reference_step"] for d in errors),
        },
        "near_errors_within_5": sum(d["step_within_5"] for d in errors),
        "joint": dict(collections.Counter(
            f'role_{bool(d["role_correct"])}_root_{bool(d["step_exact"])}' for d in details)),
        "by_source": {},
        "role_confusions": [
            {"reference": pair[0], "prediction": pair[1], "n": n}
            for pair, n in collections.Counter(
                (normalize_role(d["reference_role"]), normalize_role(d["prediction"]["predicted_role"]))
                for d in details if not d["role_correct"]
            ).most_common(20)
        ],
    }
    for source in sorted({d["source"] for d in details}):
        rows = [d for d in details if d["source"] == source]
        stats["by_source"][source] = {
            "n": len(rows), "stage_counts": dict(collections.Counter(stage(d) for d in rows)),
            "role_correct": sum(d["role_correct"] for d in rows),
            "step_exact": sum(d["step_exact"] for d in rows),
            "step_within_5": sum(d["step_within_5"] for d in rows),
        }
    # Preserve exact reduction decisions for manually discussed cases.
    stats["case_reduction_trace"] = {}
    for qid, gold in [("swe_bench_pro__026", 31), ("terminal_bench_2__016", 398)]:
        trace = []
        for path in sorted((ROOT / "results/full/calls" / qid).glob("reduce_*.json")):
            call = json.loads(path.read_text())
            options = [int(s) for s in call["request"]["questions"]["root_step"]["criteria"]]
            if gold in options:
                answer = call["response"]["answers"]["root_step"]
                trace.append({"log": str(path.relative_to(ROOT)), "gold": gold,
                              "kept": top_steps(answer, options, 3), "answer": answer})
        stats["case_reduction_trace"][qid] = trace
    assert sum(counts.values()) == len(details)
    assert stats["gold_unavailable_before_final"] + stats["gold_available_but_wrong"] == len(errors)
    reports = ROOT / "reports"
    (reports / "jev_error_analysis.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    fields = ["question_ID", "source", "stage", "reference_step", "predicted_step", "reference_role",
              "predicted_role", "role_correct", "step_exact", "step_within_5", "absolute_step_error",
              "steps", "method", "step_confidence", "call_log_directory"]
    with (reports / "jev_error_analysis.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in details:
            row = {key: d[key] for key in fields if key in d}
            row.update({key: d["prediction"][key] for key in ("predicted_step", "predicted_role", "step_confidence")})
            row.update(stage=stage(d), call_log_directory=f'results/full/calls/{d["question_ID"]}')
            writer.writerow(row)
    print(json.dumps({k: stats[k] for k in ("n", "root_errors", "stage_counts", "segmented_funnel", "joint")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
