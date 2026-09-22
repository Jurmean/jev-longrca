"""Audit full trace coverage, authentic outputs, and aggregate accounting offline."""
import argparse
import json
from pathlib import Path

from evaluate import ROOT, MODEL, load_data, record, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full:
        from evaluate_full import load_full
        rows, _ = load_full()
    else:
        rows, _ = load_data()
    run = ROOT / ("results/full" if args.full else "results/phase1")
    config = json.loads((run / "config.json").read_text())
    assert config["runner_sha256"] == sha((ROOT / "scripts/evaluate.py").read_bytes())
    if args.full:
        assert config["orchestrator_sha256"] == sha((ROOT / "scripts/evaluate_full.py").read_bytes())
    key = None
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip()
    totals = {"trajectories": 0, "calls": 0, "input_tokens": 0, "output_tokens": 0,
              "history_records_covered": 0}
    for row in rows:
        qid = row["question_ID"]
        p = json.loads((run / "predictions" / (qid + ".json")).read_text())
        assert p["status"] == "ok"
        calls = {}
        for path in sorted((run / "calls" / qid).glob("*.json")):
            raw = path.read_text()
            assert not key or key not in raw, "Secret found in saved call"
            call = json.loads(raw)
            request, response = call["request"], call["response"]
            assert request["model"] == response["model"] == MODEL
            assert not ({"mistake_step", "mistake_agent", "mistake_reason"} & set(request["state"]))
            for name, q in request["questions"].items():
                assert 1 <= len(q["criteria"]) <= 255
                assert response["answers"][name]["choice"] in q["criteria"]
            totals["input_tokens"] += response["usage"]["input_tokens"]
            totals["output_tokens"] += response["usage"]["output_tokens"]
            totals["calls"] += 1
            calls[path.stem] = call
        assert len(calls) == p["api_calls"]
        assert sum(c["response"]["usage"]["input_tokens"] for c in calls.values()) == p["input_tokens"]
        if p["method"] == "full_context":
            assert calls["direct"]["request"]["state"]["history"] == [record(h) for h in row["history"]]
            final = calls["direct"]
        else:
            reconstructed = {h["step"]: [] for h in row["history"]}
            for tag in sorted(calls):
                if tag.startswith("recall_"):
                    for h in calls[tag]["request"]["state"]["segment"]:
                        reconstructed[h["step"]].append(h["content"])
            for h in row["history"]:
                assert "".join(reconstructed[h["step"]]) == h["content"], "Trace text missing: " + qid
            final = calls["final"]
        answers = final["response"]["answers"]
        assert p["predicted_step"] == int(answers["root_step"]["choice"])
        assert p["predicted_role"] == answers["responsible_role"]["choice"]
        totals["trajectories"] += 1
        totals["history_records_covered"] += len(row["history"])
        if totals["trajectories"] % 200 == 0:
            print("Audited %d/%d" % (totals["trajectories"], len(rows)), flush=True)
    summary = json.loads((run / "summary.json").read_text())
    assert totals["trajectories"] == summary["evaluated_n"] == len(rows)
    assert totals["calls"] == summary["successful_api_calls"]
    assert totals["input_tokens"] == summary["input_tokens"]
    totals["checks_passed"] = ["dataset_checksums", "runner_digest", "pinned_response_model", "valid_api_choices",
                               "all_history_text_covered", "predictions_match_raw_responses", "token_accounting",
                               "no_api_key_in_call_artifacts", "no_top_level_reference_labels_in_state"]
    (ROOT / ("reports/full_result_audit.json" if args.full else "reports/result_audit.json")).write_text(json.dumps(totals, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
