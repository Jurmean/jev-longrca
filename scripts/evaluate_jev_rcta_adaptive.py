"""Isolated adaptive pipeline runner: offline audit/demo, explicit live evaluation."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import evaluate as ev
import jev_rcta_adaptive as method
from jev_rcta_client import BalanceExhausted, Client
from rcta_choice_policy import ChoicePolicy, distribution
from rcta_evidence import EvidenceIndex

SOURCES = ("evaluate.py", "jev_rcta_client.py", "rcta_choice_policy.py", "rcta_evidence.py",
           "jev_rcta_adaptive.py", "evaluate_jev_rcta_adaptive.py", "rcta_demo.py")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(out, manifest, config, selection, mode, calibration=None):
    if (out / "user_stop.json").exists():
        raise ValueError("Run explicitly stopped by user; automatic resume is disabled")
    frozen = {"protocol": method.PROTOCOL, "model": ev.MODEL, "endpoint": ev.ENDPOINT,
              "config": asdict(config), "mode": mode, "selection": selection,
              "manifest_sha256": ev.sha(ev.dumps(manifest).encode()), "calibration": calibration,
              "source_sha256": {name: ev.sha((ev.ROOT / "scripts" / name).read_bytes()) for name in SOURCES}}
    path = out / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("Output belongs to another protocol/code/data/selection; choose a new directory")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Non-empty output directory has no experiment config")
    out.mkdir(parents=True, exist_ok=True)
    for directory in ("calls", "predictions", "errors"):
        (out / directory).mkdir(exist_ok=True)
    write_json(path, frozen)
    return frozen


def get_key():
    key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
    if not key and (ev.ROOT / ".env").exists():
        for line in (ev.ROOT / ".env").read_text().splitlines():
            if line.startswith(("TYPESAFE_API_KEY=", "JEV_API_KEY=")):
                key = line.split("=", 1)[1].strip().strip("\"'")
                break
    if not key:
        raise ValueError("Set TYPESAFE_API_KEY or JEV_API_KEY in the environment or private .env")
    return key


def preflight(key, out):
    client = Client(key, out / "calls" / "_preflight", out)
    q = {"check": {"type": "choice", "instructions": "Choose the larger integer.",
                   "criteria": {"three": "3", "seven": "7"}}}
    response = client.call({"purpose": "Connectivity and Choice distribution validation", "integers": [3, 7]}, q, "control")
    answer = distribution(response["answers"].get("check"), q["check"]["criteria"])
    if answer["choice"] != "seven" or answer["warnings"]:
        raise ValueError("Preflight did not return a consistent correct Choice distribution")
    write_json(out / "preflight.json", {"status": "passed", "answer": answer,
                                        "usage": response.get("usage", {}), "successful_calls": 1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true", help="Audit all selected raw text offline")
    mode.add_argument("--demo", action="store_true", help="Scripted synthetic integration example, no API")
    mode.add_argument("--live", action="store_true", help="Call the paid API, starting with a connectivity control")
    parser.add_argument("--output", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--mini", action="store_true", help="Select all 200 Mini trajectories")
    selection.add_argument("--case-id", action="append", help="Select explicit Mini case IDs")
    parser.add_argument("--config", help="JSON object overriding Config defaults")
    parser.add_argument("--calibration", help="Held-out question-level calibration JSON")
    args = parser.parse_args()
    config = method.Config(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    config.validate()
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    policy = ChoicePolicy(config.selection_mass, calibration)
    out = ev.ROOT / args.output
    if args.demo:
        if args.mini or args.case_id:
            parser.error("The synthetic demo cannot select benchmark trajectories")
        from rcta_demo import HISTORY, ScriptedClient
        policy.check_holdout(["synthetic-demo"])
        prepare(out, {"synthetic": True}, config, ["synthetic-demo"], "demo", calibration)
        client = ScriptedClient()
        result = method.predict(HISTORY, client, config, calibration)
        result.update(synthetic=True, api_calls=0, purpose="Scripted control-flow demonstration; not model performance")
        write_json(out / "demo.json", result)
        write_json(out / "demo_calls.json", client.calls)
        print(json.dumps({k: result[k] for k in ("synthetic", "api_calls", "predicted_step", "predicted_role", "decision_status")}, indent=2))
        return
    rows, manifest = ev.load_data()
    if args.case_id:
        ids = set(args.case_id)
        if not ids <= {r["question_ID"] for r in rows}:
            raise ValueError("Unknown Mini case ID")
        rows = [r for r in rows if r["question_ID"] in ids]
    elif not args.mini:
        rows = [min((r for r in rows if r["source"] == s),
                    key=lambda r: (len(ev.dumps(r["history"]).encode()), r["question_ID"]))
                for s in sorted({r["source"] for r in rows})]
    policy.check_holdout([r["question_ID"] for r in rows])
    frozen = prepare(out, manifest, config, [r["question_ID"] for r in rows],
                     "audit" if args.audit_only else "live", calibration)
    if args.audit_only:
        audits = []
        for row in rows:
            index = EvidenceIndex(row["history"], config.span_bytes, config.segment_bytes, config.segment_records)
            audits.append(dict(index.audit(), question_ID=row["question_ID"]))
        audit = {"api_calls": 0, "trajectories": len(audits), "primary_text_coverage": "lossless",
                 "segments": sum(r["segments"] for r in audits),
                 "cases_exceeding_scan_call_reserve": sum(r["segments"] > config.max_calls - 3 for r in audits),
                 "budget_note": "One logical call per segment before search. Other byte/token budgets also apply; "
                                "incomplete scan yields abstention, never a verified prediction.",
                 "details": audits}
        write_json(out / "data_audit.json", audit)
        print(json.dumps({k: v for k, v in audit.items() if k != "details"}))
        return
    key = get_key()
    predictions = {}
    current_id = "_preflight"
    try:
        preflight(key, out)
        for row in rows:
            current_id = row["question_ID"]
            path = out / "predictions" / (current_id + ".json")
            digest = ev.sha(ev.dumps([ev.record(h) for h in row["history"]]).encode())
            if path.exists():
                result = json.loads(path.read_text())
                if (result.get("history_sha256") != digest or result.get("status") != "ok" or
                        result.get("method") != method.PROTOCOL):
                    raise ValueError("Cached prediction mismatch")
            else:
                client = Client(key, out / "calls" / current_id, out)
                started = time.monotonic()
                result = method.predict(row["history"], client, config, calibration)
                result.update(status="ok", question_ID=current_id, api_calls=len(client.calls),
                              input_tokens=result["budget"]["input_tokens"], output_tokens=result["budget"]["output_tokens"],
                              elapsed_seconds=time.monotonic() - started)
                write_json(path, result)
            predictions[current_id] = result
            write_json(out / "progress.json", {"status": "running", "completed_ids": list(predictions), "requested_n": len(rows)})
            print("%d/%d completed: %s" % (len(predictions), len(rows), result["decision_status"]), flush=True)
    except Exception as error:
        write_json(out / "errors" / (current_id + ".json"), {"error": str(error).replace(key, "[REDACTED]")})
        write_json(out / "progress.json", {"status": "paused_insufficient_balance" if isinstance(error, BalanceExhausted) else "error",
                   "failed_id": current_id, "completed_ids": list(predictions), "requested_n": len(rows)})
        raise SystemExit("Run stopped; exact successful calls are saved. See progress.json and errors/.") from None
    summary = ev.summarize(rows, predictions, manifest)
    summary.pop("estimated_input_cost_usd", None)
    summary["cost_note"] = "Token usage only; no unverified price or account-balance estimate. Preflight usage is separate."
    summary["protocol"] = frozen
    summary["abstentions"] = sum(p["predicted_step"] is None for p in predictions.values())
    summary["output_tokens"] = sum(p["output_tokens"] for p in predictions.values())
    summary["candidate_recall"] = {field: sum(r["mistake_step"] in predictions[r["question_ID"]][field] for r in rows) / len(rows)
                                   for field in ("recall_candidates", "expanded_candidates", "final_candidates")}
    write_json(out / "summary.json", summary)
    write_json(out / "progress.json", {"status": "complete", "completed_ids": list(predictions), "requested_n": len(rows)})
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
