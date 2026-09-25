"""Bounded Mini scheduling around the unchanged adaptive inference protocol.

One connectivity control and one shortest trajectory per source gate the batch.
Successful predictions, including explicit abstentions, satisfy the execution
gate; reference accuracy never controls scheduling or inference settings.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import threading
import time

import evaluate as ev
import evaluate_jev_rcta_adaptive as runner
import jev_rcta_adaptive as method
from jev_rcta_client import BalanceExhausted, Client


def run_case(row, out, key, stop):
    qid = row["question_ID"]
    path = out / "predictions" / (qid + ".json")
    digest = ev.sha(ev.dumps([ev.record(h) for h in row["history"]]).encode())
    if path.exists():
        result = json.loads(path.read_text())
        if (result.get("history_sha256") != digest or result.get("status") != "ok" or
                result.get("method") != method.PROTOCOL or result.get("question_ID") != qid):
            raise ValueError("Cached prediction mismatch")
        return result
    client = Client(key, out / "calls" / qid, out, stop_event=stop)
    started = time.monotonic()
    try:
        result = method.predict(row["history"], client)
        result.update(status="ok", question_ID=qid, api_calls=len(client.calls),
                      input_tokens=result["budget"]["input_tokens"], output_tokens=result["budget"]["output_tokens"],
                      elapsed_seconds=time.monotonic() - started)
        runner.write_json(path, result)
        return result
    except Exception as error:
        stop.set()
        runner.write_json(out / "errors" / (qid + ".json"), {
            "question_ID": qid, "error_type": type(error).__name__,
            "error": str(error).replace(key, "[REDACTED]"), "successful_calls": len(client.calls)})
        raise


def accounting(out):
    result = {"successful_responses": 0, "input_tokens": 0, "output_tokens": 0,
              "successful_request_network_attempts": 0, "control_responses": 0,
              "control_input_tokens": 0, "control_output_tokens": 0,
              "reported_cost_usd": None, "retry_events": 0}
    costs = []
    for path in sorted((out / "calls").glob("*/*.json")):
        call = json.loads(path.read_text())
        usage = call["response"].get("usage", {})
        result["successful_responses"] += 1
        result["successful_request_network_attempts"] += call.get("attempts", 1)
        for field in ("input_tokens", "output_tokens"):
            result[field] += usage.get(field, 0)
        if path.parent.name == "_preflight":
            result["control_responses"] += 1
            for field in ("input_tokens", "output_tokens"):
                result["control_" + field] += usage.get(field, 0)
        if "cost_usd" in usage:
            costs.append(usage["cost_usd"])
    if costs and len(costs) == result["successful_responses"]:
        result["reported_cost_usd"] = sum(costs)
    path = out / "transport_retries.jsonl"
    if path.exists():
        result["retry_events"] = sum(bool(line.strip()) for line in path.read_text().splitlines())
    result["note"] = "Includes controls and successful calls in unfinished cases. "
    result["note"] += "Unknown-response failures may be charged; no price or account balance inferred."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1..4 workers")
    if (ev.ROOT / args.output / "user_stop.json").exists():
        raise SystemExit("Run explicitly stopped by user; no API requests issued")
    rows, manifest = ev.load_data()
    out = ev.ROOT / args.output
    frozen = runner.prepare(out, manifest, method.Config(), [r["question_ID"] for r in rows], "live")
    if (out / "balance_stop.json").exists():
        raise SystemExit("Persistent quota stop; no API requests issued")
    execution_path = out / "execution.json"
    code_hash = ev.sha(Path(__file__).read_bytes())
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else {
        "started_unix": time.time(), "scheduler_sha256": code_hash, "attempts": []}
    if execution["scheduler_sha256"] != code_hash:
        raise ValueError("Scheduler changed; preserve the existing run")
    execution["attempts"].append({"started_unix": time.time(), "workers": args.workers,
                                  "preflight_only": args.preflight_only})
    runner.write_json(execution_path, execution)
    key, stop = runner.get_key(), threading.Event()
    predictions = {r["question_ID"]: run_case(r, out, key, stop) for r in rows
                   if (out / "predictions" / (r["question_ID"] + ".json")).is_file()}
    failures = []
    smoke = [min((r for r in rows if r["source"] == source),
                 key=lambda r: (len(ev.dumps(r["history"]).encode()), r["question_ID"]))
             for source in sorted({r["source"] for r in rows})]

    def progress(status="running", phase="mini"):
        runner.write_json(out / "progress.json", {
            "status": status, "phase": phase, "requested_n": len(rows), "completed_n": len(predictions),
            "completed_ids": sorted(predictions), "failed_ids": failures,
            "abstentions": sum(p["predicted_step"] is None for p in predictions.values()),
            "updated_unix": time.time()})

    def accept(row, result):
        predictions[row["question_ID"]] = result
        print("MINI %d/%d: %s %s (%s)" % (len(predictions), len(rows), row["question_ID"],
              result["decision_status"], result["decision_reason"]), flush=True)

    try:
        runner.preflight(key, out)
        for row in smoke:
            accept(row, run_case(row, out, key, stop))
            progress(phase="preflight")
        runner.write_json(out / "sample_preflight.json", {
            "status": "passed", "question_ids": [r["question_ID"] for r in smoke],
            "gate": "Execution and response validity only; no accuracy or threshold tuning"})
        print("FIVE-SOURCE PREFLIGHT PASSED", flush=True)
        if not args.preflight_only:
            remaining = iter(r for r in rows if r["question_ID"] not in predictions)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {}
                def fill():
                    while len(pending) < args.workers and not stop.is_set():
                        row = next(remaining, None)
                        if row is None:
                            return
                        pending[pool.submit(run_case, row, out, key, stop)] = row
                fill()
                while pending:
                    done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        row = pending.pop(future)
                        try:
                            accept(row, future.result())
                        except Exception:
                            stop.set()
                            failures.append(row["question_ID"])
                    progress()
                    fill()
    except Exception as error:
        stop.set()
        qid = row["question_ID"] if "row" in locals() else "_preflight"
        failures.append(qid)
        runner.write_json(out / "errors" / (qid + ".json"), {
            "error_type": type(error).__name__, "error": str(error).replace(key, "[REDACTED]")})
    status = "error" if stop.is_set() else "preflight_complete" if args.preflight_only else "complete"
    if (out / "balance_stop.json").exists():
        status = "paused_insufficient_balance"
    progress(status)
    account = accounting(out)
    runner.write_json(out / "accounting.json", account)
    small = {qid: {k: p[k] for k in ("predicted_step", "predicted_role", "method", "api_calls", "input_tokens")}
             for qid, p in predictions.items()}
    summary = ev.summarize(rows, small, manifest)
    summary.pop("estimated_input_cost_usd", None)
    summary.update(protocol=frozen, status=status, expected_n=len(rows), accounting=account,
                   cost_note=account["note"], abstentions=sum(p["predicted_step"] is None for p in predictions.values()))
    summary["candidate_recall_completed"] = {
        field: sum(r["mistake_step"] in predictions[r["question_ID"]][field]
                   for r in rows if r["question_ID"] in predictions) / len(predictions) if predictions else None
        for field in ("recall_candidates", "expanded_candidates", "final_candidates")}
    runner.write_json(out / "summary.json", summary)
    execution["attempts"][-1].update(ended_unix=time.time(), status=status)
    execution.update(status=status, updated_unix=time.time())
    runner.write_json(execution_path, execution)
    print(json.dumps({"status": status, "completed": len(predictions), "accounting": account}), flush=True)
    if stop.is_set():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
