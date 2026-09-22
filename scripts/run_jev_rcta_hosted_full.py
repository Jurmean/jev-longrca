"""Five-case preflight gate, then Full 1,140 with bounded concurrency and quota stop."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import threading
import time

import evaluate as ev
from evaluate_full import load_full
from evaluate_jev_rcta import write_json
import jev_rcta as method
from jev_rcta_client import BalanceExhausted, Client
from probe_jev_hosted import ENDPOINT, read_key

SOURCES = ("evaluate.py", "evaluate_full.py", "evaluate_jev_rcta.py", "jev_rcta.py",
           "jev_rcta_client.py", "probe_jev_hosted.py", "run_jev_rcta_hosted_full.py")


def prepare(out, manifest):
    config = dict(method.CONFIG, endpoint=ENDPOINT, expected_n=1140,
                  dataset_revision=manifest["revision"],
                  manifest_sha256=ev.sha(ev.dumps(manifest).encode()),
                  source_sha256={name: ev.sha((ev.ROOT / "scripts" / name).read_bytes()) for name in SOURCES})
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Frozen code/data/endpoint differs; use a new output directory")
    if out.exists() and any(out.iterdir()) and not path.exists():
        raise ValueError("Nonempty output has no protocol config")
    out.mkdir(parents=True, exist_ok=True)
    write_json(path, config)
    for name in ("calls", "predictions", "errors"):
        (out / name).mkdir(exist_ok=True)
    return config


def run_case(row, out, key, stop):
    qid = row["question_ID"]
    history = row["history"]
    history_hash = ev.sha(ev.dumps([ev.record(h) for h in history]).encode())
    path = out / "predictions" / (qid + ".json")
    if path.exists():
        prediction = json.loads(path.read_text())
        if prediction.get("status") != "ok" or prediction.get("history_sha256") != history_hash:
            raise ValueError("Invalid cached prediction: " + qid)
        return prediction
    client = Client(key, out / "calls" / qid, out, endpoint=ENDPOINT, stop_event=stop)
    start = time.monotonic()
    try:
        prediction = method.predict(history, client)
    except Exception as error:
        stop.set()
        write_json(out / "errors" / (qid + ".json"), {"question_ID": qid,
                   "type": type(error).__name__, "error": str(error).replace(key, "[REDACTED]"),
                   "successful_calls": len(client.calls)})
        raise
    prediction.update(status="ok", question_ID=qid, history_sha256=history_hash,
                      elapsed_seconds=time.monotonic() - start, api_calls=len(client.calls),
                      input_tokens=sum(c["response"].get("usage", {}).get("input_tokens", 0) for c in client.calls),
                      reported_cost_usd=sum(c["response"].get("usage", {}).get("cost_usd", 0) for c in client.calls))
    write_json(path, prediction)
    return prediction


def accounting(out):
    calls, tokens, cost = 0, 0, 0.0
    latest_time, balance = -1, None
    for path in (out / "calls").glob("*/*.json"):
        saved = json.loads(path.read_text())
        usage = saved["response"].get("usage", {})
        calls += 1
        tokens += usage.get("input_tokens", 0)
        cost += usage.get("cost_usd", 0)
        if path.stat().st_mtime > latest_time:
            latest_time, balance = path.stat().st_mtime, usage.get("credits_remaining_usd")
    return {"successful_calls_including_partial_cases": calls, "input_tokens": tokens,
            "reported_cost_usd": cost, "latest_reported_balance_usd": balance,
            "note": "Successful responses only; includes preflight and partial cases; separate control probe excluded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/jev_rcta_hosted_full")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("Use 1..8 workers")
    probe = json.loads((ev.ROOT / "results/jev_rcta_hosted_probe/probe.json").read_text())
    if not probe.get("passed") or probe["endpoint"] != ENDPOINT or probe["model"] != ev.MODEL:
        raise SystemExit("Compatibility probe must pass first")
    out = ev.ROOT / args.output
    if (out / "balance_stop.json").exists():
        raise SystemExit("Quota stop preserved; explicit replenishment required before resuming")
    rows, manifest = load_full()
    config = prepare(out, manifest)
    smoke = [min((r for r in rows if r["source"] == source),
                 key=lambda r: (len(ev.dumps(r["history"]).encode()), r["question_ID"]))
             for source in sorted({r["source"] for r in rows})]
    write_json(out / "selection.json", {"preflight_ids": [r["question_ID"] for r in smoke],
                                       "full_ids": [r["question_ID"] for r in rows],
                                       "rule": "One shortest serialized history per source before Full"})
    key, stop = read_key(), threading.Event()
    predictions, failures = {}, []
    timing_path = out / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {"started_unix": time.time(), "attempts": []}
    timing["attempts"].append({"started_unix": time.time(), "workers": args.workers})
    write_json(timing_path, timing)

    def progress(phase, state="running"):
        write_json(out / "progress.json", {"status": state, "phase": phase, "completed_n": len(predictions),
                   "expected_n": len(rows), "failed_ids": failures, "updated_unix": time.time()})

    def accept(row):
        prediction = run_case(row, out, key, stop)
        predictions[row["question_ID"]] = prediction
        return prediction

    progress("preflight")
    for row in smoke:
        try:
            accept(row)
            print("PREFLIGHT %d/5 completed" % len(predictions), flush=True)
            progress("preflight")
        except Exception:
            stop.set()
            failures.append(row["question_ID"])
            break
    if not stop.is_set():
        write_json(out / "preflight.json", {"passed": True, "question_ids": list(predictions),
                   "model": ev.MODEL, "endpoint": ENDPOINT,
                   "note": "Gate uses valid end-to-end predictions, not ground-truth accuracy."})
        print("PREFLIGHT PASSED: %d cases. %s" % (len(predictions),
              "Stopping as requested." if args.preflight_only else "Starting Full 1140."), flush=True)
    if not stop.is_set() and not args.preflight_only:
        remaining = iter(r for r in rows if r["question_ID"] not in predictions)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            def fill():
                while len(pending) < args.workers and not stop.is_set():
                    row = next(remaining, None)
                    if row is None:
                        break
                    pending[pool.submit(run_case, row, out, key, stop)] = row
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    try:
                        predictions[row["question_ID"]] = future.result()
                        print("FULL %d/1140 completed" % len(predictions), flush=True)
                    except Exception:
                        stop.set()
                        failures.append(row["question_ID"])
                progress("full")
                fill()
    balance_stop = (out / "balance_stop.json").exists()
    status = "paused_insufficient_balance" if balance_stop else "error" if stop.is_set() else \
             "preflight_complete" if args.preflight_only else "complete"
    progress("full" if not args.preflight_only else "preflight", status)
    timing["attempts"][-1].update(ended_unix=time.time(), status=status)
    write_json(timing_path, timing)
    account = accounting(out)
    write_json(out / "accounting.json", account)
    completed_rows = [r for r in rows if r["question_ID"] in predictions]
    if completed_rows:
        summary = ev.summarize(completed_rows, predictions, manifest)
        summary["protocol"] = config
        summary.pop("estimated_input_cost_usd", None)
        summary["cost_note"] = "Hosted billing recorded in accounting; original direct-provider price does not apply."
        summary.update(status=status, expected_n=1140, coverage=len(predictions) / 1140,
                       metrics_scope="Completed cases only; full-benchmark metrics only when evaluated_n == 1140",
                       accounting=account)
        summary["candidate_recall"] = {field: sum(r["mistake_step"] in predictions[r["question_ID"]][field]
               for r in completed_rows) / len(completed_rows)
               for field in ("recall_candidates", "trace_seeds", "expanded_candidates", "final_candidates")}
        write_json(out / "summary.json", summary)
    print(json.dumps({"status": status, "completed": len(predictions), "accounting": account}, indent=2))
    if stop.is_set():
        raise SystemExit(2 if balance_stop else 1)


if __name__ == "__main__":
    main()
