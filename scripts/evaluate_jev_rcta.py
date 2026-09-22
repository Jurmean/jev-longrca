"""Isolated Mini runner for the experimental JEV-RCTA architecture."""
import argparse
import json
import os
from pathlib import Path
import time

import evaluate as ev
import jev_rcta as method
from jev_rcta_client import Client, BalanceExhausted


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(out, manifest):
    config = dict(method.CONFIG, dataset_revision=manifest["revision"],
                  manifest_sha256=ev.sha(ev.dumps(manifest).encode()),
                  source_sha256={name: ev.sha((ev.ROOT / "scripts" / name).read_bytes())
                                 for name in ("evaluate.py", "jev_rcta.py", "evaluate_jev_rcta.py", "jev_rcta_client.py")})
    path = out / "config.json"
    if out.exists() and any(out.iterdir()) and not path.exists():
        raise ValueError("Non-empty output directory has no matching experiment config")
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Output belongs to another protocol/code/data version; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(path, config)
    for directory in ("predictions", "calls", "errors"):
        (out / directory).mkdir(exist_ok=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/jev_rcta_mini")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--smoke", action="store_true", help="Shortest case from each source")
    selection.add_argument("--pilot", action="store_true", help="Median serialized-size case from each source; no label-based selection")
    parser.add_argument("--audit-only", action="store_true", help="Offline data/segmentation audit; no API calls")
    args = parser.parse_args()
    rows, manifest = ev.load_data()
    if args.smoke:
        rows = [min((r for r in rows if r["source"] == source),
                    key=lambda r: len(ev.dumps(r["history"]).encode()))
                for source in sorted({r["source"] for r in rows})]
    elif args.pilot:
        groups = [sorted((r for r in rows if r["source"] == source),
                         key=lambda r: (len(ev.dumps(r["history"]).encode()), r["question_ID"]))
                  for source in sorted({r["source"] for r in rows})]
        rows = [group[len(group) // 2] for group in groups]
    out = ev.ROOT / args.output
    config = prepare(out, manifest)
    selected = {"selection": "smoke" if args.smoke else "pilot" if args.pilot else "mini",
                "question_ids": [r["question_ID"] for r in rows]}
    selection_path = out / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
        raise ValueError("Output directory belongs to a different sample selection")
    write_json(selection_path, selected)
    if args.audit_only:
        total = 0
        for row in rows:
            batches = list(method.segments(row["history"]))
            rebuilt = {h["step"]: [] for h in row["history"]}
            for batch in batches:
                for h in batch:
                    rebuilt[h["step"]].append(h["content"])
            for h in row["history"]:
                if "".join(rebuilt[h["step"]]) != h["content"]:
                    raise ValueError("Segmentation lost original text")
            total += len(batches)
        audit = {"trajectories": len(rows), "segments": total,
                 "primary_text_coverage": "lossless", "api_calls": 0}
        write_json(out / "data_audit.json", audit)
        print(json.dumps(audit))
        return
    key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
    if not key and (ev.ROOT / ".env").exists():
        for line in (ev.ROOT / ".env").read_text().splitlines():
            if line.startswith(("TYPESAFE_API_KEY=", "JEV_API_KEY=")):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit("Set TYPESAFE_API_KEY or JEV_API_KEY")
    predictions = {}
    started = time.monotonic()
    # Sequential execution stops immediately on any failed request, including
    # insufficient balance. Successful calls remain resumable via payload hashes.
    for row in rows:
        qid = row["question_ID"]
        path = out / "predictions" / (qid + ".json")
        history_hash = ev.sha(ev.dumps([ev.record(h) for h in row["history"]]).encode())
        if path.exists():
            prediction = json.loads(path.read_text())
            if prediction.get("history_sha256") != history_hash or prediction.get("status") != "ok":
                raise ValueError("Cached prediction does not match history or successful status")
        else:
            client = Client(key, out / "calls" / qid, out)
            begin = time.monotonic()
            try:
                prediction = method.predict(row["history"], client)
            except Exception as error:
                write_json(out / "errors" / (qid + ".json"),
                           {"question_ID": qid, "error": str(error).replace(key, "[REDACTED]")})
                write_json(out / "progress.json", {"status": "paused_insufficient_balance" if isinstance(error, BalanceExhausted) else "error",
                           "completed_ids": list(predictions), "failed_id": qid, "requested_n": len(rows),
                           "successful_calls_in_failed_case": len(client.calls)})
                raise SystemExit("Run stopped after a failed case; successful calls saved. See errors/.") from None
            prediction.update(status="ok", question_ID=qid, history_sha256=history_hash,
                              elapsed_seconds=time.monotonic() - begin, api_calls=len(client.calls),
                              input_tokens=sum(c["response"].get("usage", {}).get("input_tokens", 0)
                                               for c in client.calls))
            write_json(path, prediction)
        predictions[qid] = prediction
        write_json(out / "progress.json", {"status": "running", "completed_ids": list(predictions),
                                          "requested_n": len(rows)})
        print("%d/%d completed" % (len(predictions), len(rows)), flush=True)
    summary = ev.summarize(rows, predictions, manifest)
    summary["protocol"] = config
    summary["run_wall_seconds"] = time.monotonic() - started
    summary["candidate_recall"] = {
        field: sum(r["mistake_step"] in predictions[r["question_ID"]][field] for r in rows) / len(rows)
        for field in ("recall_candidates", "trace_seeds", "expanded_candidates", "final_candidates")}
    write_json(out / "summary.json", summary)
    write_json(out / "progress.json", {"status": "complete", "completed_ids": list(predictions),
                                      "requested_n": len(rows)})
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
