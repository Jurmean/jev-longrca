"""Full-split orchestration around the unchanged, frozen mini inference module."""
import argparse
import collections
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import threading
import time

import evaluate as ev

ROOT = ev.ROOT
COUNTS = {"swe_bench_pro": 128, "terminal_bench_2": 42, "travelplanner": 685,
          "vitabench": 108, "webarena_verified": 177}


def load_full():
    manifest = json.loads((ROOT / "data/full_manifest.json").read_text())
    rows = []
    for entry in manifest["files"]:
        raw = (ROOT / entry["path"]).read_bytes()
        if ev.sha(raw) != entry["sha256"]:
            raise ValueError("Full dataset checksum mismatch: " + entry["path"])
        row = json.loads(raw)
        history = row["history"]
        if [h["step"] for h in history] != list(range(len(history))):
            raise ValueError("Non-contiguous steps: " + row["question_ID"])
        if type(row["mistake_step"]) is not int or not 0 <= row["mistake_step"] < len(history):
            raise ValueError("Reference root outside history: " + row["question_ID"])
        if ev.normalize_role(row["mistake_agent"]) not in {ev.normalize_role(r) for r in ev.roles(history)}:
            raise ValueError("Reference role absent: " + row["question_ID"])
        row["source"] = Path(entry["path"]).parent.name
        rows.append(row)
    if len(rows) != 1140 or len({r["question_ID"] for r in rows}) != 1140:
        raise ValueError("Expected 1,140 unique full trajectories")
    if dict(collections.Counter(r["source"] for r in rows)) != COUNTS:
        raise ValueError("Unexpected full source counts")
    return rows, manifest


def prepare(out, manifest):
    mini = ROOT / "results/phase1"
    original = json.loads((mini / "config.json").read_text())
    digest = ev.sha((ROOT / "scripts/evaluate.py").read_bytes())
    if original["runner_sha256"] != digest or any(original[k] != v for k, v in ev.CONFIG.items()):
        raise ValueError("Inference protocol differs from frozen mini run")
    mini_manifest = json.loads((ROOT / "data/manifest.json").read_text())
    full_hashes = {e["question_ID"]: e["sha256"] for e in manifest["files"]}
    if mini_manifest["revision"] != manifest["revision"]:
        raise ValueError("Different dataset revisions")
    reuse_ids = []
    for entry in mini_manifest["files"]:
        qid = entry["question_ID"]
        if full_hashes.get(qid) != entry["sha256"]:
            raise ValueError("Cannot reuse different trajectory: " + qid)
        p = mini / "predictions" / (qid + ".json")
        if json.loads(p.read_text())["status"] != "ok":
            raise ValueError("Cannot reuse failed mini prediction")
        reuse_ids.append(qid)
    config = dict(ev.CONFIG, dataset_revision=manifest["revision"], subset="default", expected_n=1140,
                  runner_sha256=digest, orchestrator_sha256=ev.sha(Path(__file__).read_bytes()),
                  reused_mini_ids=sorted(reuse_ids))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Full output has a different configuration")
    path.write_text(json.dumps(config, indent=2) + "\n")
    for name in ("predictions", "calls", "errors"):
        (out / name).mkdir(exist_ok=True)
    for qid in reuse_ids:
        target = out / "predictions" / (qid + ".json")
        original_prediction = mini / "predictions" / (qid + ".json")
        if target.exists():
            if target.read_bytes() != original_prediction.read_bytes():
                raise ValueError("Reused prediction changed")
        else:
            shutil.copyfile(original_prediction, target)
        target_calls = out / "calls" / qid
        if not target_calls.exists():
            target_calls.symlink_to(os.path.relpath(mini / "calls" / qid, target_calls.parent), target_is_directory=True)
    return reuse_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    rows, manifest = load_full()
    out = ROOT / "results/full"
    reused = prepare(out, manifest)
    print("Validated 1140 trajectories; reused %d byte-identical mini predictions." % len(reused), flush=True)
    if args.audit_only:
        return
    key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
    if not key and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith(("TYPESAFE_API_KEY=", "JEV_API_KEY=")):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit("API key not configured")
    predictions, lock = {}, threading.Lock()
    started = time.monotonic()

    def work(row):
        qid = row["question_ID"]
        path = out / "predictions" / (qid + ".json")
        if path.exists():
            prediction = json.loads(path.read_text())
        else:
            client = ev.Client(key, out / "calls" / qid)
            begin = time.monotonic()
            try:
                prediction = dict(ev.predict(row["history"], client), status="ok")
            except Exception as error:
                prediction = {"status": "error", "error": str(error).replace(key, "[REDACTED]")}
            prediction.update(question_ID=qid, elapsed_seconds=time.monotonic() - begin,
                              api_calls=len(client.calls),
                              input_tokens=sum(c["response"]["usage"]["input_tokens"] for c in client.calls),
                              output_tokens=sum(c["response"]["usage"]["output_tokens"] for c in client.calls))
            target = path if prediction["status"] == "ok" else out / "errors" / (qid + ".json")
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(prediction, indent=2) + "\n")
            temp.replace(target)
        with lock:
            predictions[qid] = prediction
            if len(predictions) % 20 == 0 or prediction["status"] != "ok":
                print("%d/1140 processed; latest=%s status=%s" %
                      (len(predictions), qid, prediction["status"]), flush=True)
        return prediction

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, rows))
    summary = ev.summarize(rows, predictions, manifest)
    summary.update(run_wall_seconds=time.monotonic() - started, reused_mini_n=len(reused),
                   new_trajectory_n=len(rows) - len(reused),
                   failed_n=sum(p["status"] != "ok" for p in predictions.values()))
    summary["reused_input_tokens"] = sum(predictions[qid]["input_tokens"] for qid in reused)
    summary["new_input_tokens"] = summary["input_tokens"] - summary["reused_input_tokens"]
    summary["new_estimated_input_cost_usd"] = summary["new_input_tokens"] * 0.042 / 1000000
    summary["cost_note"] = "Full totals include 200 reused mini cases; new_* excludes their already-paid calls. Public-price estimate, not a bill."
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
    if summary["failed_n"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
