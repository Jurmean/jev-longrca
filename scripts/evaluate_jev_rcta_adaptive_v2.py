"""Explicit offline/live entry for adaptive-v2; never starts paid work by default."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import time

import evaluate as ev
import evaluate_jev_rcta_adaptive as common
import jev_rcta_adaptive_v2 as method
from jev_rcta_client import BalanceExhausted, Client
from rcta_choice_policy import ChoicePolicy
from rcta_directory import Directory
from rcta_evidence import EvidenceIndex
from run_jev_rcta_adaptive_mini import accounting

SOURCES = common.SOURCES + ("rcta_directory.py", "jev_rcta_adaptive_v2.py",
                            "evaluate_jev_rcta_adaptive_v2.py", "run_jev_rcta_adaptive_mini.py")


def prepare(out, manifest, config, selection, mode, calibration, engine=None, additional_sources=()):
    engine = engine or method
    for marker in ("user_stop.json", "balance_stop.json"):
        if (out / marker).exists():
            raise ValueError("Stopped run cannot be resumed automatically: " + marker)
    frozen = {"protocol": engine.PROTOCOL, "model": ev.MODEL, "endpoint": ev.ENDPOINT,
              "config": asdict(config), "mode": mode, "selection": selection, "calibration": calibration,
              "manifest_sha256": ev.sha(ev.dumps(manifest).encode()),
              "source_sha256": {name: ev.sha((ev.ROOT / "scripts" / name).read_bytes())
                                for name in SOURCES + tuple(additional_sources)}}
    path = out / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("Output belongs to another protocol/code/data/selection; choose a new directory")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Non-empty output directory has no experiment config")
    for name in ("calls", "predictions", "errors"):
        (out / name).mkdir(parents=True, exist_ok=True)
    common.write_json(path, frozen)
    return frozen


def main(engine=None, additional_sources=(), demo_module="rcta_demo"):
    engine = engine or method
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--demo", action="store_true")
    modes.add_argument("--live", action="store_true")
    parser.add_argument("--output", required=True)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--mini", action="store_true")
    select.add_argument("--case-id", action="append")
    parser.add_argument("--config")
    parser.add_argument("--calibration")
    args = parser.parse_args()
    config = engine.Config(**(json.loads(Path(args.config).read_text()) if args.config else {}))
    config.validate()
    calibration = json.loads(Path(args.calibration).read_text()) if args.calibration else None
    policy = ChoicePolicy(config.selection_mass, calibration)
    out = ev.ROOT / args.output
    if args.demo:
        if args.mini or args.case_id:
            parser.error("Synthetic demo cannot select benchmark trajectories")
        demo = importlib.import_module(demo_module)
        prepare(out, {"synthetic": True}, config, ["synthetic-demo"], "demo", calibration, engine, additional_sources)
        policy.check_holdout(["synthetic-demo"])
        client = demo.ScriptedClient()
        result = engine.predict(demo.HISTORY, client, config, calibration)
        result.update(synthetic=True, api_calls=0, purpose="Scripted control flow; not JEV accuracy")
        common.write_json(out / "demo.json", result)
        common.write_json(out / "demo_calls.json", client.calls)
        print(json.dumps({k: result[k] for k in ("synthetic", "api_calls", "predicted_step", "decision_status")}, indent=2))
        return
    rows, manifest = ev.load_data()
    if args.case_id:
        selected = set(args.case_id)
        if not selected <= {r["question_ID"] for r in rows}:
            raise ValueError("Unknown Mini case ID")
        rows = [r for r in rows if r["question_ID"] in selected]
    smoke = [min((r for r in rows if r["source"] == source),
                 key=lambda r: (len(ev.dumps(r["history"]).encode()), r["question_ID"]))
             for source in sorted({r["source"] for r in rows})]
    if not args.mini and not args.case_id:
        rows = smoke
    ids = {r["question_ID"] for r in smoke}
    ordered = smoke + [r for r in rows if r["question_ID"] not in ids]
    policy.check_holdout([r["question_ID"] for r in rows])
    frozen = prepare(out, manifest, config, [r["question_ID"] for r in rows],
                     "audit" if args.audit_only else "live", calibration, engine, additional_sources)
    if args.audit_only:
        details = []
        for row in rows:
            index = EvidenceIndex(row["history"], config.span_bytes, config.segment_bytes, config.segment_records)
            details.append(dict(index.audit(), question_ID=row["question_ID"],
                                directory=Directory(index, config.directory_fanout).audit()))
        result = {"trajectories": len(rows), "api_calls": 0, "primary_text_coverage": "lossless",
                  "all_segments_reachable": True, "details": details,
                  "note": "Index audit only. Does not establish model recall, actual cost, or benchmark accuracy."}
        common.write_json(out / "data_audit.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "details"}))
        return
    predictions, status, current = {}, "running", "_preflight"
    key = common.get_key()
    try:
        common.preflight(key, out)
        for i, row in enumerate(ordered):
            current = row["question_ID"]
            path = out / "predictions" / (current + ".json")
            if path.exists():
                p = json.loads(path.read_text())
                if (p.get("method") != engine.PROTOCOL or p.get("status") != "ok" or p.get("question_ID") != current or
                        p["history_sha256"] != ev.sha(ev.dumps([ev.record(h) for h in row["history"]]).encode())):
                    raise ValueError("Cached prediction mismatch")
            else:
                started = time.monotonic()
                client = Client(key, out / "calls" / current, out)
                p = engine.predict(row["history"], client, config, calibration)
                p.update(status="ok", question_ID=current, api_calls=len(client.calls),
                         input_tokens=p["budget"]["input_tokens"], output_tokens=p["budget"]["output_tokens"],
                         elapsed_seconds=time.monotonic() - started)
                common.write_json(path, p)
            predictions[current] = p
            common.write_json(out / "progress.json", {"status": status, "completed_n": len(predictions),
                              "requested_n": len(rows), "completed_ids": list(predictions)})
            print("%d/%d: %s (%s)" % (len(predictions), len(rows), current, p["decision_status"]), flush=True)
            # Connectivity alone is insufficient for scaling. This gate uses no
            # reference labels and never relaxes confidence/budgets on the fly.
            if args.mini and i + 1 == len(smoke) and not any(p["predicted_step"] is not None for p in predictions.values()):
                status = "stopped_no_usable_predictions"
                break
        else:
            status = "complete"
    except KeyboardInterrupt:
        status = "stopped_by_user"
        common.write_json(out / "user_stop.json", {"status": status, "stopped_unix": time.time()})
    except Exception as error:
        status = "paused_insufficient_balance" if isinstance(error, BalanceExhausted) else "error"
        common.write_json(out / "errors" / (current + ".json"), {
            "error_type": type(error).__name__, "error": str(error).replace(key, "[REDACTED]")})
    common.write_json(out / "progress.json", {"status": status, "completed_n": len(predictions),
                      "requested_n": len(rows), "completed_ids": list(predictions)})
    thin = {qid: {k: p[k] for k in ("predicted_step", "predicted_role", "api_calls", "input_tokens", "method")}
            for qid, p in predictions.items()}
    summary = ev.summarize(rows, thin, manifest)
    summary.pop("estimated_input_cost_usd", None)
    verified = {qid: {"predicted_step": p["verified_step"], "predicted_role": p["verified_role"]}
                for qid, p in predictions.items()}
    summary.update(status=status, protocol=frozen, completed_n=len(predictions),
                   output_policy="Point predictions include best_effort; supported subset reported separately.",
                   abstentions=sum(p["predicted_step"] is None for p in predictions.values()),
                   supported_n=sum(p["verified_step"] is not None for p in predictions.values()),
                   verified_metrics_all_selected=ev.metrics([ev.score_row(r, verified.get(r["question_ID"], {})) for r in rows]),
                   accounting=accounting(out))
    common.write_json(out / "summary.json", summary)
    print(json.dumps({"status": status, "completed_n": len(predictions), "metrics": summary["overall"]}, indent=2))
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
