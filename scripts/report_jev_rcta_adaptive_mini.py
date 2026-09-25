"""Read-only audit of raw adaptive calls plus a compact, matched Mini report."""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

import evaluate as ev
import evaluate_jev_rcta_adaptive as runner
import jev_rcta_adaptive as method
from run_jev_rcta_adaptive_mini import accounting


def audited_predictions(out, rows):
    frozen = json.loads((out / "config.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        source = out / "frozen_sources" / name
        if not source.exists():
            source = ev.ROOT / "scripts" / name
        if ev.sha(source.read_bytes()) != digest:
            raise ValueError("Frozen inference source changed: " + name)
    by_id = {r["question_ID"]: r for r in rows}
    calls = collections.defaultdict(list)
    for path in sorted((out / "calls").glob("*/*.json")):
        saved = json.loads(path.read_text())
        request = saved["request"]
        if (ev.sha(ev.dumps(request).encode()) != saved["request_sha256"] or
                request["model"] != frozen["model"] or saved["response"]["model"] != frozen["model"] or
                saved.get("endpoint", ev.ENDPOINT) != frozen["endpoint"]):
            raise ValueError("Raw request/model/endpoint mismatch: " + str(path))
        calls[path.parent.name].append(saved)
    predictions = {}
    for path in sorted((out / "predictions").glob("*.json")):
        p = json.loads(path.read_text())
        qid = path.stem
        row = by_id[qid]
        if (p.get("question_ID") != qid or p.get("status") != "ok" or p.get("method") != method.PROTOCOL or
                p["history_sha256"] != ev.sha(ev.dumps([ev.record(h) for h in row["history"]]).encode())):
            raise ValueError("Prediction provenance mismatch: " + qid)
        case_calls = calls[qid]
        if p["api_calls"] != len(case_calls) or p["budget"]["logical_calls"] != len(case_calls):
            raise ValueError("Call count mismatch: " + qid)
        request_bytes = sum(len(ev.dumps(c["request"]).encode()) for c in case_calls)
        if request_bytes != p["budget"]["request_bytes"]:
            raise ValueError("Request-byte accounting mismatch: " + qid)
        if any(len(ev.dumps(c["request"]).encode()) > frozen["config"]["request_bytes"] for c in case_calls):
            raise ValueError("Per-request budget exceeded: " + qid)
        for field in ("input_tokens", "output_tokens"):
            total = sum(c["response"].get("usage", {}).get(field, 0) for c in case_calls)
            if total != p[field] or total != p["budget"][field]:
                raise ValueError("Token accounting mismatch: " + qid)
        if p["predicted_step"] is not None:
            if p["decision_status"] != "supported" or not p.get("evidence_refs"):
                raise ValueError("Unverified step reported as a prediction: " + qid)
            for ref in p["evidence_refs"]:
                raw = row["history"][ref["step"]]["content"].encode()[ref["start_byte"]:ref["end_byte"]]
                if ref["step"] != p["predicted_step"] or ev.sha(raw) != ref["sha256"]:
                    raise ValueError("Original evidence pointer mismatch: " + qid)
        predictions[qid] = p
    return predictions, calls, frozen


def metrics(rows, predictions):
    return ev.metrics([ev.score_row(row, predictions.get(row["question_ID"], {})) for row in rows])


def pct(value):
    return "—" if value is None else "%.2f%%" % (value * 100)


def diagnostics(predictions, calls):
    """Describe saved execution only; never rerun or relax the inference gates."""
    cases = {}
    for qid, p in predictions.items():
        nodes = p["search_nodes"]
        chosen = next((n for n in nodes if n["step"] == p.get("tentative_step")), None)
        required = ("inspect", "upstream", "repair", "challenge")
        cases[qid] = {
            "decision_reason": p["decision_reason"],
            "complete_scan": p["scanned_segments"] == p["segments"],
            "initial_candidates": len(p["recall_candidates"]),
            "search_nodes": len(nodes),
            "fully_checked_nodes": sum(all(a in n["done"] for a in required) for n in nodes),
            "search_actions": dict(collections.Counter(t["action"] for t in p["search_trace"] if "round" in t)),
            "tentative_missing_checks": [a for a in required if a not in chosen["done"]] if chosen else None,
            "scan_input_tokens": sum(c["response"].get("usage", {}).get("input_tokens", 0)
                                     for c in calls[qid] if c["tag"].endswith("_scan")),
            "input_tokens": p["input_tokens"], "api_calls": p["api_calls"],
        }
    by_reason = {}
    for reason in sorted({p["decision_reason"] for p in predictions.values()}):
        group = [d for d in cases.values() if d["decision_reason"] == reason]
        by_reason[reason] = {
            "n": len(group),
            "mean_calls": statistics.mean(d["api_calls"] for d in group),
            "mean_input_tokens": statistics.mean(d["input_tokens"] for d in group),
            "mean_initial_candidates": statistics.mean(d["initial_candidates"] for d in group),
            "mean_fully_checked_nodes": statistics.mean(d["fully_checked_nodes"] for d in group),
            "incomplete_scan_n": sum(not d["complete_scan"] for d in group),
        }
    return {"by_reason": by_reason, "per_case": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stem", default="jev_rcta_adaptive_mini_20260924")
    args = parser.parse_args()
    if Path(args.stem).name != args.stem:
        raise ValueError("Report stem must be a filename")
    out = ev.ROOT / args.output
    rows, manifest = ev.load_data()
    predictions, calls, frozen = audited_predictions(out, rows)
    progress = json.loads((out / "progress.json").read_text())
    completed = [r for r in rows if r["question_ID"] in predictions]
    baselines = {}
    for name, directory in (("JEV baseline", "phase1"), ("JEV-RCTA v2", "jev_rcta_mini_20260923")):
        values = {}
        for row in completed:
            path = ev.ROOT / "results" / directory / "predictions" / (row["question_ID"] + ".json")
            if path.exists():
                values[row["question_ID"]] = json.loads(path.read_text())
        if len(values) == len(completed):
            baselines[name] = metrics(completed, values)
    totals = accounting(out)
    decisions = collections.Counter(p["decision_reason"] for p in predictions.values())
    stops = collections.Counter(p["stop_reason"] for p in predictions.values())
    phases = collections.Counter(c["tag"].split("_", 1)[1] for qid, case in calls.items()
                                 if qid != "_preflight" for c in case)
    candidate = {field: sum(r["mistake_step"] in predictions[r["question_ID"]][field] for r in completed) / len(completed)
                 if completed else None for field in ("recall_candidates", "expanded_candidates", "final_candidates")}
    tentative = {qid: {"predicted_step": p.get("tentative_step"), "predicted_role": p.get("tentative_role")}
                 for qid, p in predictions.items()}
    answered = [r for r in completed if predictions[r["question_ID"]]["predicted_step"] is not None]
    report = {"protocol": method.PROTOCOL, "status": progress["status"], "expected_n": len(rows),
              "completed_n": len(completed), "supported_n": len(answered), "abstentions": len(completed) - len(answered),
              "primary_metrics_completed": metrics(completed, predictions),
              "all_200_missing_and_abstained_counted_wrong": metrics(rows, predictions),
              "answered_subset_metrics": metrics(answered, predictions),
              "tentative_diagnostic_only": metrics(completed, tentative),
              "matched_baselines": baselines, "candidate_recall": candidate,
              "decision_reasons": dict(decisions), "stop_reasons": dict(stops), "calls_by_phase": dict(phases),
              "execution_diagnostics": diagnostics(predictions, calls),
              "accounting": totals,
              "by_source": {source: metrics([r for r in completed if r["source"] == source], predictions)
                            for source in sorted({r["source"] for r in rows})},
              "audit": {"frozen_sources": "passed", "raw_request_hashes": "passed", "model_and_endpoint": "passed",
                        "prediction_history_hashes": "passed", "token_and_call_counts": "passed", "evidence_pointers": "passed"},
              "note": "Defaults are uncalibrated and unchanged. Tentative predictions are diagnostics, not official outputs. "
                      "A partial run does not establish full Mini performance."}
    reports = ev.ROOT / "reports"
    runner.write_json(reports / (args.stem + "_metrics.json"), report)
    with (reports / (args.stem + "_predictions.csv")).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_ID", "source", "completed", "reference_step", "reference_role", "predicted_step",
                         "predicted_role", "decision_reason", "tentative_step", "tentative_role", "api_calls", "input_tokens", "output_tokens"])
        for row in rows:
            p = predictions.get(row["question_ID"], {})
            writer.writerow([row["question_ID"], row["source"], bool(p), row["mistake_step"], row["mistake_agent"],
                             *[p.get(k) for k in ("predicted_step", "predicted_role", "decision_reason", "tentative_step",
                                                  "tentative_role", "api_calls", "input_tokens", "output_tokens")]])
    lines = ["# Adaptive JEV-RCTA Mini 实测", "",
             "协议 `%s`；模型 `%s`；数据版本 `%s`。" % (method.PROTOCOL, frozen["model"], manifest["revision"]), "",
             "状态：**%s**，已完成 **%d/%d** 条，其中正式输出 %d 条、明确弃答 %d 条。" %
             (progress["status"], len(completed), len(rows), len(answered), report["abstentions"]), "",
             "架构、默认阈值及每例预算保持冻结，不使用 Mini 标签校准或边测边调。先完成连接控制题与五来源真实样本预检；其结果复用到本次统计。", "",
             "## 同样本比较", "", "下表分母为已完成的 %d 条，弃答计错；未完成时不能当成完整 Mini 得分。" % len(completed), "",
             "| 方法 | 角色准确率 | 根因 Exact | 根因 ±5 | 有效步骤 MAE |",
             "|---|---:|---:|---:|---:|"]
    for name, score in list(baselines.items()) + [("Adaptive JEV-RCTA", report["primary_metrics_completed"])]:
        mae = score["valid_output_root_mae"]
        lines.append("| %s | %s | %s | %s | %s |" % (name, pct(score["role_correct"]), pct(score["step_exact"]),
                     pct(score["step_within_5"]), "—" if mae is None else "%.2f" % mae))
    lines += ["", "MAE 只覆盖有效步骤输出；弃答较多时不能直接用它与全覆盖方法比较。", "",
              "## 弃答与候选", "", "| 判定原因 | 条数 |", "|---|---:|"]
    lines += ["| `%s` | %d |" % pair for pair in sorted(decisions.items())]
    lines += ["", "弃答表示当前协议未能在预算内形成满足核验条件的正式输出，不表示接口拒绝服务。",
              "`input_token_budget` 是单例累计输入上限，不是账户额度耗尽；`final_context_ambiguity` 表示分组筛选无法继续缩减候选；",
              "`incomplete_evidence_checks` 表示最终候选尚未完成必要检查；`uncertain_final_choice` 表示最终选择或证据支持不足。",
              "这些原因是程序记录的首要终止原因，不能当作彼此独立的因果归因。", "",
              "| 判定原因 | 平均初筛候选数 | 平均完成全部检查的节点数 | 平均调用数 | 平均输入 tokens |",
              "|---|---:|---:|---:|---:|"]
    for reason, d in report["execution_diagnostics"]["by_reason"].items():
        lines.append("| `%s` | %.1f | %.1f | %.1f | %.0f |" %
                     (reason, d["mean_initial_candidates"], d["mean_fully_checked_nodes"],
                      d["mean_calls"], d["mean_input_tokens"]))
    lines += ["", "候选召回：初筛 %s，扩展后 %s，最终集合 %s。" % tuple(pct(candidate[k]) for k in
              ("recall_candidates", "expanded_candidates", "final_candidates")), "",
              "未通过核验的暂定步骤仅作为诊断：Exact %s、±5 %s，不能替代正式成绩。" %
              (pct(report["tentative_diagnostic_only"]["step_exact"]), pct(report["tentative_diagnostic_only"]["step_within_5"])), "",
              "## 用量", "",
              "- 成功返回：%d 次（含连接控制题 %d 次及未完成样本的成功中间调用）。" %
              (totals["successful_responses"], totals["control_responses"]),
              "- 输入：%s tokens；输出：%s tokens。" % (format(totals["input_tokens"], ","), format(totals["output_tokens"], ",")),
              "- 重试日志事件：%d。无响应请求是否计费无法由本地成功响应确定。" % totals["retry_events"],
              "- 返回中未提供完整美元费用时不估算账单，也不推断剩余额度。", "",
              "## 审计", "", "已核对冻结代码、请求摘要、模型/端点、历史摘要、逐例调用与 token 计数，以及正式输出的原文证据指针。", "",
              "运行目录：`%s`。完整指标和逐例对照见同前缀 JSON/CSV。" % out.relative_to(ev.ROOT), ""]
    (reports / (args.stem + "_report.md")).write_text("\n".join(lines))
    print(json.dumps({"status": report["status"], "completed": len(completed), "supported": len(answered),
                      "metrics": report["primary_metrics_completed"], "accounting": totals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
