"""Audit a frozen v3.1 Mini run and report completed-only and full-set metrics."""
import argparse
import collections
import contextlib
import csv
import io
import json
from pathlib import Path
import sys

import evaluate as ev
from evaluate_jev_rcta_adaptive import write_json
from report_jev_rcta_adaptive_v2 import main as audit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stem", default="jev_rcta_adaptive_v31_mini_20260924")
    args = parser.parse_args()
    if Path(args.stem).name != args.stem:
        raise ValueError("Report stem must be a filename")
    out = ev.ROOT / args.output
    rows, manifest = ev.load_data()
    config = json.loads((out / "config.json").read_text())
    if (config["protocol"] != "jev-rcta-adaptive-v3.1" or config["mode"] != "live"
            or len(config["selection"]) != len(rows)
            or set(config["selection"]) != {r["question_ID"] for r in rows}
            or config["manifest_sha256"] != ev.sha(ev.dumps(manifest).encode())):
        raise ValueError("Expected a frozen live run of the entire official Mini")
    # Reuse the source, request, history, usage and original-pointer auditor.
    previous_argv = sys.argv
    try:
        sys.argv = [__file__, "--output", args.output, "--stem", args.stem]
        with contextlib.redirect_stdout(io.StringIO()):
            audit_report(protocol=config["protocol"], baseline_sets=(
                ("JEV baseline", ("phase1",)),
                ("Historical JEV-RCTA", ("jev_rcta_mini_20260923",))))
    finally:
        sys.argv = previous_argv
    reports = ev.ROOT / "reports"
    metric_path = reports / (args.stem + "_metrics.json")
    facts = json.loads(metric_path.read_text())
    predictions = {p.stem: json.loads(p.read_text()) for p in (out / "predictions").glob("*.json")}
    progress = json.loads((out / "progress.json").read_text())
    if (facts["n"] != len(predictions) or progress["completed_n"] != len(predictions)
            or set(progress["completed_ids"]) != set(predictions)):
        raise ValueError("Progress changed during audit; run the report again when the process is stopped")
    completed = [r for r in rows if r["question_ID"] in predictions]
    scored = {r["question_ID"]: ev.score_row(r, predictions.get(r["question_ID"], {})) for r in rows}
    by_source = {source: ev.metrics([scored[r["question_ID"]] for r in completed if r["source"] == source])
                 for source in sorted({r["source"] for r in rows})}
    facts.update(scope="Official 200-case Mini; includes previously inspected development cases, not an unseen holdout",
                 status=progress["status"], expected_n=len(rows), completed_n=len(completed),
                 abstentions=len(completed) - facts["answered_n"],
                 all_200_missing_and_abstained_counted_wrong=ev.metrics(list(scored.values())),
                 by_source=by_source,
                 decision_reasons=dict(collections.Counter(p["decision_reason"] for p in predictions.values())),
                 stop_reasons=dict(collections.Counter(p["stop_reason"] for p in predictions.values())))
    if facts["status"] == "complete" and len(completed) != len(rows):
        raise ValueError("Complete run is missing predictions")
    write_json(metric_path, facts)
    with (reports / (args.stem + "_predictions.csv")).open("w", newline="") as handle:
        writer = csv.writer(handle)
        fields = ("predicted_step", "predicted_role", "verified_step", "verified_role", "decision_status",
                  "decision_reason", "api_calls", "input_tokens", "output_tokens")
        writer.writerow(["question_ID", "source", "completed", "reference_step", "reference_role", *fields])
        for row in rows:
            p = predictions.get(row["question_ID"], {})
            writer.writerow([row["question_ID"], row["source"], bool(p), row["mistake_step"], row["mistake_agent"],
                             *[p.get(k) for k in fields]])
    pct = lambda x: "—" if x is None else "%.2f%%" % (100 * x)
    totals = facts["runs"][0]["accounting"]
    lines = ["# JEV-RCTA v3.1 Mini 实测", "",
             "状态：**%s**；完成 **%d/200** 条；点预测 **%d** 条、弃答 **%d** 条、严格核验通过 **%d** 条。" %
             (facts["status"], len(completed), facts["answered_n"], facts["abstentions"], facts["supported_n"]), "",
             "代码、参数、数据与预算冻结。连接检查后先运行五来源短样本，再顺序执行剩余样本；本次未复用以往开发预测。",
             "Mini 包含已查看过的开发样本，并非未见留出集；本次没有根据中途分数调整方法。", "",
             "## 同样本比较", "",
             "以下分母均为本次完成的 %d 条，弃答计错。若未完成 200 条，不代表完整 Mini 成绩。" % len(completed), "",
             "| 方法 | 根因 Exact | 根因 ±5 | 角色准确率 |",
             "|---|---:|---:|---:|"]
    groups = [(name, value["metrics"]) for name, value in facts["matched_comparisons"].items()]
    groups.append(("JEV-RCTA v3.1 点预测（含未充分核验）", facts["metrics"]))
    for name, score in groups:
        lines.append("| %s | %s | %s | %s |" % (name, pct(score["step_exact"]),
                     pct(score["step_within_5"]), pct(score["role_correct"])))
    lines += ["", "历史结果来自不同时间的独立运行，不能据此把差异归因于单一模块。", "",
              "## 分来源", "", "| 来源 | 完成数 | 根因 Exact | 根因 ±5 | 角色准确率 |",
              "|---|---:|---:|---:|---:|"]
    for source, score in by_source.items():
        lines.append("| %s | %d/40 | %s | %s | %s |" % (source, score["n"], pct(score["step_exact"]),
                     pct(score["step_within_5"]), pct(score["role_correct"])))
    lines += ["", "## 用量与判定", "",
              "成功响应 **%d** 次（含 %d 次连接检查及未完成样本中的成功中间请求）；输入 **%s** tokens，输出 **%s** tokens。" %
              (totals["successful_responses"], totals["control_responses"], format(totals["input_tokens"], ","),
               format(totals["output_tokens"], ",")), "",
              "重试事件 %d 次。未完整返回美元费用，不能从本地记录推断账单或余额；无响应请求是否计费未知。" % totals["retry_events"], "",
              "点预测与严格核验分别统计。预算耗尽不是账户余额耗尽；核验未通过也不等于接口调用失败。", "",
              "| 判定原因 | 数量 |", "|---|---:|"]
    lines += ["| `%s` | %d |" % pair for pair in sorted(facts["decision_reasons"].items())]
    lines += ["", "冻结源码、请求摘要、模型/端点、原始轨迹摘要、逐例请求数/token 数与证据位置均已核验。",
              "JSON 同时记录已完成子集与未完成/弃答均计错的 200 条分母；CSV 包含所有 200 条及完成标记。", ""]
    (reports / (args.stem + "_report.md")).write_text("\n".join(lines))
    print(json.dumps({k: facts[k] for k in ("status", "completed_n", "answered_n", "supported_n", "metrics")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
