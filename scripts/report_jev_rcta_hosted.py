"""Replay saved decisions offline, audit provenance, and report hosted evaluation."""
import collections
import argparse
import csv
import json
from pathlib import Path

import evaluate as ev
from evaluate_full import load_full
from evaluate_jev_rcta import write_json
import jev_rcta as method
from probe_jev_hosted import ENDPOINT
from run_jev_rcta_hosted_full import accounting

OUT = ev.ROOT / "results/jev_rcta_hosted_full"


class Replay:
    def __init__(self, directory):
        self.directory, self.tags = directory, []

    def call(self, state, questions, tag):
        payload = {"model": ev.MODEL, "state": state, "questions": questions}
        saved = json.loads((self.directory / (tag + ".json")).read_text())
        digest = ev.sha(ev.dumps(payload).encode())
        if saved["request_sha256"] != digest or ev.sha(ev.dumps(saved["request"]).encode()) != digest:
            raise ValueError("Replay request differs from saved evidence: " + tag)
        if saved.get("endpoint") != ENDPOINT or saved["response"].get("model") != ev.MODEL:
            raise ValueError("Endpoint or returned model mismatch")
        self.tags.append(tag)
        return saved["response"]


def percent(value):
    return "—" if value is None else "%.2f%%" % (100 * value)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/jev_rcta_hosted_full_v2")
    args = parser.parse_args()
    OUT = ev.ROOT / args.output
    rows, manifest = load_full()
    config = json.loads((OUT / "config.json").read_text())
    for name, digest in config["source_sha256"].items():
        if ev.sha((ev.ROOT / "scripts" / name).read_bytes()) != digest:
            raise ValueError("Frozen source changed: " + name)
    if (OUT / "transport_recovery.json").exists():
        recovery = json.loads((OUT / "transport_recovery.json").read_text())
        if recovery["adapter_sha256"] != ev.sha((ev.ROOT / "scripts/resume_jev_rcta_hosted.py").read_bytes()):
            raise ValueError("Transport recovery adapter changed")
    if config["manifest_sha256"] != ev.sha(ev.dumps(manifest).encode()):
        raise ValueError("Dataset manifest changed")
    completed, predictions, calls, relations = [], {}, 0, collections.Counter()
    for row in rows:
        qid = row["question_ID"]
        path = OUT / "predictions" / (qid + ".json")
        if not path.exists():
            continue
        p = json.loads(path.read_text())
        if p.get("status") != "ok" or p["history_sha256"] != ev.sha(ev.dumps([ev.record(h) for h in row["history"]]).encode()):
            raise ValueError("Prediction/history mismatch: " + qid)
        replay = Replay(OUT / "calls" / qid)
        replayed = method.predict(row["history"], replay)
        if any(p.get(k) != value for k, value in replayed.items()):
            raise ValueError("Saved prediction does not reproduce: " + qid)
        if len(replay.tags) != p["api_calls"]:
            raise ValueError("Call count mismatch: " + qid)
        rebuilt = collections.defaultdict(list)
        for batch in method.segments(row["history"]):
            for fragment in batch:
                rebuilt[fragment["step"]].append(fragment["content"])
        if any("".join(rebuilt[h["step"]]) != h["content"] for h in row["history"]):
            raise ValueError("Incomplete initial evidence coverage")
        calls += len(replay.tags)
        completed.append(row)
        predictions[qid] = p
        relations.update(edge["relation"] for edge in p["relation_hypotheses"])
    status = json.loads((OUT / "progress.json").read_text())["status"]
    stop_record = json.loads((OUT / "balance_stop.json").read_text()) if (OUT / "balance_stop.json").exists() else None
    if stop_record and stop_record.get("status") == "paused_balance_unverifiable":
        status = "paused_balance_unverifiable"
    account = accounting(OUT)
    audit = {"status": status, "expected_n": len(rows), "completed_n": len(completed),
             "replayed_successful_calls": calls, "prediction_replay": "pass",
             "initial_text_coverage": "lossless for all completed cases", "frozen_sources": "pass",
             "relation_counts": dict(relations), "accounting": account}
    if (OUT / "data_audit.json").exists():
        audit["full_segmentation_audit"] = json.loads((OUT / "data_audit.json").read_text())
    report_dir = ev.ROOT / "reports"
    write_json(report_dir / "jev_rcta_hosted_audit.json", audit)
    lines = ["# JEV-RCTA 托管端点评测报告", "",
             "状态：**%s**；完成 **%d / %d** 条。" % (status, len(completed), len(rows)), "",
             "模型固定 `jev-1.13.0`，请求发送到用户指定的独立托管端点 `https://jevtypesafeai.com/api/v1/decide`。",
             "先通过两个简单控制问题，再通过五来源各一条真实轨迹的端到端测试，才启动 Full。小样本门槛是输出有效，不以人工答案准确率调参。", ""]
    if len(completed) != len(rows):
        lines += ["**本次未完成 Full；下列指标仅覆盖已完成样本，不是 Full 分数，也不能与论文全量分数直接比较。**", ""]
    if completed:
        summary = ev.summarize(completed, predictions, manifest)
        summary["protocol"] = config
        summary.pop("estimated_input_cost_usd", None)
        summary["cost_note"] = "Use reported hosted costs in accounting, not original direct-provider pricing."
        summary.update(status=status, expected_n=len(rows), coverage=len(completed) / len(rows), accounting=account,
                       metrics_scope="Completed cases only; Full results only when evaluated_n == expected_n")
        summary["candidate_recall"] = {field: sum(r["mistake_step"] in predictions[r["question_ID"]][field]
                for r in completed) / len(completed)
                for field in ("recall_candidates", "trace_seeds", "expanded_candidates", "final_candidates")}
        write_json(report_dir / "jev_rcta_hosted_metrics.json", summary)
        m = summary["overall"]
        lines += ["| 指标 | 已完成样本结果 |", "|---|---:|",
                  "| 责任角色准确率 | %s |" % percent(m["role_correct"]),
                  "| 根因步骤精确准确率 | %s |" % percent(m["step_exact"]),
                  "| 根因步骤 ±5 命中 | %s |" % percent(m["step_within_5"]),
                  "| 来源加权有效根因 MAE | %s |" % m["source_weighted_valid_output_root_mae"], "",
                  "| 来源 | 已完成数 | Full 总数 | 角色准确率 | 根因精确 | ±5 命中 |",
                  "|---|---:|---:|---:|---:|---:|"]
        totals = collections.Counter(r["source"] for r in rows)
        for source, total in sorted(totals.items()):
            s = summary["by_source"].get(source)
            lines.append("| %s | %d | %d | %s | %s | %s |" % (source, s["n"] if s else 0, total,
                         percent(s["role_correct"] if s else None), percent(s["step_exact"] if s else None),
                         percent(s["step_within_5"] if s else None)))
        lines += ["", "候选召回（仅评分时使用参考步骤）：", ""]
        for name, rate in summary["candidate_recall"].items():
            lines.append("- `%s`：%s" % (name, percent(rate)))
        baseline = {}
        for r in completed:
            path = ev.ROOT / "results/full/predictions" / (r["question_ID"] + ".json")
            if path.exists():
                baseline[r["question_ID"]] = json.loads(path.read_text())
        if len(baseline) == len(completed):
            b = ev.summarize(completed, baseline, manifest)["overall"]
            lines += ["", "同一批已完成样本的历史 JEV 基线对照：", "",
                      "| 方法 | 角色准确率 | 根因精确 | ±5 命中 |", "|---|---:|---:|---:|",
                      "| 原 JEV Choice 基线 | %s | %s | %s |" % tuple(percent(b[k]) for k in ("role_correct", "step_exact", "step_within_5")),
                      "| JEV-RCTA | %s | %s | %s |" % tuple(percent(m[k]) for k in ("role_correct", "step_exact", "step_within_5")), "",
                      "基线来自历史保存结果，本次经独立托管端点调用；没有重跑基线或进行显著性检验。部分完成样本存在来源/顺序偏差，不能外推全量表现。"]
        csv_path = report_dir / "jev_rcta_hosted_predictions.csv"
        with csv_path.open("w", newline="") as handle:
            fields = ["question_ID", "source", "predicted_role", "predicted_step", "reference_role", "reference_step",
                      "role_correct", "step_exact", "step_within_5", "api_calls", "input_tokens"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in completed:
                p, score = predictions[r["question_ID"]], ev.score_row(r, predictions[r["question_ID"]])
                writer.writerow({"question_ID": r["question_ID"], "source": r["source"],
                                 "predicted_role": p["predicted_role"], "predicted_step": p["predicted_step"],
                                 "reference_role": r["mistake_agent"], "reference_step": r["mistake_step"],
                                 **{k: score[k] for k in ("role_correct", "step_exact", "step_within_5")},
                                 "api_calls": p["api_calls"], "input_tokens": p["input_tokens"]})
    lines += ["", "## 调用与审计", "",
              "- 成功调用（含小样本测试及未完成轨迹的中间调用）：%d。" % account["successful_calls_including_partial_cases"],
              "- API 返回的输入 token 合计：%s。" % format(account["input_tokens"], ","),
              "- 成功响应报告的费用合计：$%.6f；不含单独接口控制测试。" % account["reported_cost_usd"],
              "- 最近一次成功响应报告余额：%s。" % ("未知（API 返回 null）" if account["latest_reported_balance_usd"] is None else "$%.6f" % account["latest_reported_balance_usd"]),
              "- 已完成预测均通过离线重放：输入摘要、原始响应、输出选择、模型版本与端点一致。",
              "- 初筛主分段原文覆盖无损；后续证据节选不等于完整上下文。",
              "- 关系判断分布：`%s`。关系分类是模型假设，未经因果正确性标注验证。" % json.dumps(dict(relations), ensure_ascii=False),
              "- 日志、响应与预测保存在 `%s`；密钥不在这些输出中。" % OUT.relative_to(ev.ROOT), ""]
    lines += ["费用按成功响应记录汇总。没有收到响应的中断请求可能已在服务端计费，因此该合计不一定等于账户余额减少量，也不是账单。", ""]
    if (OUT / "transport_recovery.json").exists():
        lines += ["运行中曾发生无响应连接中断。恢复入口只增加有界传输重试，方法、模型、端点和请求内容保持冻结；恢复适配器摘要已核验。", ""]
    if "full_segmentation_audit" in audit:
        coverage = audit["full_segmentation_audit"]
        lines += ["另外，全量离线分段审计覆盖 %d 条、%d 段，全部原文可完整还原；该审计不调用 API，也不代表在线推理已完成。" %
                  (coverage["trajectories"], coverage["segments"]), ""]
    if (OUT / "balance_stop.json").exists():
        stop = json.loads((OUT / "balance_stop.json").read_text())
        if stop.get("status") == "paused_balance_unverifiable":
            anomaly = json.loads((OUT / "balance_anomaly.json").read_text())
            lines += ["## 停止原因", "",
                      "**未收到 HTTP 402。** 服务在余额接近零后继续返回 HTTP 200，但 `credits_remaining_usd` 变为 null。控制端发现后主动暂停新请求；已经发出的并发请求随后完成。",
                      "最后一个数值余额为 $%.6f，随后首个余额为 null 的成功响应报告费用 $%.6f。" %
                      (anomaly["last_numeric_balance"][1]["credits_remaining_usd"], anomaly["first_null_balance"][1]["cost_usd"]),
                      "成功响应的费用累计超过初始报告的约 $10 余额，不能据此确认实际账单或仍有可用额度。需要用户核对网站后台余额后才能继续。",
                      "冻结调度器把任何 balance_stop 标记统一归为 paused_insufficient_balance；本报告根据控制端记录准确区分为 paused_balance_unverifiable。",
                      "详见 `balance_stop.json`、`balance_anomaly.json`。所有成功请求与预测均保留。", ""]
        else:
            lines += ["## 停止原因", "", "服务返回额度不足（HTTP %s），已停止新请求和重试。并发中已经发出的请求可能完成。" % stop["http_status"],
                      "停止位置：`%s` / `%s`。保存成功调用，可在用户明确补充额度后续跑。" % (stop["question_ID"], stop["tag"]), ""]
    (report_dir / "jev_rcta_hosted_full_report.md").write_text("\n".join(lines))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
