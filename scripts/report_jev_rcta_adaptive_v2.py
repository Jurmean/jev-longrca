"""Audit saved v2 runs and compare matching examples without making API calls."""
import argparse
import collections
import json
from pathlib import Path
import shutil

import evaluate as ev
from evaluate_jev_rcta_adaptive import write_json
from run_jev_rcta_adaptive_mini import accounting


def score(rows, predictions):
    return ev.metrics([ev.score_row(r, predictions.get(r["question_ID"], {})) for r in rows])


def main(protocol="jev-rcta-adaptive-v2", default_stem="jev_rcta_adaptive_v2_optimization_20260924",
         baseline_sets=None, test_report=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("--stem", default=default_stem)
    parser.add_argument("--protocol", default=protocol)
    args = parser.parse_args()
    protocol = args.protocol
    if Path(args.stem).name != args.stem:
        raise ValueError("Report stem must be a filename")
    rows, _ = ev.load_data()
    by_id = {r["question_ID"]: r for r in rows}
    predictions, runs = {}, []
    for directory in args.output:
        out = ev.ROOT / directory
        config = json.loads((out / "config.json").read_text())
        if config["protocol"] != protocol or config["mode"] != "live":
            raise ValueError("Only real runs of the specified protocol are comparable")
        for name, digest in config["source_sha256"].items():
            archived = out / "frozen_sources" / name
            source = archived if archived.exists() else ev.ROOT / "scripts" / name
            if ev.sha(source.read_bytes()) != digest:
                raise ValueError("Frozen source mismatch: " + name)
            if not archived.exists():
                archived.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, archived)
        calls = collections.defaultdict(list)
        for path in sorted((out / "calls").glob("*/*.json")):
            c = json.loads(path.read_text())
            if (ev.sha(ev.dumps(c["request"]).encode()) != c["request_sha256"] or
                    c["request"]["model"] != config["model"] or c["response"]["model"] != config["model"] or
                    c.get("endpoint", ev.ENDPOINT) != config["endpoint"]):
                raise ValueError("Raw request provenance mismatch")
            calls[path.parent.name].append(c)
        for path in sorted((out / "predictions").glob("*.json")):
            p = json.loads(path.read_text())
            qid, row = path.stem, by_id[path.stem]
            if qid in predictions:
                raise ValueError("Do not count repeated cases as independent examples")
            if (p["question_ID"] != qid or p["status"] != "ok" or p["method"] != config["protocol"] or
                    p["history_sha256"] != ev.sha(ev.dumps([ev.record(h) for h in row["history"]]).encode())):
                raise ValueError("Prediction provenance mismatch")
            if p["api_calls"] != len(calls[qid]) or p["budget"]["logical_calls"] != len(calls[qid]):
                raise ValueError("Call accounting mismatch")
            if sum(len(ev.dumps(c["request"]).encode()) for c in calls[qid]) != p["budget"]["request_bytes"]:
                raise ValueError("Byte accounting mismatch")
            for field in ("input_tokens", "output_tokens"):
                if sum(c["response"].get("usage", {}).get(field, 0) for c in calls[qid]) != p[field]:
                    raise ValueError("Token accounting mismatch")
            for ref in p.get("evidence_refs", []) + p.get("candidate_evidence_refs", []):
                raw = row["history"][ref["step"]]["content"].encode()[ref["start_byte"]:ref["end_byte"]]
                if ev.sha(raw) != ref["sha256"]:
                    raise ValueError("Original evidence pointer mismatch")
            if p["verified_step"] is not None and (p["decision_status"] != "supported" or not p.get("evidence_refs")):
                raise ValueError("Unverified prediction marked supported")
            predictions[qid] = p
        runs.append({"directory": directory, "progress": json.loads((out / "progress.json").read_text()),
                     "accounting": accounting(out)})
    completed = [r for r in rows if r["question_ID"] in predictions]
    comparisons = {}
    if baseline_sets is None:
        baseline_sets = (("JEV baseline", ("phase1",)), ("JEV-RCTA historical", ("jev_rcta_mini_20260923",)),
                         ("Adaptive v1 strict", ("jev_rcta_adaptive_mini_20260924",)))
    for label, directories in baseline_sets:
        values = {r["question_ID"]: json.loads(p.read_text()) for r in completed
                  for directory in directories
                  for p in [ev.ROOT / "results" / directory / "predictions" / (r["question_ID"] + ".json")] if p.exists()}
        if len(values) == len(completed):
            comparisons[label] = dict(metrics=score(completed, values),
                                     calls=sum(p["api_calls"] for p in values.values()),
                                     input_tokens=sum(p["input_tokens"] for p in values.values()))
    verified = {qid: {"predicted_step": p["verified_step"], "predicted_role": p["verified_role"]}
                for qid, p in predictions.items()}
    selected = [r for r in completed if predictions[r["question_ID"]]["predicted_step"] is not None]
    facts = {"protocol": protocol, "scope": "small selected diagnostic sample; not full Mini or unbiased accuracy evidence",
             "runs": runs, "n": len(completed), "answered_n": len(selected),
             "supported_n": sum(p["verified_step"] is not None for p in predictions.values()),
             "metrics": score(completed, predictions), "answered_metrics": score(selected, predictions),
             "verified_metrics_all_cases": score(completed, verified), "matched_comparisons": comparisons,
             "states": dict(collections.Counter(p["decision_status"] for p in predictions.values())),
             "case_calls": sum(p["api_calls"] for p in predictions.values()),
             "case_input_tokens": sum(p["input_tokens"] for p in predictions.values()),
             "case_output_tokens": sum(p["output_tokens"] for p in predictions.values()),
             "audit": "frozen sources, requests, histories, usage and evidence pointers passed",
             "cases": [{"question_ID": r["question_ID"], "reference_step": r["mistake_step"],
                        "reference_role": r["mistake_agent"], **{k: predictions[r["question_ID"]][k] for k in
                         ("predicted_step", "predicted_role", "verified_step", "decision_status", "decision_reason",
                          "api_calls", "input_tokens", "output_tokens", "coverage", "stop_reason")}} for r in completed]}
    write_json(ev.ROOT / "reports" / (args.stem + "_metrics.json"), facts)
    lines = ["# %s 小样本验证" % protocol, "",
             "代码、数据选择与配置按运行目录冻结；点预测与严格核验覆盖分别统计。", "",
             "本次完成 %d 条，点预测 %d 条，严格证据支持 %d 条。点预测包括 best_effort，不能与旧版严格弃答成绩直接等同。" %
             (len(completed), len(selected), facts["supported_n"]), "",
             "| 方法 | 角色准确率 | 根因 Exact | 根因 ±5 | 样本调用数 | 样本输入 tokens |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, values in list(comparisons.items()) + [(protocol + " point prediction", {
            "metrics": facts["metrics"], "calls": facts["case_calls"], "input_tokens": facts["case_input_tokens"]})]:
        m = values["metrics"]
        lines.append("| %s | %.1f%% | %.1f%% | %.1f%% | %d | %s |" %
                     (label, 100*m["role_correct"], 100*m["step_exact"], 100*m["step_within_5"],
                      values["calls"], format(values["input_tokens"], ",")))
    lines += ["", "| 样本 | 参考步骤 | 预测步骤 | 状态 | 调用数 | 输入 tokens | 原文视图覆盖 |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for p in facts["cases"]:
        c = p["coverage"]
        coverage = ("%d 步摘录；涉及 %d/%d 段原文" % (c["excerpt_steps_presented"], c["action_segments_read"], c["total_segments"])) \
            if "excerpt_steps_presented" in c else "%d/%d 段" % (c["read_segments"], c["total_segments"])
        lines.append("| %s | %s | %s | %s | %d | %s | %s |" %
                     (p["question_ID"], p["reference_step"], p["predicted_step"], p["decision_status"],
                      p["api_calls"], format(p["input_tokens"], ","), coverage))
    lines += ["", "以上调用与用量按完成样本对齐；连接题及未完成请求另见 JSON 的 runs/accounting。",
              "样本按来源与长度选择且已反复查看，不是未见测试集。不得据此宣称完整 Mini 准确率提升；实际只读取部分原文，仍有漏检风险。",
              "降低弃答与降低调用量不等同于提高根因定位准确率。未报告美元账单或剩余额度。", "",
              "不同协议的严格核验判据可能不同；没有消融对照时不能将变化归因于某一个组件或论文。", ""]
    if test_report:
        tests = json.loads((ev.ROOT / test_report).read_text())
        lines += ["离线测试：%d 项，%d 项通过、%d 项跳过。" % (tests["run"], tests["passed"], tests["skipped"]), ""]
    (ev.ROOT / "reports" / (args.stem + "_report.md")).write_text("\n".join(lines))
    print(json.dumps({k: facts[k] for k in ("n", "answered_n", "supported_n", "metrics", "case_calls", "case_input_tokens")}, indent=2))


if __name__ == "__main__":
    main()
