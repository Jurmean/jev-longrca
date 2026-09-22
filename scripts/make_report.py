"""Build an auditable Chinese report from completed predictions; no inference."""
import argparse
import collections
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from evaluate import ROOT, load_data, roles, summarize, metrics

LABELS = {"swe_bench_pro": "SWE-bench Pro", "terminal_bench_2": "Terminal-Bench 2",
          "travelplanner": "TravelPlanner", "vitabench": "VitaBench", "webarena_verified": "WebArena Verified"}


def pct(x):
    return "—" if x is None else "%.1f%%" % (100 * x)


def wilson(p, n):
    z = 1.959963984540054
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return "%s–%s" % (pct(center - radius), pct(center + radius))


def stage_stats(details):
    subset = [d for d in details if d["method"] == "segmented_choice"]
    return {"n": len(subset), **{key: sum(d["reference_step"] in d["prediction"].get(key, []) for d in subset)
                                for key in ("recall_candidates", "expanded_candidates", "final_candidates")},
            "exact": sum(d["step_exact"] for d in subset)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/phase1")
    args = parser.parse_args()
    run = ROOT / args.input
    rows, manifest = load_data()
    predictions = {p.stem: json.loads(p.read_text()) for p in (run / "predictions").glob("*.json")}
    if set(predictions) != {r["question_ID"] for r in rows}:
        raise SystemExit("A final report requires all 200 unique predictions")
    summary = summarize(rows, predictions, manifest)
    saved = json.loads((run / "summary.json").read_text())
    if summary["overall"] != saved["overall"]:
        raise SystemExit("Recomputed metrics do not match saved summary")
    details, overall = summary["details"], summary["overall"]
    row_by_id = {r["question_ID"]: r for r in rows}
    single_role = [d for d in details if len(roles(row_by_id[d["question_ID"]]["history"])) == 1]
    multi_role = [d for d in details if len(roles(row_by_id[d["question_ID"]]["history"])) > 1]
    stage = stage_stats(details)
    n = len(rows)
    random_role = statistics.mean(1 / len(roles(r["history"])) for r in rows)
    random_step = statistics.mean(1 / len(r["history"]) for r in rows)
    random_pm5 = statistics.mean((min(len(r["history"]) - 1, r["mistake_step"] + 5)
                                 - max(0, r["mistake_step"] - 5) + 1) / len(r["history"]) for r in rows)
    latency = sorted(p["elapsed_seconds"] for p in predictions.values())
    api_latencies, retries, models, input_max, total_request_bytes = [], 0, set(), 0, 0
    call_count = 0
    for path in (run / "calls").glob("*/*.json"):
        call = json.loads(path.read_text())
        call_count += 1
        api_latencies.append(call["elapsed_seconds"])
        retries += call["attempts"] - 1
        models.add(call["response"]["model"])
        input_max = max(input_max, call["response"]["usage"]["input_tokens"])
        total_request_bytes += len(json.dumps(call["request"], ensure_ascii=False).encode())
    failures = sorted((d for d in details if not d["step_exact"]),
                      key=lambda d: -(d["absolute_step_error"] or 0))
    # Illustrative examples chosen deterministically after inference, without prompt tuning.
    examples = []
    for source in sorted(LABELS):
        candidates = [d for d in failures if d["source"] == source]
        if candidates:
            examples.append(candidates[0])
    lines = ["# 第一阶段：JEV 在 LongRCA-Mini 上的效果", "",
             "生成时间：%s。" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "",
             "已完成官方固定 Mini 的 **200/200 条真实 API 评测**，5 个来源各 40 条。模型为 `%s`，"
             "使用固定的 JEV Choice 分段候选筛选流程；无其他模型参与推理。" % ", ".join(sorted(models)), "",
             "## 主要结果", "", "| 指标 | 结果 | 命中数 | 95% Wilson 区间 |",
             "|---|---:|---:|---:|"]
    for key, label in [("role_correct", "责任角色准确率"), ("step_exact", "根因步骤精确命中"),
                       ("step_within_5", "根因步骤 ±5 命中")]:
        lines.append("| %s | **%s** | %d/200 | %s |" % (label, pct(overall[key]), round(overall[key] * n), wilson(overall[key], n)))
    lines += ["", "来源加权有效预测 Root MAE：**%.2f 步**。有效角色输出 %s，有效步骤输出 %s。"
              % (overall["source_weighted_valid_output_root_mae"], pct(overall["valid_role"]), pct(overall["valid_step"])), "",
              "区间是对样本命中率的描述性二项区间，未建模来源/生成模型相关性，也不表示重复运行的波动范围。", "",
              "## 各来源", "", "| 来源 | n | 角色准确率 | Root Exact | Root ±5 | Root MAE |",
              "|---|---:|---:|---:|---:|---:|"]
    for source, m in summary["by_source"].items():
        lines.append("| %s | %d | %s | %s | %s | %.2f |" % (LABELS[source], m["n"], pct(m["role_correct"]),
                       pct(m["step_exact"]), pct(m["step_within_5"]), m["valid_output_root_mae"]))
    role_note = ("其中 %d 条轨迹只有一个合法角色。其余 %d 条多角色轨迹上的角色准确率为 **%s**。"
                 % (len(single_role), len(multi_role), pct(metrics(multi_role)["role_correct"]))) if single_role else (
                 "核对完整角色集合后，200 条轨迹均至少有两个合法角色选项（包括日志中记录的工具角色），没有单选项角色题。")
    lines += ["", role_note, "",
              "## 按轨迹步骤数分组", "", "| 步骤数 | n | 角色准确率 | Root Exact | Root ±5 |",
              "|---|---:|---:|---:|---:|"]
    for label, low, high in [("≤100", 0, 100), ("101–200", 101, 200), ("201–400", 201, 400), (">400", 401, 100000)]:
        m = metrics([d for d in details if low <= d["steps"] <= high])
        lines.append("| %s | %d | %s | %s | %s |" % (label, m["n"], pct(m["role_correct"]), pct(m["step_exact"]), pct(m["step_within_5"])))
    lines += ["", "长度与任务来源、角色组织相关，此处为描述性分组，不能据此归因于长度本身。"]
    lines += ["", "## 上下文处理与候选保留", "",
              "| 路径 | n | 角色准确率 | Root Exact | Root ±5 |", "|---|---:|---:|---:|---:|"]
    for method, m in summary["by_method"].items():
        lines.append("| %s | %d | %s | %s | %s |" % (method, m["n"], pct(m["role_correct"]), pct(m["step_exact"]), pct(m["step_within_5"])))
    lines += ["", "完整上下文与分段路径处理的样本长度、来源不同，表格不能作为方法优劣的受控对比。", "",
              "分段路径 %d 条的真实根因候选保留情况（事后用标签统计，不参与预测）：" % stage["n"], "",
              "| 阶段 | 含真实根因的轨迹数 | 占分段样本比例 |", "|---|---:|---:|"]
    for key, label in [("recall_candidates", "各段 Top-3 合并后"), ("expanded_candidates", "加入前置交接步骤后"),
                       ("final_candidates", "递归筛选到最终候选后"), ("exact", "最终选中真实根因")]:
        lines.append("| %s | %d/%d | %s |" % (label, stage[key], stage["n"], pct(stage[key] / stage["n"])))
    lines += ["", "在分段样本中，交接扩展后含真根因的 %d 条中有 %d 条在递归筛选时丢失；最终仍含真根因的 %d 条中仅 %d 条选择正确。"
              "这说明后续改进需要同时检查候选保留和最终判定，不能只提高初筛召回率。"
              % (stage["expanded_candidates"], stage["expanded_candidates"] - stage["final_candidates"],
                 stage["final_candidates"], stage["exact"])]
    lines += ["", "初筛读取所有原始记录文本，超长记录按 UTF-8 无损拆片且保留 step ID。任务开头、末尾背景及候选复核阶段使用明确标记的首尾节选；"
              "因此所有文本被初筛读取，不代表最终判定同时拥有完整轨迹。详细阈值与提示词见 README 和运行配置。", "",
              "## 随机选择参照", "",
              "按每条轨迹的合法角色集合与全部步骤均匀随机选择，解析计算的期望为：角色 %s、Root Exact %s、Root ±5 %s。"
              "这是无需模型的机会水平参照，不是实测模型基线；未按标签设计固定角色预测。" % (pct(random_role), pct(random_step), pct(random_pm5)), "",
              "## 代表性定位失败", "",
              "每个来源选择绝对步骤误差最大的一例；仅用于展示本次结果的失败位置。", "",
              "| 实例 | 标注步骤 | 预测步骤 | 误差 | 标注角色 | 预测角色 | 真根因在哪个阶段丢失 |", "|---|---:|---:|---:|---|---|---|"]
    for d in examples:
        p, ref = d["prediction"], d["reference_step"]
        if ref not in p["expanded_candidates"]:
            lost = "初筛/交接扩展未召回"
        elif ref not in p["final_candidates"]:
            lost = "候选递归筛选"
        else:
            lost = "最终选择"
        lines.append("| %s | %d | %d | %d | %s | %s | %s |" %
                     (d["question_ID"], ref, p["predicted_step"], d["absolute_step_error"],
                      d["reference_role"], p["predicted_role"], lost))
    lines += ["", "此表只能定位流程中的丢失阶段，不能证明模型内部的错误原因；JEV 不生成文本推理。逐条预测及原始调用均已保存。", "",
              "## 用量与复现", "",
              "- 成功 API 请求：%d 次；重试：%d 次；输入 tokens：%s。" % (call_count, retries, format(summary["input_tokens"], ",")),
              "- 公开单价估算：**$%.4f**，按 $0.042 / 百万输入 tokens 计算，输出免费；不是实际账单。" % summary["estimated_input_cost_usd"],
              "- 单条轨迹耗时中位数 %.2f 秒、P95 %.2f 秒；单请求耗时中位数 %.2f 秒。" %
              (statistics.median(latency), latency[math.ceil(len(latency) * .95) - 1], statistics.median(api_latencies)),
              "- 本次全量命令墙钟耗时 %.2f 秒（含缓存复用；各来源最短的 5 条在 smoke 阶段已完成）。" % saved.get("run_wall_seconds", 0),
              "- 最大单请求计费输入 tokens：%s。初始接口连通性测试 317 输入 tokens 另计，不在上述评测用量内。" % format(input_max, ","),
              "- 数据固定版本：`%s`，200 个原始文件的 SHA-256 均已校验。" % manifest["revision"],
              "- 原始请求/响应、模型返回版本、用量、耗时与重试信息均在 `results/phase1/calls/`。认证头和密钥不写入结果。", "",
              "## 结论边界", "",
              "本次测得的是 **JEV + 已记录的上下文处理/候选筛选流程** 在此固定 Mini 上的表现。仅运行一次，未测种子/顺序稳定性，也未通过测试标签调优。",
              "责任角色与根因步骤独立预测、独立评分；±5 是宽松辅助指标，Exact 是主要定位指标。",
              "公开数据只提供轨迹和人工标注，没有独立 evaluator outcome 字段；只依据日志中已有的信息并告知模型运行已失败，未读取人工理由作为提示。",
              "Mini 的来源比例、模型和方法都与论文全量实验不同，因此不能直接宣称超过/低于论文方法。下一阶段如需评估改进，需固定独立开发/验证划分或新增未用于调优的测试集，再作受控比较。", "",
              "## 来源", "",
              "- [官方数据集](https://huggingface.co/datasets/CLoud5-real/longrca-bench)",
              "- [LongRCA 论文：评分定义 §6.1](https://arxiv.org/html/2608.15242v1)",
              "- [TypeSafe 模型、上下文与价格](https://docs.typesafe.ai/models)",
              "- [TypeSafe Choice 接口](https://docs.typesafe.ai/primitives/choice)", ""]
    (ROOT / "reports/phase1_report.md").write_text("\n".join(lines))
    compact = {key: value for key, value in summary.items() if key != "details"}
    compact["cost_note"] = "Includes the 5 reused benchmark smoke cases; excludes only the separate 317-token API connectivity check. Price estimate, not a bill."
    compact["stage_recall"] = stage
    compact["random_expected"] = {"role": random_role, "step_exact": random_step, "step_pm5": random_pm5}
    compact["retry_count"] = retries
    compact["run_wall_seconds"] = saved.get("run_wall_seconds")
    (ROOT / "reports/phase1_metrics.json").write_text(json.dumps(compact, indent=2) + "\n")
    with (ROOT / "reports/phase1_predictions.csv").open("w", newline="") as handle:
        fields = ["question_ID", "source", "steps", "method", "reference_role", "predicted_role", "reference_step",
                  "predicted_step", "role_correct", "step_exact", "step_within_5", "absolute_step_error", "role_confidence", "step_confidence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for d in details:
            output = {key: d.get(key, d["prediction"].get(key)) for key in fields}
            writer.writerow(output)
    print(json.dumps({"overall": overall, "stage_recall": stage, "random_expected": {
        "role": random_role, "step_exact": random_step, "step_pm5": random_pm5}, "calls": call_count,
        "retries": retries, "report": str(ROOT / "reports/phase1_report.md")}, indent=2))


if __name__ == "__main__":
    main()
