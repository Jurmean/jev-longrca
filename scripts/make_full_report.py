"""Full-release report and published-method comparison; offline only."""
import csv
from datetime import datetime, timezone
import json
import statistics

from evaluate import ROOT, summarize, metrics, roles
from evaluate_full import load_full
from make_report import LABELS, pct, wilson, stage_stats

PAPER = [
    ("All-at-once", 26.2, 7.6, 19.9, 55.9),
    ("Step-by-step", 22.2, 5.3, 16.9, 52.3),
    ("Binary search", 23.0, 3.4, 13.3, 61.7),
    ("ECHO", 27.5, 13.2, 24.7, 50.4),
    ("FALAT", 19.0, 2.8, 12.5, 66.6),
    ("RCTA", 51.1, 24.1, 37.4, 38.6),
]


def main():
    rows, manifest = load_full()
    run = ROOT / "results/full"
    saved = json.loads((run / "summary.json").read_text())
    predictions = {p.stem: json.loads(p.read_text()) for p in (run / "predictions").glob("*.json")}
    if set(predictions) != {r["question_ID"] for r in rows}:
        raise SystemExit("Full report requires all 1140 predictions")
    summary = summarize(rows, predictions, manifest)
    if summary["overall"] != saved["overall"]:
        raise SystemExit("Recomputed and saved metrics differ")
    details, m = summary["details"], summary["overall"]
    stage = stage_stats(details)
    audit = json.loads((ROOT / "reports/full_result_audit.json").read_text())
    assert audit["trajectories"] == 1140
    mini = json.loads((ROOT / "reports/phase1_metrics.json").read_text())
    mini_ids = set(json.loads((run / "config.json").read_text())["reused_mini_ids"])
    reused = metrics([d for d in details if d["question_ID"] in mini_ids])
    assert all(reused[k] == mini["overall"][k] for k in reused)
    new_only = metrics([d for d in details if d["question_ID"] not in mini_ids])
    chance = {"role": statistics.mean(1 / len(roles(r["history"])) for r in rows),
              "exact": statistics.mean(1 / len(r["history"]) for r in rows)}
    lines = ["# JEV 在 LongRCA Full 上的评测", "",
             "生成时间：%s。" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "",
             "已完成官方 Full **1,140/1,140 条**。使用与 Mini 完全相同的 `jev-1.13.0`、提示词、上下文处理和候选筛选协议。"
             "200 条 Mini 的原始数据逐字节一致，复用其真实预测；其余 940 条新增运行。未根据 Mini 结果调参。", "",
             "## 总体结果", "", "| 指标 | 结果 | 命中数 | 95% Wilson 区间 |", "|---|---:|---:|---:|"]
    for key, label in [("role_correct", "责任角色准确率"), ("step_exact", "根因步骤精确命中"), ("step_within_5", "根因步骤 ±5 命中")]:
        lines.append("| %s | **%s** | %d/1140 | %s |" % (label, pct(m[key]), round(m[key] * 1140), wilson(m[key], 1140)))
    lines += ["", "来源加权有效预测 Root MAE：**%.2f 步**；有效角色 %s，有效步骤 %s。"
              % (m["source_weighted_valid_output_root_mae"], pct(m["valid_role"]), pct(m["valid_step"])),
              "区间只描述样本命中率，不建模来源相关性或重复运行波动。", "",
              "## 与论文全量结果对照", "",
              "下表论文数值来自 [LongRCA 论文表 3](https://arxiv.org/html/2608.15242v1#S6.T3)。"
              "所有行均为 Full 1,140 条和相同评分定义；论文六种方法使用 DeepSeek-V4-Flash，本次使用 JEV。"
              "论文行是发表值，未在本项目重跑，不能隔离模型和方法各自的贡献，也没有逐例配对显著性检验。", "",
              "数据版本核验发现：本次固定的官方 JSON 发布版本合计 **%s 个历史步骤**，论文报告 178,137 个；"
              "轨迹数量和各来源数量一致，但总步骤统计略有差异。因此也不能假定本次原始记录与论文实验版本逐字节一致。"
              % format(audit["history_records_covered"], ","), "",
              "| 方法 | 角色准确率 ↑ | Root Exact ↑ | Root ±5 ↑ | Root MAE ↓ |", "|---|---:|---:|---:|---:|"]
    for name, role, exact, pm5, mae in PAPER:
        lines.append("| %s（论文） | %.1f%% | %.1f%% | %.1f%% | %.1f |" % (name, role, exact, pm5, mae))
    lines.append("| **JEV + 固定候选筛选（本次）** | **%s** | **%s** | **%s** | **%.2f** |" %
                 (pct(m["role_correct"]), pct(m["step_exact"]), pct(m["step_within_5"]), m["source_weighted_valid_output_root_mae"]))
    lines += ["", "按 Root Exact 的发表值差异：本次相对 ECHO 为 **%+.2f 个百分点**，相对 RCTA 为 **%+.2f 个百分点**。"
              % (m["step_exact"] * 100 - 13.2, m["step_exact"] * 100 - 24.1), "",
              "## 各来源", "", "| 来源 | n | 角色准确率 | Root Exact | Root ±5 | Root MAE |", "|---|---:|---:|---:|---:|---:|"]
    for source, v in summary["by_source"].items():
        lines.append("| %s | %d | %s | %s | %s | %.2f |" % (LABELS[source], v["n"], pct(v["role_correct"]),
                     pct(v["step_exact"]), pct(v["step_within_5"]), v["valid_output_root_mae"]))
    lines += ["", "## Mini 与 Full", "", "| 范围 | n | 角色准确率 | Root Exact | Root ±5 |", "|---|---:|---:|---:|---:|"]
    for label, v in [("Mini（复用）", reused), ("新增部分", new_only), ("Full", m)]:
        lines.append("| %s | %d | %s | %s | %s |" % (label, v["n"], pct(v["role_correct"]), pct(v["step_exact"]), pct(v["step_within_5"])))
    lines += ["", "Full 包含 Mini，这不是两个独立测试集。Mini 各来源均占 20%；Full 中 TravelPlanner 占约 60.1%，总体分数变化受来源分布影响。", "",
              "## 上下文处理与候选保留", "", "| 路径 | n | 角色准确率 | Root Exact | Root ±5 |", "|---|---:|---:|---:|---:|"]
    for name, v in summary["by_method"].items():
        lines.append("| %s | %d | %s | %s | %s |" % (name, v["n"], pct(v["role_correct"]), pct(v["step_exact"]), pct(v["step_within_5"])))
    lines += ["", "这两条路径处理的样本不同，不能用作受控消融。初筛读取所有原始文本；最终复核仍使用有标记的节选。", "",
              "| 分段路径阶段 | 含真根因的轨迹数 | 比例 |", "|---|---:|---:|"]
    for key, label in [("recall_candidates", "初筛合并"), ("expanded_candidates", "交接扩展后"),
                       ("final_candidates", "最终候选集合"), ("exact", "最终选择正确")]:
        lines.append("| %s | %d/%d | %s |" % (label, stage[key], stage["n"], pct(stage[key] / stage["n"])))
    lines += ["", "全部合法角色/步骤均匀随机选择的解析期望：角色 %s，根因精确命中 %s，仅作机会水平参照。"
              % (pct(chance["role"]), pct(chance["exact"])), "",
              "## 用量与核验", "",
              "- Full 总计成功调用 **%s 次**，输入 **%s tokens**，按已核查公开单价估算累计 **$%.4f**。"
              % (format(saved["successful_api_calls"], ","), format(saved["input_tokens"], ","), saved["estimated_input_cost_usd"]),
              "- 其中本次新增 940 条：输入 **%s tokens**，估算新增费用 **$%.4f**；Mini 部分未重复付费。"
              % (format(saved["new_input_tokens"], ","), saved["new_estimated_input_cost_usd"]),
              "- 最后一轮全量命令墙钟耗时 **%.1f 分钟**（服务恢复稳定后使用 6 个样本并发，包含缓存复用，不含前期运行、数据下载和核验）。" % (saved["run_wall_seconds"] / 60),
              "- 初始 8 并发期间出现 HTTP 403，后续还出现上游 503 和间歇性 Unknown model 错误；同一请求复查仍返回 jev-1.13.0。暂停、降低并发并添加传输退避后恢复；成功中间调用均复用。历史错误记录不等于最终失败。",
              "- 费用使用 $0.042 / 百万输入 tokens；输出免费。仅统计保存成功响应的评测请求，不含少量诊断请求或无用量响应的失败调用；不是实际账单。",
              "- 审计覆盖 %s 条原始历史记录：完整初筛文本覆盖、预测与原始响应一致、模型版本、token 用量、数据摘要及代码摘要均已核验。"
              % format(audit["history_records_covered"], ","),
              "- 数据固定版本 `%s`；推理只读取 history，不读取人工答案及理由。" % manifest["revision"], "",
              "## 文件与限制", "",
              "- `reports/full_metrics.json`：指标、用量、候选保留统计。",
              "- `reports/full_predictions.csv`：1,140 条预测与标签对照。",
              "- `reports/full_result_audit.json`：独立核验结果。",
              "- `results/full/`：配置、逐条预测和原始请求/响应。复用样本的 calls 目录链接到 Mini 原始记录。",
              "- 仅单次固定流程评测，不是 JEV 原生无限上下文能力，也不是论文 RCTA 的复现。",
              "- 发布 JSON 无独立 evaluator outcome 字段；只使用日志已有信息并告知模型运行已失败，未从人工理由中提取失败提示。",
              "- 详细方法及重现命令见项目 README。", "",
              "参考：[官方数据集](https://huggingface.co/datasets/CLoud5-real/longrca-bench)；"
              "[论文评分定义与结果](https://arxiv.org/html/2608.15242v1#S6)；"
              "[JEV 模型、限制与价格](https://docs.typesafe.ai/models)。", ""]
    (ROOT / "reports/full_report.md").write_text("\n".join(lines))
    compact = {k: v for k, v in saved.items() if k != "details"}
    compact.update(stage_recall=stage, random_expected=chance)
    (ROOT / "reports/full_metrics.json").write_text(json.dumps(compact, indent=2) + "\n")
    fields = ["question_ID", "source", "steps", "method", "reference_role", "predicted_role", "reference_step",
              "predicted_step", "role_correct", "step_exact", "step_within_5", "absolute_step_error", "role_confidence", "step_confidence"]
    with (ROOT / "reports/full_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for d in details:
            writer.writerow({key: d.get(key, d["prediction"].get(key)) for key in fields})
    print(json.dumps({"overall": m, "stage_recall": stage, "new_cost_estimate_usd": saved["new_estimated_input_cost_usd"]}, indent=2))


if __name__ == "__main__":
    main()
