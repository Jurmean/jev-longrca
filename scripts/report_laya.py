"""Independent artifact/coverage audit, then score Laya with the frozen JEV scorer."""
import collections
import csv
import gzip
import json
from pathlib import Path
import statistics
import time

import evaluate as ev
from evaluate_full import load_full
from evaluate_laya import CONFIG, MODEL_DIR, atomic_json
from make_report import LABELS, pct, stage_stats
from make_full_report import PAPER


def main():
    rows, manifest = load_full()
    run = ev.ROOT / "results/laya_full"
    config = json.loads((run / "config.json").read_text())
    assert config["runner_sha256"] == ev.sha((ev.ROOT / "scripts/evaluate_laya.py").read_bytes())
    jev_config = json.loads((ev.ROOT / "results/full/config.json").read_text())
    assert ev.sha((ev.ROOT / "scripts/evaluate.py").read_bytes()) == jev_config["runner_sha256"]
    assert ev.sha((ev.ROOT / "scripts/evaluate_full.py").read_bytes()) == jev_config["orchestrator_sha256"]
    dependency_path = run / "shared_dependencies.json"
    if dependency_path.exists():
        for name, digest in json.loads(dependency_path.read_text())["sha256"].items():
            assert ev.sha((ev.ROOT / name).read_bytes()) == digest
    for name, digest in config["upstream_source_sha256"].items():
        assert digest == ev.sha((MODEL_DIR / name).read_bytes())
    downloads = json.loads((ev.ROOT / "reports/laya_download_manifest.json").read_text())
    for name, entry in downloads["files"].items():
        assert ev.sha((MODEL_DIR / name).read_bytes()) == entry["sha256"]
    predictions = {p.stem: json.loads(p.read_text()) for p in (run / "predictions").glob("*.json")}
    assert set(predictions) == {r["question_ID"] for r in rows}, "All 1140 successful cases required"
    call_seconds, input_tokens, forwards, forced, max_tokens = [], 0, 0, 0, 0
    history_records, text_characters, shrink_calls = 0, 0, 0
    forwarded_recall_chars, forced_recall_chars = 0, 0
    for i, row in enumerate(rows):
        qid = row["question_ID"]
        p = predictions[qid]
        assert p["status"] == "ok" and p["question_ID"] == qid
        calls = []
        for path in sorted((run / "calls" / qid).glob("*.json.gz")):
            with gzip.open(path, "rt") as f:
                c = json.load(f)
            assert ev.sha(ev.dumps(c["request"]).encode()) == c["request_sha256"]
            assert c["request"]["model"] == c["response"]["model"] == CONFIG["model"]
            assert set(c["request"]) == {"model", "state", "questions"}
            expected_usage = 0
            for name, a in c["token_audit"].items():
                assert not a["silent_truncation"] and a["sequence_tokens"] <= CONFIG["max_len"]
                assert a["state_tokens"] + a["header_tokens"] + 4 == a["sequence_tokens"]
                q = c["request"]["questions"][name]
                answer = c["response"]["answers"][name]
                assert answer["choice"] in q["criteria"]
                assert set(answer["probabilities"]) == set(q["criteria"])
                if a["options"] == 1:
                    assert answer["deterministic_singleton"] and a["processed_tokens"] == 0
                    forced += 1
                else:
                    assert a["processed_tokens"] == a["sequence_tokens"]
                    forwards += 1
                expected_usage += a["processed_tokens"]
                max_tokens = max(max_tokens, a["sequence_tokens"])
            assert expected_usage == c["response"]["usage"]["input_tokens"]
            calls.append(c)
        assert len(calls) == p["local_calls"]
        assert sum(c["response"]["usage"]["input_tokens"] for c in calls) == p["input_tokens"]
        by_tag = {c["tag"]: c for c in calls}
        if p["method"] == "full_context":
            assert by_tag["direct"]["request"]["state"]["history"] == [ev.record(h) for h in row["history"]]
            final = by_tag["direct"]
            chars = sum(len(h["content"]) for h in row["history"])
            if any(a["processed_tokens"] > 0 for a in final["token_audit"].values()):
                forwarded_recall_chars += chars
            else:
                forced_recall_chars += chars
            candidates = [h["step"] for h in row["history"]]
            assert p["recall_candidates"] == p["expanded_candidates"] == candidates
            expected_tags = {"direct"}
        else:
            recall = [c for c in calls if c["tag"].startswith("recall_")]
            assert len(recall) == p["segments"]
            parts = collections.defaultdict(list)
            recalled = set()
            for c in recall:
                records = c["request"]["state"]["segment"]
                chars = sum(len(r["content"]) for r in records)
                if c["token_audit"]["root_step"]["processed_tokens"] > 0:
                    forwarded_recall_chars += chars
                else:
                    forced_recall_chars += chars
                ids = sorted({r["step"] for r in records})
                assert len(ids) <= CONFIG["max_recall_options"]
                assert set(c["request"]["questions"]["root_step"]["criteria"]) == {str(s) for s in ids}
                recalled.update(ev.top_steps(c["response"]["answers"]["root_step"], ids, CONFIG["recall_k"]))
                for r in records:
                    assert set(r) == {"step", "name", "role", "content", "content_start", "content_end"}
                    parts[r["step"]].append(r)
            assert sorted(recalled) == p["recall_candidates"]
            expanded = set(recalled)
            for step in recalled:
                handoff = ev.nearest_handoff(row["history"], step)
                if handoff is not None:
                    expanded.add(handoff)
            candidates = sorted(expanded)
            assert p["expanded_candidates"] == candidates
            expected_tags = {c["tag"] for c in recall} | {"final"}
            level = 0
            while len(candidates) > CONFIG["final_group_size"]:
                reduced = set()
                for offset in range(0, len(candidates), CONFIG["final_group_size"]):
                    group = candidates[offset:offset + CONFIG["final_group_size"]]
                    if len(group) <= CONFIG["recall_k"]:
                        reduced.update(group)
                        continue
                    tag = "reduce_%02d_%03d" % (level, offset)
                    expected_tags.add(tag)
                    c = by_tag[tag]
                    assert c["request"]["questions"] == ev.questions(group)
                    assert [x["candidate"]["step"] for x in c["request"]["state"]["candidate_evidence"]] == group
                    reduced.update(ev.top_steps(c["response"]["answers"]["root_step"], group, CONFIG["recall_k"]))
                candidates = sorted(reduced)
                level += 1
            assert set(parts) == set(range(len(row["history"])))
            for h in row["history"]:
                pieces = parts[h["step"]]
                offset = 0
                for r in pieces:
                    assert r["name"] == h.get("name", "") and r["role"] == h.get("role", "")
                    assert r["content_start"] == offset
                    offset += len(r["content"])
                    assert r["content_end"] == offset
                assert "".join(r["content"] for r in pieces) == h["content"]
            final = by_tag["final"]
        assert p["final_candidates"] == candidates
        assert set(by_tag) == expected_tags
        assert final["request"]["questions"] == ev.questions(candidates, ev.roles(row["history"]))
        a = final["response"]["answers"]
        assert p["predicted_step"] == int(a["root_step"]["choice"])
        assert p["predicted_role"] == a["responsible_role"]["choice"]
        history_records += len(row["history"])
        text_characters += sum(len(h["content"]) for h in row["history"])
        call_seconds.extend(c["elapsed_seconds"] for c in calls)
        input_tokens += p["input_tokens"]
        shrink_calls += sum(a["shrink_rounds"] > 0 for a in p["evidence_adaptations"])
        if (i + 1) % 100 == 0:
            print("Audited %d/1140" % (i + 1), flush=True)
    runtime = [json.loads(p.read_text()) for p in run.glob("runtime_*.json")]
    assert all("finished_unix" in r for r in runtime)
    launcher_path = run / "launcher.json"
    launcher = json.loads(launcher_path.read_text()) if launcher_path.exists() else None
    if launcher:
        assert all(code == 0 for code in launcher["exit_codes"])
    audit = {"trajectories": len(rows), "history_records_covered": history_records,
             "original_content_characters_covered": text_characters,
             "recall_original_characters_in_forwarded_requests": forwarded_recall_chars,
             "recall_original_characters_in_singleton_only_requests": forced_recall_chars,
             "coverage_note": "Coverage means lossless recall request construction. Singleton-only requests retain the sole candidate without a model forward.",
             "local_calls": len(call_seconds), "model_forwards": forwards,
             "deterministic_singleton_questions": forced, "input_tokens": input_tokens,
             "max_sequence_tokens": max_tokens, "silent_truncations": 0,
             "explicit_evidence_shrink_calls": shrink_calls,
             "median_local_call_seconds": statistics.median(call_seconds),
             "p95_local_call_seconds": sorted(call_seconds)[int(.95*(len(call_seconds)-1))],
             "sum_local_call_seconds": sum(call_seconds),
             "inference_wall_seconds": max(r["finished_unix"] for r in runtime)-min(r["started_unix"] for r in runtime),
             "full_run_wall_seconds": launcher["finished_unix"]-launcher["started_unix"] if launcher else None,
             "gpu_workers": len(runtime), "gpu_model": runtime[0]["gpu"],
             "peak_memory_bytes_per_worker": max(r["peak_cuda_memory_bytes"] for r in runtime),
             "upstream_weights_and_source_hashes_verified": True}
    assert forwarded_recall_chars + forced_recall_chars == text_characters
    atomic_json(ev.ROOT / "reports/laya_result_audit.json", audit)
    ev.CONFIG.clear()
    ev.CONFIG.update(CONFIG)
    summary = ev.summarize(rows, predictions, manifest)
    for key in ("successful_api_calls", "estimated_input_cost_usd", "cost_note"):
        summary.pop(key, None)
    summary.update(local_inference=audit, inference_api_cost_usd=0,
                   cost_note="Local GPU inference; electricity/hardware costs not estimated.")
    summary["stage_recall"] = stage_stats(summary["details"])
    mini_ids = {f["question_ID"] for f in json.loads((ev.ROOT / "data/manifest.json").read_text())["files"]}
    summary["mini_subset"] = ev.metrics([d for d in summary["details"] if d["question_ID"] in mini_ids])
    summary["uniform_random_expected"] = {
        "role_correct": statistics.mean(1 / len(ev.roles(r["history"])) for r in rows),
        "step_exact": statistics.mean(1 / len(r["history"]) for r in rows)}
    atomic_json(run / "summary.json", summary)
    atomic_json(ev.ROOT / "reports/laya_full_metrics.json", {k: v for k, v in summary.items() if k != "details"})
    fields = ["question_ID", "source", "steps", "reference_role", "predicted_role", "reference_step",
              "predicted_step", "role_correct", "step_exact", "step_within_5", "absolute_step_error"]
    with (ev.ROOT / "reports/laya_full_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for d in summary["details"]:
            writer.writerow({k: d["prediction"].get(k, d.get(k)) for k in fields})
    jev = json.loads((ev.ROOT / "reports/full_metrics.json").read_text())
    m, j = summary["overall"], jev["overall"]
    lines = ["# Laya 与 JEV：LongRCA Full 本地评测", "",
             "已完成官方 Full **1,140/1,140 条**；Laya 的全部预测均为本次本地计算。"
             "与 JEV 使用相同数据文件、原始步骤编号、角色标准化和评分函数。", "",
             "## Full 结果", "",
             "| 方法 | 角色准确率 ↑ | 根因精确命中 ↑ | 根因 ±5 命中 ↑ | MAE ↓ |",
             "|---|---:|---:|---:|---:|"]
    def metric_line(label, x):
        return "| %s | %s | %s | %s | %.2f |" % (label, pct(x["role_correct"]), pct(x["step_exact"]),
                                                  pct(x["step_within_5"]), x["valid_output_root_mae"])
    lines.extend([metric_line("JEV + 原有候选筛选", j), metric_line("Laya 英文权重 + 上下文适配", m), "",
                  "Laya − JEV：角色 **%+.2f**、根因精确 **%+.2f**、±5 **%+.2f** 个百分点；MAE **%+.2f 步**。"
                  % ((m["role_correct"]-j["role_correct"])*100, (m["step_exact"]-j["step_exact"])*100,
                     (m["step_within_5"]-j["step_within_5"])*100, m["valid_output_root_mae"]-j["valid_output_root_mae"]),
                  "有效角色 %s，有效步骤 %s。均匀随机选择合法角色/全部原始步骤的解析期望分别为 %s / %s。"
                  % (pct(m["valid_role"]), pct(m["valid_step"]), pct(summary["uniform_random_expected"]["role_correct"]),
                  pct(summary["uniform_random_expected"]["step_exact"])), "",
                  "## 论文方法参考", "",
                  "以下为 [LongRCA 论文表 3](https://arxiv.org/html/2608.15242v1#S6.T3) 的发表值，"
                  "未在本项目重跑，统一使用 DeepSeek-V4-Flash。它们与本次 Laya/JEV 的模型及输入处理不同。"
                  "论文报告 178,137 个历史步骤，本次固定 JSON 发布版为 177,884 个；"
                  "虽然同为 Full 1,140 条，不能假定原始记录逐字节相同。", "",
                  "| 论文方法 | 角色准确率 | 根因精确 | 根因 ±5 | MAE |", "|---|---:|---:|---:|---:|"])
    for name, role, exact, pm5, mae in PAPER:
        lines.append("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f |" % (name, role, exact, pm5, mae))
    lines += ["", "## 各来源", "", "| 来源 / 方法 | 角色准确率 | 根因精确 | 根因 ±5 | MAE |", "|---|---:|---:|---:|---:|"]
    for source, value in summary["by_source"].items():
        lines.extend([metric_line(LABELS[source] + " / Laya (n=%d)" % value["n"], value),
                      metric_line(LABELS[source] + " / JEV", jev["by_source"][source])])
    stage = summary["stage_recall"]
    lines += ["", "## 分段路径的候选保留", "",
              "| 阶段 | 含参考根因的轨迹数 | 比例 |", "|---|---:|---:|"]
    for key, label in [("recall_candidates", "完整分段初筛后"), ("expanded_candidates", "交接扩展后"),
                       ("final_candidates", "最终候选集合"), ("exact", "最终选择正确")]:
        lines.append("| %s | %d/%d | %s |" % (label, stage[key], stage["n"], pct(stage[key]/stage["n"])))
    lines += ["", "此表只含分段路径。两个模型的分段大小不同，不把候选保留率差异解释成单独的模型效应。"]
    lines += ["", "## Mini 子集", "", "由 Full 预测中提取原有 200 条 Mini，不额外推理。",
              "", "| 方法 | 角色准确率 | 根因精确 | 根因 ±5 | MAE |", "|---|---:|---:|---:|---:|",
              metric_line("Laya", summary["mini_subset"]),
              metric_line("JEV", json.loads((ev.ROOT / "reports/phase1_metrics.json").read_text())["overall"]), "",
              "## 协议与可比性", "",
              "- 官方模型 `convaiinnovations/laya` 根目录英文权重，固定版本 `%s`。未微调，未根据测试答案调参。" % CONFIG["model_revision"],
              "- Laya 默认 max_len=512、head_max_len=192。本次将 max_len 扩展为 8192（编码器配置上限）；"
              "每题动态给足提示词和选项预算。属于超出默认长度的测试，不代表默认 512-token 配置性能。",
              "- 保留 JEV 的根因定义提示词、每段取前 3、交接扩展、8 候选复核和独立责任角色选择。"
              "改用 tokenizer 精确分段，每段至多 16 个不同步骤；共享首尾节选上限 2048 tokens。",
              "- 构造的初筛请求无损覆盖每条原始记录的全部内容。最终复核与 JEV 一样使用有标记节选；"
              "若超预算，显式缩短候选及相邻/交接记录的节选，并逐次记录。没有静默截断。",
              "- 官方推理会把文本里的字面 [MASK] 替换为空格；本次沿用该行为。原始文本保存在请求中。",
              "- 仅一个合法选项时直接返回唯一选项，避免官方 topk(2) 对单选项报错，单独统计、不冒充模型推理。"
              "这类片段直接保留唯一候选；完整文本覆盖指请求构造，不能声称模型实际阅读了每个单选片段。",
              "- 本次是两套可运行流程对比，输入预算不同，不能单独归因于模型。Laya 是英文权重；"
              "VitaBench 等中文记录存在语言不匹配，尚未测试 multilingual 或 Router。",
              "- 两种流程只接收 history，不传入 mistake_agent、mistake_step、mistake_reason；"
              "只有全部推理结束后才计算分数。此项目未复现论文 RCTA，也未对 Laya 训练或校准。", "",
              "## 本地用量与审计", "",
              "- %s 个 %s 工作进程；完整推理墙钟 **%.1f 分钟**（最早模型加载完成至最晚完成，不含下载、环境安装、冒烟和审计）。"
              % (audit["gpu_workers"], audit["gpu_model"], audit["inference_wall_seconds"]/60),
              ("- 从全量进程启动到所有分片结束 **%.1f 分钟**，含模型加载和调度等待。"
               % (audit["full_run_wall_seconds"]/60)) if launcher else "- 手动启动分片，未记录统一进程启动时间。",
              "- 本地逻辑调用 %s 次，实际模型 forward %s 次，单选项直接返回 %s 题。"
              % (format(audit["local_calls"], ","), format(forwards, ","), format(forced, ",")),
              "- 实际输入 %s tokens，最长序列 %s；推理 API 费用 $0，未估算显卡/电费。"
              % (format(input_tokens, ","), max_tokens),
              "- 调用耗时中位数 %.3f 秒，P95 %.3f 秒；包含输入校验与题目处理，不是纯 GPU forward 延迟。"
              % (audit["median_local_call_seconds"], audit["p95_local_call_seconds"]),
              "- 核验 %s 条历史记录和 %s 个原始内容字符，所有请求摘要、模型/代码摘要、预测来源和输入 token 计数一致。"
              % (format(history_records, ","), format(text_characters, ",")),
              "- 初筛原始片段中，%s 个字符位于实际 forward 的请求，%s 个字符位于只含唯一选项的直接保留请求；"
              "不含重复提供的共享上下文，也不计后续节选复核中的重复内容。"
              % (format(forwarded_recall_chars, ","), format(forced_recall_chars, ",")),
              "- 与 JEV API 的云端执行、并发数及上下文不同，不据此宣称端到端速度优势。", "",
              "数据版本：`%s`。" % manifest["revision"], "",
              "文件：`laya_full_metrics.json`、`laya_full_predictions.csv`、`laya_result_audit.json`；"
              "原始请求和响应位于 `results/laya_full/calls/`。", "",
              "参考：[Laya 官方模型](https://huggingface.co/convaiinnovations/laya)；"
              "[官方代码](https://github.com/NandhaKishorM/laya)；"
              "[LongRCA 数据](https://huggingface.co/datasets/CLoud5-real/longrca-bench)。", ""]
    (ev.ROOT / "reports/laya_full_report.md").write_text("\n".join(lines))
    if (ev.ROOT / "reports/laya_sanity_checks.json").exists():
        from diagnose_laya_results import write_diagnostics
        write_diagnostics()
    print(json.dumps({"overall": m, "mini": summary["mini_subset"], "audit": audit}, indent=2), flush=True)


if __name__ == "__main__":
    main()
