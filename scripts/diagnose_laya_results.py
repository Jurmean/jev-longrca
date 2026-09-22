"""Describe the observed failure mode without changing predictions or tuning."""
import collections
import json
from pathlib import Path

from evaluate_full import load_full
from evaluate import ROOT


def write_diagnostics():
    rows, _ = load_full()
    pred = [json.loads(p.read_text()) for p in (ROOT / "results/laya_full/predictions").glob("*.json")]
    assert len(pred) == len(rows) == 1140
    sanity_path = ROOT / "reports/laya_sanity_checks.json"
    if not sanity_path.exists():
        return
    sanity = json.loads(sanity_path.read_text())
    zero = sum(p["predicted_step"] == 0 for p in pred)
    first = sum(p["predicted_step"] == p["final_candidates"][0] for p in pred)
    role_counts = collections.Counter(p["predicted_role"] for p in pred)
    diagnosis = {"step_zero_predictions": zero, "first_final_candidate_predictions": first,
                 "reference_step_zero_count": sum(r["mistake_step"] == 0 for r in rows),
                 "predicted_role_counts": dict(role_counts),
                 "short_control_correct": sanity["correct"], "short_control_n": sanity["n"],
                 "unique_short_control_texts": len({r["state"] for r in sanity["tests"]}),
                 "notes": "Post-run diagnostics only. Three short English texts, three option orders and two max_len settings; not a test of long-context RCA. No benchmark predictions were changed."}
    (ROOT / "reports/laya_diagnostics.json").write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n")
    metrics_path = ROOT / "reports/laya_full_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["post_run_diagnostics"] = diagnosis
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    title = "## 异常结果复查"
    section = [title, "",
               "本次配置的根因精确命中为 0/1140，明显低于均匀随机选择原始步骤约 1.0% 的解析期望。"
               "该配置未能形成有效的根因判断，不应作为当前 JEV 流程的等效替代。", "",
               "- **%d/1140（%.2f%%）** 选择 step 0；参考答案中 step 0 为 %d 条。"
               % (zero, zero/1140*100, diagnosis["reference_step_zero_count"]),
               "- %d/1140 选择最终候选列表的第一项，%d/1140 把责任归给 Computer_terminal，输出高度集中。"
               % (first, role_counts.get("Computer_terminal", 0)),
               "- 追加了与 benchmark 标签无关的 3 个短英文分类控制样例（账单、登录、物流），"
               "各使用 3 种选项顺序和 max_len=512/8192 两种配置，共 **%d/%d** 次正确。"
               % (sanity["correct"], sanity["n"]),
               "- 控制样例说明基本本地推理可用，并非对任何输入都固定选第一项；"
               "但输入均很短，这不验证 8K 长日志归因能力，也不能分离上下文长度、提示词和任务适配的影响。",
               "- 诊断在全量预测完成之后进行，没有修改预测、重调参数或据此重新选择本次结果。"
               "结论限定于本次英文 checkpoint 与 8K 候选筛选配置；尚未评测 multilingual 或默认 512-token 的 Full 方案。",
               "- 记录见 `laya_diagnostics.json` 和 `laya_sanity_checks.json`。", ""]
    report = ROOT / "reports/laya_full_report.md"
    text = report.read_text()
    marker = "## 论文方法参考"
    if title in text:
        start = text.index(title)
        end = text.index(marker, start)
        text = text[:start] + text[end:]
    report.write_text(text.replace(marker, "\n".join(section) + "\n" + marker, 1))
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_diagnostics()
