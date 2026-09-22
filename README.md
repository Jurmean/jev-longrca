# JEV × LongRCA

使用 JEV Choice 接口对 LongRCA Bench 的失败轨迹进行责任角色归因与根因步骤定位，包含原始候选筛选基线、RCTA 启发的实验架构和 Laya 本地对照。

**当前 JEV-RCTA 只完成 64 / 1,140 条，不能作为 Full 结果。** 已完成结果通过原始请求与响应的离线重放审计。原始 JEV 基线和 Laya 对照分别完成了 Full 1,140 条。

## 架构

![JEV-RCTA architecture](figures/jev-rcta-main.png)

JEV-RCTA 用固定选项判断组织长日志归因：

1. 无损分段，附带前置重叠上下文。
2. 选择可疑步骤及关键原文，形成抽取式证据卡。
3. 分组比较并保留多个候选。
4. 检索上游交接指令，判断上游引入、本步引入、已修复、无关或证据不足。
5. 有界向前追溯，独立选择责任角色和根因步骤。

该方法不训练、不生成摘要，也不是论文 RCTA 的等价复现。关系判断是模型假设，不能视为已验证因果。初筛覆盖全部原文，后续节选仍可能丢失证据。协议 `jev-rcta-choice-v2` 对单一选项直接确定结果，不调用 API，也不赋予模型置信度。

## 当前结果

同一批已完成的 64 条样本中，60 条来自 SWE-bench Pro，其他四来源各 1 条，存在明显来源偏差。

| 方法 | 角色准确率 | 根因精确准确率 | ±5 命中 |
|---|---:|---:|---:|
| 历史 JEV Choice 基线 | 32.81% | 7.81% | 25.00% |
| JEV-RCTA | 28.12% | 10.94% | 28.12% |

本次托管评测保存 2,642 次成功响应，输入约 2,525.7 万 tokens，成功响应报告费用约 $10.61。它还包括未完成轨迹的中间调用，不是 64 次 API 调用。余额接近零后接口返回 `null` 而非 HTTP 402，因此控制端主动暂停；该费用合计不是账单。所有在线调用已停止。

- [JEV-RCTA 部分评测报告](reports/jev_rcta_hosted_full_report.md)、[逐条结果](reports/jev_rcta_hosted_predictions.csv)、[汇总指标](reports/jev_rcta_hosted_metrics.json)
- [原 JEV Full 报告](reports/full_report.md)、[Laya Full 报告](reports/laya_full_report.md)
- [完整实验说明](docs/EXPERIMENT_NOTES.md)、[从零复现与环境](REPRODUCE.md)

## 本地检查

JEV 部分仅依赖 Python 标准库，建议 Python 3.11。默认测试不调用付费 API；Laya 测试需要其独立环境与模型资源，在不具备依赖时跳过。

```bash
python3 -m unittest discover -s tests -v
```

数据不放进 Git；下载固定版本后校验摘要：

```bash
python3 scripts/download_mini.py
python3 scripts/download_full.py
python3 scripts/evaluate_jev_rcta.py --audit-only --output results/local_audit
```

官方数据：[CLoud5-real/longrca-bench](https://huggingface.co/datasets/CLoud5-real/longrca-bench)，固定版本 `9f45acb66948d5d20c663b4ce4ec8ea5ab0076dd`。数据与模型遵循各自上游条款。

## 运行 JEV-RCTA

以下推理命令会调用付费 API。先跑少量样本核对调用量与费用，不应把“轨迹数”当作“请求数”。

官方直连使用 `.env.example` 中的配置，或设置 `TYPESAFE_API_KEY` / `JEV_API_KEY`：

```bash
python3 scripts/evaluate_jev_rcta.py --smoke --output results/my_smoke
python3 scripts/evaluate_jev_rcta.py --pilot --output results/my_pilot
```

独立托管端点使用 `.env.jev-hosted.example`，复制为私有 `.env.jev-hosted` 并填入自己的 key，或设置 `JEV_HOSTED_API_KEY`。该站点是独立代理，费用不能套用官方直连历史单价。

```bash
python3 scripts/probe_jev_hosted.py
python3 scripts/run_jev_rcta_hosted_full.py --preflight-only --workers 1 --output results/my_hosted_run
```

确认预检有效、可接受费用且余额充足后，同一个输出目录可继续 Full：

```bash
python3 scripts/run_jev_rcta_hosted_full.py --workers 4 --output results/my_hosted_run
python3 scripts/report_jev_rcta_hosted.py --output results/my_hosted_run
```

调度器要求控制测试通过，再完成五来源各一条真实轨迹，才启动全量；门槛是输出有效，不以人工标签准确率调参。成功请求按精确输入摘要复用，配置、代码或数据不同会拒绝混用结果。

**已知限制：** 传输代码能自动处理明确的额度不足错误，但不把余额字段 `null` 自动视为额度不足，也没有本地累计费用硬上限。本次余额异常由控制端监测后主动暂停。使用该独立代理时须监测余额与累计费用；遇到不明余额应暂停并核对后台。

## 仓库内容与审计范围

- `scripts/`、`tests/`：方法、调度、报告与离线测试。
- `reports/`：参考报告、汇总指标、预测 CSV；托管指标 JSON 已移除含原文节选的逐例详情。
- `figures/`：架构图与生成提示词。
- `data/*manifest.json`：数据版本和摘要，不含轨迹正文。
- `reproducibility/jev-rcta-v2/`：历史方法配置、样本清单、分段审计和恢复适配器摘要。
- `results/phase1/config.json`、`results/full/config.json`：历史基线配置，不含预测和调用缓存。
- `REPOSITORY_MANIFEST.json`：导出文件与摘要。

仓库不含 API key、模型权重、虚拟环境、原始数据、原始 API 请求/响应缓存。仓库内的审计报告是历史审计结果；要重新重放该次真实调用，仍需另外取得原始调用档案。自行重新运行会产生新结果和费用。

GitHub Actions 只运行离线测试，不配置或调用 API key。

## 参考

- [LongRCA Bench 论文](https://arxiv.org/abs/2608.15242)
- [官方 JEV Choice 文档](https://docs.typesafe.ai/primitives/choice)
- [本次使用的独立托管 API 文档](https://jevtypesafeai.com/docs)

本仓库是评测与方法实验项目，不隶属于 TypeSafe AI 或 LongRCA Bench 作者团队。
