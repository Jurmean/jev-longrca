# JEV × LongRCA-Mini：第一阶段评测

**转交和从零复现请先看 [REPRODUCE.md](../REPRODUCE.md)。**
`requirements.txt` 是 Laya 的核心依赖；`requirements-lock-cu124.txt` 固定本次完整 CUDA 12.4 环境。
仅运行 JEV 不需要第三方依赖。`python3 scripts/package_project.py` 可生成不含密钥、虚拟环境、数据正文和大权重的源码分享包。

使用 TypeSafe 官方 `jev-1.13.0` 在官方固定 200 条 LongRCA-Mini 上做离线失败归因。仅使用 JEV 的 Choice 接口，不调用其他生成模型，不训练、不用测试标签调参。

## JEV-RCTA Adaptive：概率驱动的多轮取证

最新的论文方法实验分支是 **v3.1**：按视图大小选择完整原文或 RRF/MMR 全局检索 → 结构化多轮检索反馈 → 分轮预测根因与角色 → 原文核验上下文隔离。总预算沿用 v2。入口为 `scripts/evaluate_jev_rcta_adaptive_v3.py`，见 [使用说明](../docs/ADAPTIVE_RCTA_V3.md) 和 [论文到代码的对应记录](../reports/jev_rcta_v3_literature_20260924.md)。v3 的六条实测没有整体优于 v2；v3.1 随后修复短轨迹的证据漏筛，结果分开保存，不将研究分支宣称为已证实的改进。

v3.1 的单条回归定位到正确步骤，角色仍未确定、严格核验未通过；不能代替完整对照评测。见 [回归报告](../reports/jev_rcta_adaptive_v31_regression_20260924_report.md)。

后续 Mini 在完成 100/200 条后按用户要求停止，不自动恢复。相同 100 条上 v3.1 根因 Exact 为 10%，历史 JEV baseline 为 20%；本版未体现性能优势，保留为负面实验。见 [Mini 停止报告](../reports/jev_rcta_adaptive_v31_mini_20260924_report.md)。

当前优化版入口是 `scripts/evaluate_jev_rcta_adaptive_v2.py`：无损原文索引 → 树形目录与概率导航 → 按需读取、打标签 → 集中核验一个候选 → 预留预算做最终选择与独立核验。未探索的分支保留，点预测与已核验子集分别报告。

v1 Mini 在完成 45 条、全部弃答后按用户要求停止，原结果与代码归档保留。新版减少初始扫描成本、候选分散和最终判断被预算跳过的问题；真实收益需单独验证。完整配置与输出含义见 [Adaptive v2 使用说明](../docs/ADAPTIVE_RCTA_V2.md)，[v1 文档](../docs/ADAPTIVE_RCTA.md)用于复现。先离线运行：

```bash
python3 -m unittest discover -s tests -p 'test_rcta_adaptive_v2.py' -v
python3 scripts/evaluate_jev_rcta_adaptive_v2.py --demo --output results/jev_rcta_adaptive_v2_demo_20260924
python3 scripts/evaluate_jev_rcta_adaptive_v2.py --audit-only --mini --output results/jev_rcta_adaptive_v2_audit_20260924
```

演示使用合成回复，不调用 JEV，也不是准确率实验。只有显式使用 `--live` 才读取 API 配置并发起真实请求；真实运行先做连接预检，保留额度耗尽即停与精确缓存恢复。默认阈值未经任务校准，不声称具有整棵搜索树的覆盖率保证。

![Adaptive JEV-RCTA architecture](../figures/jev-rcta-adaptive-main.png)

这张主图由 imagegen 按 v1 实现生成；v2 的目录导航、集中核验和双层输出以新版文档为准。图中日志与概率柱状图为流程示意，不是实验结果。[原始提示词](../figures/jev-rcta-adaptive-main-prompt.txt)与[局部修订提示词](../figures/jev-rcta-adaptive-main-edit-prompts.txt)一并保存。

## JEV-RCTA 历史实验架构

新增独立入口 `scripts/evaluate_jev_rcta.py`，方法实现在 `scripts/jev_rcta.py`。这是受 RCTA 启发的纯 JEV Choice 改造，未经训练，也不使用生成式摘要。原始 JEV、Laya 推理流程与历史结果保持原样。

流程为：无损分段与前置重叠 → 每段选择可疑步骤和原文证据锚点 → 组成按时间排列的证据卡 → 分组保留多个候选 → 检索上游交接 → JEV 判断上游引入/本步引入/已修复/无关/证据不足 → 有界向前追溯 → 独立预测角色与根因步骤。关系分类只作为假设，不能当作已验证因果；“已修复”或“不确定”不直接删除候选。不同请求的选项概率不跨请求直接排序。

默认每段 24,000 UTF-8 字节、最多 60 个记录片段，单条超长记录按 8,000 字节无损拆分；另附最多 3 个前置步骤的节选。每段最多保留 3 个候选，追溯前缩减至最多 16 个，回溯深度最多 2，最终候选最多 8。完整请求限制为 62,000 UTF-8 字节；这是保守预算，不是 tokenizer 的精确上下文保证。初筛覆盖全部原文，但后续使用有明确省略标记的节选，仍可能丢失证据。极长轨迹的证据卡索引超预算时，模型输入会明确标记省略，完整卡片保留在预测文件中。

仅依赖 Python 标准库。先离线核验，再跑最短样本和按文本长度中位数选择的样本；两组各来源一条、各 5 条，不按答案标签选样本：

```bash
python3 -m unittest discover -s tests -p 'test_jev_rcta.py' -v
python3 scripts/evaluate_jev_rcta.py --audit-only --output results/jev_rcta_audit
python3 scripts/evaluate_jev_rcta.py --smoke --output results/jev_rcta_smoke
python3 scripts/evaluate_jev_rcta.py --pilot --output results/jev_rcta_pilot
```

不指定 `--smoke` 或 `--pilot` 才运行 Mini 200 条。当前任务先验证可行性，不自动启动全量。所有输出独立保存为 `config.json`、`selection.json`、`calls/`、`predictions/`、`progress.json` 和完成后的 `summary.json`；配置冻结方法、传输、运行入口与基础模块的代码摘要。相同命令可恢复，成功调用按精确请求摘要复用。

传输入口 `scripts/jev_rcta_client.py` 遇到 HTTP 402 或明确额度不足立即停止，写入 `balance_stop.json`，不再重试或继续其他样本；即使额度错误包在 HTTP 429/503 中也如此。该停止标记会阻止后续启动继续请求。只有在用户明确确认充值或提供新 API 配置后，才手动归档停止标记并续跑，不自动换账户。其他临时错误最多重试 3 次，等待 5/10/15 秒，最终失败则停止并保留已成功的调用。

少量样本只检查接口、分段、候选筛选和追溯能否运行，不能证明优于原基线。评分严格在推理完成后读取参考标签，并报告初筛、追溯前、扩展后和最终候选的召回率。

### 独立托管端点 Full 评测

用户后续授权使用 `jevtypesafeai.com` 的托管 key，在确认可用后运行 Full 1,140 条。该站点自述为独立代理，端点是 `https://jevtypesafeai.com/api/v1/decide`，不是官方直连地址；请求和返回模型均固定为 `jev-1.13.0`。单独配置文件 `.env.jev-hosted` 保存 `JEV_HOSTED_API_KEY`，权限 600，并被版本控制与分享包排除。不要把 key 写入命令行或报告。

真实预检发现该代理要求 Choice 至少两个选项，因此方法升级为 `jev-rcta-choice-v2`：单一选项由程序确定性返回，不发送 API，也不声称具有模型置信度。其余推理参数不变。v1 尝试目录保留；v2 只复用输入摘要、端点、模型一致的成功调用，并重新计算预测。

```bash
python3 scripts/probe_jev_hosted.py
python3 scripts/run_jev_rcta_hosted_full.py --workers 4 --output results/jev_rcta_hosted_full_v2
python3 scripts/report_jev_rcta_hosted.py --output results/jev_rcta_hosted_full_v2
```

Full 调度器先要求接口控制测试通过，再串行完成五来源各一条最短真实轨迹；五条全部成功后才扩展全量。预检样本选择自 Full，只依赖日志长度。支持 `--preflight-only`。推理保存所有请求与响应，同一命令可恢复；配置、代码与数据摘要变化时拒绝混用结果。并发额度停止由共享事件和持久标记控制：停止新请求与等待重试，已发出的请求可能完成。其他最终失败同样停止，不伪造预测。

费用使用托管 API 每次成功响应的 `cost_usd`，不套用原官方直连单价。报告包含完成范围、候选召回、同样本历史 JEV 基线对照和逐条预测 CSV；离线重放验证每次请求摘要及最终预测。若额度不足或错误导致未完成，报告明确标识为部分样本结果，不能作为 Full 指标。

本次服务出现过无响应连接中断，使用 `python3 scripts/resume_jev_rcta_hosted.py --workers 4 --output results/jev_rcta_hosted_full_v2` 续跑。恢复入口只对该类连接错误增加 5/10/15 秒有界重试，保留原请求与推理代码，适配器摘要记录在 `transport_recovery.json`。无响应请求可能产生服务端费用，成功响应的费用合计不等同账户账单。该入口同样遵守额度停止标记。

## Laya 本地对照

`scripts/evaluate_laya.py` 使用官方 `convaiinnovations/laya` 英文根目录权重，版本固定为 `1c5edc17a7acd8701df6fc341c0d179f1c62c982`。模型在 `models/laya/`，下载文件的摘要见 `reports/laya_download_manifest.json`。此入口不读取 API key、不调用 JEV、不训练。原始 JEV 推理代码保持不变。

Laya 默认 512 tokens。本次显式扩展为编码器支持的 8192 tokens，逐题确保提示词、所有选项和提供的日志没有被 SDK 静默截断。初筛按真实 tokenizer 预算分段、每段最多 16 个步骤，无损覆盖原始内容；复核保留原有 top-3 / 8 候选 / 交接扩展，并显式缩短超预算节选。共享首尾上下文预算为 2048 tokens。仅一个选项时确定性返回，不计模型 forward。属于适配后流程比较，不能隔离模型能力与输入差异；英文权重对中文数据的语言限制需单独考虑。

在独立 Python 3.11 环境安装 CUDA 12.4 的 `torch==2.6.0` 和 `transformers==5.0.0`、`safetensors`、`numpy`。模型加载完全离线，使用 float32 权重与 CUDA bfloat16 autocast；需要可用 NVIDIA GPU。运行与核验：

```bash
HF_HUB_OFFLINE=1 .venv-laya/bin/python -m unittest discover -s tests -p 'test_laya.py' -v
CUDA_VISIBLE_DEVICES=0 .venv-laya/bin/python scripts/evaluate_laya.py --smoke --output results/laya_smoke
CUDA_VISIBLE_DEVICES=0 .venv-laya/bin/python scripts/evaluate_laya.py
python3 scripts/report_laya.py
```

多 GPU 时分别指定 `--shard 0 --shards N` 至 `--shard N-1 --shards N`，每个进程绑定一个空闲 GPU。成功请求和预测支持精确摘要匹配后的断点恢复。全量报告要求 1,140 条全部成功并逐条核验原始文本覆盖、预测来源和 token 计数；输出 `reports/laya_full_report.md`、`laya_full_metrics.json`、`laya_full_predictions.csv`、`laya_result_audit.json`。原始请求与响应为 `results/laya_full/calls/` 中的 gzip JSON，Mini 200 条结果直接从本次 Full 预测提取。

本次历史运行使用 GPU 1、3–9。为方便转交，调度入口现支持 `.venv-laya/bin/python scripts/run_laya_full.py --gpus 0`（单卡）或 `--gpus 0,1`（多卡）；不指定时，新运行只选一张空闲卡，断点恢复沿用原映射。不要同时设置 CUDA_VISIBLE_DEVICES。该入口只调整调度，冻结的推理代码保持不变；逐分片输出日志并记录全量起止时间。环境版本完整记录在 `reports/laya_environment.txt`。日志中的 tokenizer 长度警告可能来自分段前的预算探测，实际 forward 前另有强制长度及完整性断言。

本次 Full 已完成并审计：角色 14.47%、根因精确 0%、±5 命中 6.14%。输出出现明显 step 0 偏置，详见报告的异常复查。追加的 `scripts/check_laya_health.py` 使用 3 个短英文控制样例、选项换序及两种 max_len 配置，共 18 次正确；仅验证基本本地推理，不验证长日志能力。`scripts/diagnose_laya_results.py` 将诊断加入报告，不修改评测预测。此结果限定于本次配置，不能作为所有 Laya 变体的结论。

## Full 全量扩展

沿用原始 Mini 推理脚本的精确代码摘要与所有参数，扩展至官方 Full 1,140 条；数据版本不变。Full 中的 200 条 Mini 文件逐字节相同，复用已核验的真实预测和原始调用，其余 940 条新增运行。Mini 文件及结果保留。

```bash
python3 scripts/download_full.py
python3 scripts/evaluate_full.py --audit-only
python3 scripts/evaluate_full.py --workers 8
python3 scripts/verify_results.py --full
python3 scripts/make_full_report.py
```

全量结果为 `reports/full_report.md`、`reports/full_metrics.json`、`reports/full_predictions.csv`。`data/full_manifest.json` 固定全部数据摘要，`results/full/config.json` 同时固定推理脚本及全量调度脚本的摘要。`results/full/calls/` 中复用样本通过相对符号链接指向 `results/phase1/calls/`，归档时应一并保留这两个结果目录。断点续跑使用同一命令，成功请求和预测不会重复调用。公开单价估算在报告中分别列出累计用量和新增用量。

本次遇到服务端间歇性 403、503 和固定模型名路由错误后，使用 `python3 scripts/resume_full.py --workers 4` 恢复。该入口仅增加有界传输重试（15/30/45 秒），保留同一模型版本、端点、请求内容和评分协议；返回模型版本仍严格校验。重试记录保存在 `results/full/transport_retries.jsonl`，适配器摘要在 `transport_recovery.json`。`errors/` 保存历史失败，判断最终是否恢复应以同 ID 的成功预测和最后 `summary.json` 为准。

按用户要求，恢复入口在检测到 HTTP 402 或明确的余额/credits 不足错误时立即停止新请求和待重试请求，保存 `results/full/balance_stop.json`。已发出的请求可能完成，但不会再启动下一次请求。此时等待用户提供新的 API 配置，不自动更换账户或提高额度。服务稳定后本次并发数调整为 6；请求内容和模型版本不变。

## 数据与输出

- 官方数据：[CLoud5-real/longrca-bench](https://huggingface.co/datasets/CLoud5-real/longrca-bench)，`mini/test`，5 个来源各 40 条。
- 固定版本：`9f45acb66948d5d20c663b4ce4ec8ea5ab0076dd`。`data/manifest.json` 保存每个文件的 SHA-256。
- `reports/phase1_report.md`：完整运行后生成的中文效果报告。
- `reports/phase1_metrics.json` / `reports/phase1_predictions.csv`：可直接分析的指标和 200 条预测对照。
- `results/phase1/summary.json`：总体、各来源、处理路径指标及逐例对照。
- `results/phase1/predictions/`：逐条预测、置信度、候选集合和用量。
- `results/phase1/calls/`：完整请求和原始响应（不含认证头或密钥），可审计和断点续跑。

## 运行

Python 3.8+，仅依赖标准库。将 `TYPESAFE_API_KEY` 或 `JEV_API_KEY` 配置在环境变量或私有 `.env`。不要提交 `.env`。

```bash
python3 scripts/download_mini.py
python3 scripts/evaluate.py --audit-only
python3 -m unittest discover -s tests -v
python3 scripts/evaluate.py --smoke --workers 3
python3 scripts/evaluate.py --workers 6
python3 scripts/verify_results.py
python3 scripts/make_report.py
```

上述 smoke 选择各来源按序列化字节数最短的一条，只验证接口；不代表最终效果。全量运行复用这 5 条，最终仍是 200 个唯一实例。程序固定代码摘要与协议，修改推理脚本后必须使用新的 `--output` 目录。失败请求不会伪造预测；成功中间请求落盘，可以重跑恢复。

## 第一阶段方法：JEV Choice 候选筛选

JEV 是结构化判断模型，不生成根因解释。此实验评估责任角色和根因步骤，不把人工 `mistake_reason` 输入模型，也不将置信度当作解释。

1. **短轨迹**：序列化历史不超过 44,000 UTF-8 字节且步骤数不超过 255，一次请求读取完整轨迹，独立回答责任角色和根因步骤。
2. **长轨迹初筛**：依原始顺序按约 44,000 字节分段，每段最多 120 个记录片段。超过 16,000 字节的单步无损拆片，保留原始步骤编号。所有轨迹文本均进入初筛；分段边界不重叠。每段共享任务开头和轨迹末尾的有标记节选，从段内步骤中按 JEV 概率保留 3 个候选。
3. **交接候选**：对每个初筛候选加入最近的前置交接步骤，优先匹配接收角色，无法匹配则取最近交接。此规则只读日志角色名。
4. **候选缩减**：按时间顺序每 8 个候选组成一组，每组保留 3 个，递归至不超过 8 个候选。候选证据含该步骤首尾共约 2,400 字节、相邻步骤各约 500 字节、最近交接约 1,600 字节，所有节选明确标记省略。
5. **最终判定**：在候选证据与共享背景上，分别用两个 Choice 选择责任角色和根因步骤。角色选项由整条轨迹出现的角色构造，不从所选步骤反推。

所有阈值、完整提示词及模型版本见 `scripts/evaluate.py` 和运行目录 `config.json`。这是为了适应 JEV 上下文限制建立的基线流程，不是论文 RCTA 的复现。段间远距离因果关系、超长记录分片、候选丢失及最终证据节选会影响结果；报告同时输出候选召回率以定位瓶颈。按字节设定的是保守输入预算，不是官方 tokenizer 的精确 token 数。

## 评分口径

遵循[论文 §6.1](https://arxiv.org/html/2608.15242v1)：

- 责任角色准确率：独立预测；统一大小写、空白及显式 `(-> …)` 交接后缀。
- 根因步骤 Exact：原始 0-based 步骤编号精确相等。
- 根因步骤 ±5：与标签的绝对距离不超过 5。
- Root MAE：按来源样本量加权的有效步骤预测绝对误差。若任何来源没有有效步骤，整体 MAE 为 null。
- 无效/缺失预测在准确率分母中计错；角色和步骤互不影响各自评分。

Mini 均衡抽样与论文全量数据的来源比例不同，推理模型和方法也不同，不能将本次分数与论文全量表格直接比较。发布数据只有 `question_ID/history/mistake_agent/mistake_step/mistake_reason`；不额外提供独立 evaluator outcome。推理只知轨迹已失败，并使用日志中实际可见的末尾信息，绝不从人工 rationale 提取失败提示。

## 官方资料

- [JEV 简介](https://docs.typesafe.ai/introduction)
- [HTTP API](https://docs.typesafe.ai/api)
- [Choice 选项与概率](https://docs.typesafe.ai/primitives/choice)
- [模型版本、上下文、价格](https://docs.typesafe.ai/models)：核查于 2026-09-21，当前输入 $0.042 / million tokens，输出免费；费用报告为公开单价估算，不是账单。
