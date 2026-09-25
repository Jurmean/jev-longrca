# Adaptive JEV-RCTA v3

当前实验协议：`jev-rcta-adaptive-v3.1`。本轮以检索召回、问题依赖和核验偏差为目标。
论文原理、会议信息、原文链接、对应实现与借鉴边界见 [文献与实现记录](../reports/jev_rcta_v3_literature_20260924.md)。

## 架构

**原文索引 → 多视角全局检索 → RRF 融合与 MMR 去冗余 → JEV 选择原文候选 → 按缺失事实补证、更新检索 → 根因预测 → 针对该根因判断责任角色并核验。**

1. 本地 BM25 对整个轨迹检索；任务要求、最终结果、已选原文与缺失证据类型是不同查询。RRF 只融合名次，不混用跨问题的 JEV 概率。
   v3.1 在检索前检查全部符合范围的原文能否直接装入预算；能装入时完整提供，避免短任务的关键步骤因没有关键词重合而消失。
2. 在请求字节预算内以 MMR 选择相关且较少重复的原文摘录。JEV 的单题分布决定候选优先级与保留集合；候选数量不由固定 top-k 硬截断。
3. 已选原文指针成为下一轮查询的依据。模型选择缺失证据类型，控制器据此查询上游、展开本步、查修复或反证。被引用的早期原因可以先成为待查假设，发现阶段不要求已经证明因果。
4. 最终根因题与责任角色题分轮。责任角色题能看到已选目标步骤和原始作者/交接证据，不再与根因题在同轮互相“猜答案”。
5. 最终核验不带之前的标签、概率或角色猜测，只阅读原始任务、结果和证据，分别判断错误来源与后续结果。未知关系不因反复提问而升级为确定关系。

`best_effort` 表示当前点预测；`supported` 表示满足本版在已检索证据范围内的核验条件。原文未穷尽，仍不能声称全局因果证明。`origin` 和 `outcome` 的完整概率与引用保存到 `verification`。

## 预算与版本

总预算沿用 v2：24 次逻辑请求、80,000 输入 tokens、240,000 请求字节；为最终选择和核验保留两次调用及预算。服务端实际 tokens 在响应后返回，预算估计不是费用硬保证。`global_retrieval_rounds`、`excerpt_steps_presented`、`action_segments_read` 分别记录检索轮数、看过摘录的步骤数和取过部分原文的段数，不能混称为整段全文已读。

v1、v2 推理模块及已有运行记录保留。共享运行器支持显式传入新协议，并冻结全部依赖代码；已停止的 v1 Mini 不恢复。真实测试固定配置运行，不根据参考答案在线更改问题或阈值。

## 离线与真实运行

```bash
python3 -m unittest discover -s tests -p 'test_rcta_adaptive_v3.py' -v
python3 scripts/evaluate_jev_rcta_adaptive_v3.py --demo --output results/jev_rcta_adaptive_v31_demo
python3 scripts/evaluate_jev_rcta_adaptive_v3.py --audit-only --mini --output results/jev_rcta_adaptive_v31_audit_20260924
```

演示为合成回复，审计为本地原文检查，均不读密钥、不调用 API。只有显式 `--live` 才运行真实调用：

```bash
python3 scripts/evaluate_jev_rcta_adaptive_v3.py --live --output results/jev_rcta_adaptive_v31_smoke
```

默认是五来源各一条、先连接检查。只有显式 `--mini` 才选择全部 Mini；五条预检若全部没有点预测则自动停止，额度不足或用户停止标记同样阻止继续。

可用 `--config` 指定 JSON 配置进行预先规划的消融。例如 `{"iterative_queries": 0}` 去掉检索反馈，`{"global_retrieval": 0}` 恢复树目录导航，`{"factored_verification": 0}` 恢复 v2 的角色题/核验方式。修改配置或代码须用新的结果目录，不能混合计分。

新入口：`scripts/evaluate_jev_rcta_adaptive_v3.py`；核心控制器：`scripts/jev_rcta_adaptive_v3.py`；本地检索：`scripts/rcta_retrieval.py`。

v3 六条开发样本没有取得整体提升，仍作为研究记录保留；v3.1 的短轨迹路径是随后修复。两版不混合计分，v2 仍保留为对照。结果与局限见 [研究报告](../reports/jev_rcta_v3_literature_20260924.md)。

v3.1 已独立复测 `vitabench__089`：预测步骤 8 正确，角色未确定、严格核验未通过。当前代码通过 99 项离线测试，另有 5 项因可选环境缺失跳过。该回归不能代表整体准确率改善，见 [单条结果](../reports/jev_rcta_adaptive_v31_regression_20260924_report.md)。

随后启动的 v3.1 Mini 在完成 100/200 条后按用户要求停止，禁止自动恢复。相同已完成样本上，本版根因 Exact 10%、±5 为 17%、角色准确率 35%；历史 JEV baseline 分别为 20%、35%、37%。本版有 79 条点预测、21 条弃答、0 条严格核验通过。该版本未体现性能优势，作为负面实验保留；这些是部分 Mini 的成绩，不能冒充完整 200 条结果。见 [停止后的审计报告](../reports/jev_rcta_adaptive_v31_mini_20260924_report.md)。
