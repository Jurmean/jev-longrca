# 转交与从零复现

源码包包含 JEV/Laya 代码、测试、固定依赖、数据/模型版本清单、本次参考报告及逐条结果 CSV。
不含 `.env` 密钥、虚拟环境、模型权重、数据正文、逐例预测缓存或原始调用档案。
`results/` 中只携带 JEV 的两个配置文件，供冻结代码校验和 Laya 对照报告使用，**不会因此跳过任何推理样本**。

## 怎么传给别人

直接发送 `dist/Jev-longRCA-source.zip` 和同目录的 `.zip.sha256` 文件即可，可用聊天工具、邮件附件或网盘。
对方解压后先阅读本文件。无需发送整个工作目录，也无需发送你自己的 API key。

需要重新打包时，在项目根目录运行：

```bash
python3 scripts/package_project.py
```

打包按明确的文件清单选取内容，检查可能的凭据，并在包内写入 `SHARE_MANIFEST.json`。
对方可在压缩包所在目录检查传输完整性：

```bash
sha256sum -c Jev-longRCA-source.zip.sha256
unzip Jev-longRCA-source.zip
cd Jev-longRCA
```

若需要长期协作，可将**解压后的源码包**放入自己的私有 Git 仓库，再邀请对方。
数据和权重由下面的固定版本下载命令获取，不通过 Git 提交。
源码包没有原始调用档案，因此能重新跑实验、查看参考分数，但不能直接离线复核本次全部原始响应；
需要原始档案时另行转交 `results/phase1/`、`results/full/`、`results/laya_full/` 和对应数据、模型文件，保留 Full 到 Mini 的相对符号链接。

## 环境与依赖

- 建议使用 **Linux x86_64 + Python 3.11**；本次 Laya 实测 Python **3.11.15**。
- **只跑 JEV**：Python 标准库即可，不需要 PyTorch、GPU，也不需要安装 `requirements.txt`。
- **跑 Laya**：本次实测 NVIDIA RTX 3080 Ti 12 GB，PyTorch 2.6.0 + CUDA 12.4 wheel；需要兼容的 NVIDIA 驱动。
  单张相同显卡可以顺序运行全部样本，不要求具备原机器的 8 张卡。其他平台未验证。
- `requirements.txt`：直接使用的核心依赖；`requirements-lock-cu124.txt`：本次全部传递依赖的版本快照，优先用于复现。
- 固定包版本不保证跨硬件逐位相同。远端 JEV 服务版本的可用性和行为也由服务商决定。

在项目根目录创建环境（下文命令均在激活后执行）：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

仅 JEV 可直接跳过安装第三方依赖。Laya 推荐安装完整锁定环境：

```bash
python -m pip install -r requirements-lock-cu124.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -m pip check
```

如果只需要核心依赖而不锁定全部传递依赖，可改用下面两条；它与完整锁定环境不是同等严格的复现：

```bash
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## 下载同一版本的数据

先 Mini 再 Full；Full 下载器会验证 Mini 确为 Full 的同一子集。两条命令都只需 Python 标准库。

```bash
python scripts/download_mini.py
python scripts/download_full.py
```

数据仓库为 `CLoud5-real/longrca-bench`，固定版本 `9f45acb66948d5d20c663b4ce4ec8ea5ab0076dd`。
Mini 200 条，Full 1,140 条；每个文件的摘要记录在 `data/manifest.json` 与 `data/full_manifest.json`。

随包的 `reports/` 是**原实验参考结果**。新报告会写回同名文件；若要保留前后对照，可先备份：

```bash
cp -a reports reference_reports
```

## 只复现 Laya：不需要任何 API key

下载器使用随包清单中的固定版本和 SHA-256，已有正确文件会直接校验跳过。
默认英文权重约 804 MiB；另外下载分词器、配置和该版本的上游推理源码。
清单中的 multilingual/typed-decisions 项只有小配置文件，**不下载这些变体的权重，也不运行它们**。

```bash
python scripts/download_laya.py
python scripts/download_laya.py --verify-only
python -m unittest discover -s tests -v
```

选择自己机器上的空闲 GPU 编号。以下以 GPU 0 为例，先运行短文本健康检查和五来源冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_laya_health.py
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_laya.py --smoke --output results/laya_smoke
```

正式跑 Full，下面两种方式择一；调度器用物理 GPU 编号，**不要同时设置 `CUDA_VISIBLE_DEVICES`**：

```bash
# 单卡
python scripts/run_laya_full.py --gpus 0

# 或使用两张空闲卡
python scripts/run_laya_full.py --gpus 0,1
```

如果所在环境必须通过 `CUDA_VISIBLE_DEVICES` 提供显卡，可绕过自动调度器直接单卡执行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_laya.py
```

成功请求和预测会保存；使用**原命令、原分片数量和映射**恢复即可。不要在有活跃进程时重复启动相同分片。
正式推理结束后运行：

```bash
python scripts/report_laya.py
```

这个命令检查 1,140 条输出、每次候选选择、文本覆盖与 token 记录，然后生成：

- `reports/laya_full_report.md`：Full、Mini 子集、各来源、JEV 对照和异常复查。
- `reports/laya_full_metrics.json`、`laya_full_predictions.csv`、`laya_result_audit.json`：可分析结果和审计。
- `results/laya_full/`：本次机器上新生成的配置、模型调用和预测。

Laya 单独复现时，JEV 对照来自源码包中的**历史参考分数**；不需要付费重跑 JEV。
本实验测试的是英文权重 + 8K 上下文适配，不是默认 512-token 或 multilingual 配置。
参考结果为角色 14.47%、根因精确 0%、±5 命中 6.14%，存在强烈 step 0 偏置；并非预期的高性能方案。
原运行使用 8 张卡，含模型加载约 47.4 分钟；单卡耗时更长，不应按原耗时判断是否卡住。

## 复现 JEV Mini → Full

对方使用自己的 TypeSafe/JEV API key。这部分会实际调用付费服务。
模型名称固定为 `jev-1.13.0`；若服务商已不提供该版本，不要悄悄换模型并称为同一次实验。

```bash
cp .env.example .env
```

在本机编辑 `.env`，把 `TYPESAFE_API_KEY` 的占位值替换为自己的 key，或者通过环境变量提供该值。
随后按顺序执行，首次必须先完成 Mini，Full 才能复用其中 200 条：

```bash
python scripts/evaluate.py --audit-only
python scripts/evaluate.py --workers 4
python scripts/verify_results.py
python scripts/make_report.py

python scripts/evaluate_full.py --audit-only
python scripts/resume_full.py --workers 4
python scripts/verify_results.py --full
python scripts/make_full_report.py
```

Full 恢复入口保留原模型、提示词与候选协议，增加传输退避，并在余额不足时停止新请求。
参考 Full 分数为角色 44.56%、根因精确 14.30%、±5 命中 28.16%；报告中的原实验调用次数、时间和费用只描述原运行。
若同时重跑 JEV 和 Laya，建议先完成 JEV 报告，再生成 Laya 报告，使最终对照引用新生成的 JEV 分数。

## 复现范围与常见情况

- 所有命令都在项目根目录执行。数据/权重下载需要联网；准备完成后 Laya 推理设置为离线运行。
- 仅 JEV 环境运行测试时，Laya 测试会自动跳过；装好依赖并下载模型后再执行全部测试。
- 新环境缺少模型文件时，先运行 `download_laya.py`；不要把原机器的 `.venv-laya` 目录直接复制过去。
- 出现配置或源码摘要不匹配时，表示代码/协议与已有结果不同。保留旧结果，使用干净解压目录重跑，不要删除摘要校验来混用结果。
- 要独立重跑而不是继续缓存，同样使用一个干净解压目录。源码包没有逐例缓存，所以首次运行会真实重新计算。
- 该项目是固定候选筛选基线，不是 LongRCA 论文 RCTA 方法的实现。更多方法和评分细节见 `README.md`。
