---
annotations_creators:
- expert-generated
language:
- en
size_categories:
- 1K<n<10K
task_categories:
- other
pretty_name: LongRCA Bench
tags:
- agents
- multi-agent-systems
- failure-analysis
- root-cause-analysis
configs:
- config_name: default
  default: true
  data_files:
  - split: test
    path: longrca-full.parquet
- config_name: mini
  data_files:
  - split: test
    path: longrca-mini.parquet
---

# LongRCA Bench

A benchmark for diagnosing responsible roles and root causes in long-horizon agent failures.

- **1,140 observed, non-injected failed trajectories** from SWE-bench Pro, Terminal-Bench 2, TravelPlanner, VitaBench, and WebArena Verified.
- **Human annotations** identifying the responsible role, earliest decisive root-cause step, and supporting rationale.
- **LongRCA-Mini:** a fixed 200-trajectory subset, randomly sampled without replacement with 40 from each benchmark, preserving the original IDs and annotations.

Individual JSON files are grouped by benchmark in `data/full/` and `data/mini/`. The corresponding Parquet files are available at the repository root.

See the [paper](https://arxiv.org/abs/2608.15242) and [project page](https://longrca-bench.github.io/) for details. If you use either version, please cite our paper.

## Usage

```python
from datasets import load_dataset

full = load_dataset("CLoud5-real/longrca-bench", split="test")
mini = load_dataset("CLoud5-real/longrca-bench", "mini", split="test")
```
