# 目录职责与整理规则

## 日常入口

根目录以`README.md`、`PROJECT_STATUS.md`、`AGENTS.md`作为项目阅读入口。
评分Python模块、现有`test_*.py`及`SCALE4_REFGRADER_GUIDE.md`仍保留原位：它们受导入、命令或构建路径约束，不能为减少文件数量直接移动。

| 目录 | 应放内容 | 不应混入 |
|---|---|---|
| `scripts/` | 工作流、诊断、审核、归档工具；`benchmarks/`为对比适配 | 个人环境、实验缓存 |
| `prompts/`、`canonicalizers/`、`ocr/` | 评分提示与运行组件 | 汇报临时脚本 |
| `data/csbench/` | 版本化数据、当前准则、划分与审核工作资料 | 新方法的试验输出 |
| `experiments/run_registry/` | 小体积实验身份、冻结位置、状态索引 | 大型检查点 |
| `experiments/initial_rubric_ablation/` | 已冻结的消融输入；保持字节和路径 | 后续审核或实验结果 |
| `experiments/analysis/` | 有明确输入来源的离线报告；每项独立目录 | 修改后的基线文件 |
| `docs/guides/` | 可复制操作和使用条件 | 历史结果堆叠 |
| `docs/research/` | 研究与复现专题说明 | 当前生效配置 |
| `docs/archive/` | 历史交接、旧审计、旧状态及早期脚本文本 | 应被继续执行的当前指令 |
| `results_runs/`、`logs/` | 设备本地的工作结果，Git忽略 | 跨机唯一历史副本 |
| `ocr_cache/`、`venv/`、`.venv-ocr/` | 设备本地缓存和环境 | Git提交 |
| `.local-maintenance/` | 维护清单、恢复备份、临时文档构建 | 论文结果真源 |
| 同级`refgrader-artifacts/` | 独立run的可移植冻结结果，单独提交同步 | 当前审核文件或设备环境 |

`database/`、`results_rrd_vlm/`、`outputs/`、`benchmark_datasets/`、原始试卷及图片目录包含历史或其他协议的路径依赖。
这次仅建立目录职责，不递归搬迁这些目录；历史清单和resume指纹仍能指向原位置。

## 两个结果副本的关系

- `results_runs/...`：原设备的原运行目录，保留续跑和原始状态。
- `refgrader-artifacts/csbench/frozen_baselines/<run_id>/`：经过完整性核验的冻结副本，跨设备用于分析。
- `experiments/run_registry/`：指向冻结结果的小索引，不再复制一套大检查点。
- 原先的`csbench/<question_id>/...`与`csbench/experiments/...`历史归档继续保留，不清空或混入PC2。

## 本次实际迁移

8份操作指南进入`docs/guides/`；SAS-Bench记录进入`docs/research/`；旧交接、研发日志、审计、旧状态和旧README进入`docs/archive/`。
`_fix3.py`与`main_pipeline_old.py`按原字节保存为`docs/archive/legacy_code/*.py.txt`。
本机主要迁移及原始SHA清单位于`.local-maintenance/organization_20261022/moves.json`。
没有删除这些历史资料，也没有移动任何当前数据、检查点、模型环境或活跃审核文件。

## 后续新文件规则

每项分析明确写出run ID、数据政策、代码版本和输入SHA；不把`latest`当作可复现版本。
临时汇报资料进入`.local-maintenance/`；成品按用户指定位置输出，不混入研究代码。
新评估和新审核结果写独立版本；不要向已冻结目录添加备注、重算指标或临时文件，否则SHA校验会失败。
