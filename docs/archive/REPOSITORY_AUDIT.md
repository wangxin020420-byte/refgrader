# RefGrader 仓库审计与清理清单

> 审计日期：2026-08-31  
> 审计范围：全部受 Git 管理的文件做结构校验；对全部 Python 源码做语法和入口分类；对主方法、公共基准和 SAS-Bench 协议边界做依赖审查。实验历史、图片像素、OCR 缓存和虚拟环境不逐文件解释。

## 1. 全仓结构审计结果

| 项目 | 数量 | 结果 |
| --- | ---: | --- |
| 受控文件 | 5,342 | 均可访问 |
| Python 文件 | 80 | 语法解析错误 0 |
| JSON 文件 | 717 | 解析错误 0 |
| JSONL 文件 | 全部受控 JSONL | 逐行解析错误 0 |
| 空受控文件 | 0 | 通过 |
| 单元测试文件 | 25 | 均有明确覆盖对象 |

大量受控文件来自 `data/`、原始试卷和裁剪图像。它们属于数据快照或可复现实验输入，不能因“不是 Python 代码”而删除。

## 2. Python 文件功能分类

### 2.1 当前主方法，不能删除

| 文件 | 功能 |
| --- | --- |
| `main_pipeline.py` | 主流水线、并发、checkpoint、运行状态 |
| `step0_extract_ground_truth.py` | 教师分数图像提取工具 |
| `step3_rrd_generator.py` | 初始/候选评分准则生成 |
| `step4_vlm_grader.py` | 事实提取、独立评分、风险、路由、证据核验 |
| `calibration_utils.py` | A3WA、BND、残差、方向风险工具 |
| `rubric_semantics.py` | 评分准则语义契约和候选门禁 |
| `canonicalizers/engine.py` | 数值、结构、集合等答案规范化 |
| `model_runtime.py` | GLM/DeepSeek/视觉模型兼容配置 |
| `sample_quality.py` | 教师标签审核策略的只读应用 |
| `evaluate.py` | 私有数据和统一输出评估 |
| `portable_hash.py` | 跨平台哈希检查 |
| `ocr/*.py` | 可选 PaddleOCR 隔离运行与缓存 |
| `monitor.py` | 旧主流水线的运行监控 |

### 2.2 私有数据编排和审计，不能删除

| 文件组 | 功能 |
| --- | --- |
| `scripts/run_csbench.py` | 私有数据统一命令入口 |
| `scripts/prepare_csbench.py`、`embed_csbench_snapshot.py` | 外部数据导入和仓库快照 |
| `scripts/audit_csbench_snapshot.py`、`audit_question_splits.py` | 数据与 split 完整性 |
| `scripts/calibrate_a3wa.py`、`replay_calibration.py` | 校准和离线回放 |
| `scripts/audit_teacher_labels.py` | 教师标签疑似样本筛选 |
| `scripts/review_teacher_labels.py` | 本地人工审核界面 |
| `scripts/compile_sample_quality_policy.py` | 将审核决定编译为可选策略 |
| `scripts/restore_csbench_artifacts.py`、`evaluate_artifacts.py` | 跨设备恢复和 artifacts 评估 |
| `scripts/check_*.py` | 集成与特定证据检查 |
| `scripts/diagnose_*.py` | 风险可分性诊断，不参与正式路由 |

### 2.3 公共数据与论文对比，和主方法隔离

| 文件组 | 功能 |
| --- | --- |
| `benchmark_datasets/contract.py` | prepared 数据统一格式、哈希和审计 |
| `benchmark_datasets/adapters/asap_sas.py` | ASAP-SAS 适配 |
| `benchmark_datasets/adapters/mohler.py` | Mohler 适配 |
| `benchmark_datasets/adapters/sas_bench.py` | SAS-Bench 标签盲适配与无效记录排除 |
| `benchmark_datasets/protocols/mohler_acl2011.py` | Mohler ACL 2011 对比协议 |
| `scripts/benchmarks/prepare_dataset.py` | 公共数据准备入口 |
| `scripts/benchmarks/run_benchmark.py` | 公共数据主方法运行入口 |
| `scripts/benchmarks/run_mohler_acl2011.py`、`analyze_mohler_acl2011.py` | Mohler 对比和报告 |
| `scripts/benchmarks/aggregate_repeats.py` | 多次重复实验汇总 |
| `scripts/benchmarks/evaluate_sasbench_holistic_qwk.py` | SAS-Bench 整体 QWK |
| `scripts/benchmarks/project_sasbench_official_protocol.py` | 标签盲分步协议投影 |
| `scripts/benchmarks/filter_sasbench_official_common_subset.py` | 官方复现与 RefGrader 公平公共子集 |

审查结果：上述公共模块没有被 `main_pipeline.py` 导入；适配器只写入显式 `output-dir`，协议脚本只写入显式外部协议目录。它们不会覆盖 `data/csbench`、活动评分准则或活动 A3WA。

### 2.4 测试文件，当前都有用途

25 个 `test_*.py` 分别覆盖模型契约、A3WA、准则语义、规范化、OCR、视觉证据、Stage 2 得分合同、教师标签 UI、公共数据适配、Mohler 协议和 SAS-Bench 协议。删除测试不会立刻改变运行结果，但会失去回归保护，因此不属于清理对象。

`test_air.py` 名称像单元测试，但实际是手动 API 连通性检查；导入时不发请求。建议以后改名为 `scripts/check_zhipu_connectivity.py`，当前不必删除。

## 3. 兼容性审查

### 3.1 模型兼容

- `model_runtime.py` 默认仍是 GLM-5.2/GLM-4.6v。
- DeepSeek 通过命令参数或当前进程环境变量启用，不会永久改写 GLM 默认配置。
- 公共运行会把模型契约传给子进程并记录签名；不同模型不能混用 checkpoint。
- SAS-Bench 协议投影默认 DeepSeek，但它是独立命令，不会改变主方法默认提供商。

### 3.2 数据兼容

- 私有数据默认根目录为 `data/csbench`。
- 公共数据必须通过 `--prepared-dir` 显式选择。
- SAS-Bench 准则和 split 生成在外部 prepared 目录，不写入私有数据目录。
- 论文协议投影读取完整评分后的 `compare.csv`，不回写整体评分结果。

### 3.3 配置兼容

- 私有 active rubric/A3WA 使用内容哈希，错误组合会拒绝运行。
- 公共 SAS-Bench test-only 数据没有数据内 calibration/validation，不能假装生成数据内 A3WA。
- `step4_vlm_grader.py` 保留 `results_rrd_vlm/a3wa_calibration_config.json` 旧默认路径，用于兼容旧入口；统一编排会显式传入配置。

## 4. 明确发现的问题与处理

| 级别 | 问题 | 影响 | 处理 |
| --- | --- | --- | --- |
| P1 | SAS-Bench 投影失败只记录异常，不记录原始响应 | 无法判断单条失败原因 | 已增强失败记录；不改成功合同和主评分 |
| P2 | `CURRENT_PROGRESS.md` 过长且以历史英文记录为主 | 当前状态难以查找 | 新增 `PROJECT_STATUS.md` 与专项复现文档 |
| P2 | 根目录存在未引用旧脚本 | 容易误用旧流程 | 列入候选清理，不在本次无证据删除 |
| P3 | 本机 `tmp/`、PPT 检查旁文件未忽略 | Git 状态噪声 | 可按下表清理或补充忽略规则 |

## 5. 可删除性清单

### 5.1 从运行依赖看可删除，但建议先打标签归档

| 文件 | 依据 | 建议 |
| --- | --- | --- |
| `_fix3.py` | 无任何代码或入口引用，早期一次性修复脚本 | 在历史标签可追溯后删除 |
| `main_pipeline_old.py` | 无入口引用，当前入口为 `main_pipeline.py` | 删除或移到历史分支，避免误运行 |
| `clean_dataset_initial.py` | 无引用、硬编码图片名、导入即执行交互窗口 | 删除；所需能力已有受控数据准备流程 |

### 5.2 当前未引用，但仍可能用于旧数据重建

| 文件 | 原因 | 结论 |
| --- | --- | --- |
| `auto_process.py` | 固定坐标裁剪和去红笔，可能用于早期原卷重建 | 暂不删除，先确认旧数据不再重建 |
| `step0_extract_ground_truth.py` | 独立教师分提取工具 | 保留 |
| `test_air.py` | 手动模型连通性诊断 | 保留并计划重命名 |
| `run_experiment.sh` | 被 `scripts/run_csbench.py` 和 `monitor.py` 引用 | 不能删除 |

### 5.3 本机生成内容

| 路径 | 是否可清理 | 条件 |
| --- | --- | --- |
| `__pycache__/`、`.tmp/`、`tmp/` | 可以 | 确认没有正在使用的临时文件 |
| `logs/` | 可以归档后清理 | 不影响 checkpoint，但会失去日志证据 |
| `ocr_cache/` | 可以重建 | 重建耗时，正式运行前不建议删除 |
| `results_runs/` | 不能直接清理 | 先确认已完成、已发布且不再断点续传 |
| `outputs/*.inspect.ndjson` 和 PPT 渲染检查目录 | 可以 | 保留最终 PPTX 后再清理 |
| `venv/`、`.venv-ocr/` | 可以重建 | 会失去当前环境，非必要不清理 |

## 6. 清理原则

1. 先创建可追溯 Git 标签或确认目标文件已存在于历史提交。
2. 一次提交只做旧文件清理，不与模型、准则或实验逻辑修改混合。
3. 删除后运行完整单元测试和一个公共数据 dry-run。
4. `results_runs`、OCR 缓存和 artifacts 不与源码清理混在同一次操作。
5. 本文中的“可删除”只表示当前受控入口无依赖，不表示历史研究价值为零。

