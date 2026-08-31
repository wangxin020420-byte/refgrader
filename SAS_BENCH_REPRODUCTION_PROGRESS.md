# SAS-Bench 复现与对比实验进展

> 状态日期：2026-08-31  
> 目标：在不读取教师整体分和教师分步错因的条件下，用 RefGrader 完成整体评分，并按 SAS-Bench 官方 CCS/ECS 协议进行公平比较。

## 1. 数据与版本

| 项目 | 当前值 |
| --- | --- |
| SAS-Bench 源代码固定版本 | `89e572c` |
| 原始任务文件 | 12 |
| 原始记录 | 4,109 |
| 原始题目上下文 | 1,112 |
| 无效记录 | 16（`manual_label > total`） |
| 正式测试记录 | 4,093 |
| 正式有效题目上下文 | 1,111 |
| 公共标签用于调参 | 否 |

准备后的 split 只有 `test`，没有 calibration/validation。16 条无效记录通过 `quality_control/excluded_records.jsonl` 保留审计依据，不进入评分和指标。

## 2. 两条实验路线

### 2.1 官方代码复现路线

使用 SAS-Bench 官方仓库的评分、后处理、QWK/CCS/ECS 代码。当前实验模型为 DeepSeek 官方兼容接口中的 `deepseek-chat`，温度 0、关闭思考、零样本、使用官方 guideline。

这是一项“官方代码 + 当前模型”的复现，不是论文 DeepSeek-V3 数值的严格重复。已完成 4,109 条官方源记录的预测与静默回退修复。已知宏平均结果：

| 指标 | 论文 DeepSeek-V3 | 当前官方代码复现 |
| --- | ---: | ---: |
| CCS | 74.11 | 76.86 |
| ECS | 54.00 | 57.39 |

模型版本差异、API 实现、生成稳定性和无效记录处理都可能造成与论文数值不同，因此报告中必须保留协议说明。

### 2.2 RefGrader 主方法路线

使用独立 prepared 目录运行当前主方法，文本模型为 DeepSeek 官方 `deepseek-chat`、温度 0、关闭思考；视觉模型配置保留兼容性，但 SAS-Bench 为文本数据，不触发视觉输入。

整体评分已完成并通过 4,093 条完整性门禁：

| 方法 | N | MAE | RMSE | Pearson | SER2 | Bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single | 4,093 | 2.1234 | 3.3707 | 0.8202 | 33.74% | 0.5649 |
| Average | 4,093 | 2.0816 | 3.2991 | 0.8252 | 33.94% | 0.5685 |
| 3WD-Core | 4,093 | 2.0794 | 3.2971 | 0.8265 | 33.81% | 0.6194 |
| 3WD | 4,093 | 2.0794 | 3.2971 | 0.8265 | 33.81% | 0.6194 |

整体 QWK 对齐结果：

| 方法 | 12 任务宏平均 QWK | 答案数加权 QWK |
| --- | ---: | ---: |
| Single | 76.17% | 76.24% |
| Average | 76.58% | 76.95% |
| 3WD-Core | 76.71% | 77.02% |
| 3WD | 76.71% | 77.02% |

论文没有以可直接读取的数值表报告这组整体 QWK，因此不能用雷达图人工读数冒充精确原文数值。

## 3. 官方 CCS/ECS 对齐为什么还未完成

SAS-Bench 的 CCS/ECS 不只使用整体分：

- CCS 需要整体预测分与教师整体分。
- ECS 需要每个原始步骤的预测得分和官方错因类别。
- RefGrader 现有结果包含整体分，但不天然包含 SAS-Bench 官方格式的 `pred_steps`。

`project_sasbench_official_protocol.py` 因此执行一个独立、标签盲的协议投影：只向模型提供题目、参考答案、解析、满分、学生分步作答、官方 guideline 和错因类别定义；明确禁止教师 `manual_label`、教师步骤 `label` 和教师 `errors` 进入提示。

该投影不修改 RefGrader 的整体分。整体 `pred_label` 直接来自已完成的 `3WD` 结果；模型只生成官方协议需要的分步得分和错因。

## 4. 当前冒烟状态

| 项目 | 状态 |
| --- | --- |
| 任务覆盖 | 12/12 |
| 计划记录 | 12 |
| 旧策略 checkpoint 成功 | 11 |
| 失败 | 1 |
| 是否可进入全量 | 否 |

输出目录：

```text
D:\Users\王鑫020420\Desktop\refgrader-public-datasets\sas_bench\protocol_comparison\refgrader_smoke12_RefGrader_3WD_Deepseek_t0_smoke_v1_Scored
```

旧策略的 11 条 checkpoint 保留作为失败诊断记录。由于修复重试现在会携带上一次合同错误，投影策略版本已升级为 2；为防止混合协议，新冒烟使用 `smoke_v2` 标签重新运行 12 条，不覆盖旧目录。

当前代码已增强失败记录：新失败文件同时保存异常类型、异常信息和最后一次无效模型输出，便于区分步骤数量、分值上限、JSON 或错因标签问题。该修改只位于论文协议投影脚本，不改变主评分。

## 5. 进入全量前的通过条件

1. 冒烟必须为 `12/12`，`projection_failures.jsonl` 不存在。
2. `protocol_manifest.json` 必须记录 `label_blind_projection=true`。
3. `forbidden_model_inputs` 必须包含 `manual_label`、`step.label`、`step.errors`。
4. 所有任务的输出条数、步骤数、分值范围和错误类别均通过合同校验。
5. 全量继续使用固定模型、端点、温度、思考模式和提示哈希，不得中途切换。
6. 正式报告同时给出 4,093 条公平公共子集；若与 4,109 条官方复现比较，必须说明样本口径不同。

## 6. 后续待办

1. 读取唯一失败的原始输出并分类。
2. 同标签补跑冒烟剩余 1 条；禁止删除整个输出目录重跑 12 条。
3. 运行 4,093 条协议投影，保留 checkpoint 以支持断点续传。
4. 用官方 `2_process_prediction.py` 及 QWK/CCS/ECS 脚本计算指标。
5. 生成三方同表：论文 DeepSeek-V3、官方代码当前模型复现、RefGrader 当前模型。
6. 对任务级差异做错误分析，并明确模型差异与方法差异不可完全分离。

## 7. 实验电脑执行命令

### 7.1 查看旧失败并运行修复策略冒烟

```powershell
cd "D:\Users\王鑫020420\Desktop\refgrader-main"

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:REFGRADER_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
$env:REFGRADER_DEEPSEEK_MODEL = "deepseek-chat"
$env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable(
    "DEEPSEEK_API_KEY",
    "User"
)

$Python = (Resolve-Path ".\venv\Scripts\python.exe").Path
$Source = "D:\Users\王鑫020420\Desktop\refgrader-public-datasets\sas_bench\official_89e572c\datasets"
$RunDir = "D:\Users\王鑫020420\Desktop\refgrader-main\results_runs\public_benchmarks\sas_bench_v1\runs\sasbench_refgrader_deepseek_official_t0_zeroshot_20260828_173432"
$Compare = Join-Path $RunDir "evaluation\compare.csv"
$ProtocolBase = "D:\Users\王鑫020420\Desktop\refgrader-public-datasets\sas_bench\protocol_comparison\refgrader_smoke12"
$Tag = "RefGrader_3WD_Deepseek_t0_smoke_v2"
$Output = "${ProtocolBase}_${Tag}_Scored"

& $Python scripts\benchmarks\project_sasbench_official_protocol.py `
    --source-dir $Source `
    --compare-csv $Compare `
    --protocol-base-dir $ProtocolBase `
    --save-type-name $Tag `
    --method 3wd `
    --text-provider deepseek `
    --thinking-mode disabled `
    --temperature 0 `
    --workers 1 `
    --limit-per-task 1 `
    --expected-records 12

if ($LASTEXITCODE -ne 0) {
    Get-Content "$Output\projection_failures.jsonl" -Encoding UTF8
    throw "冒烟仍有协议失败；先分析 raw_response，不要启动全量"
}

$CheckpointCount = @(
    Get-Content "$Output\projection_checkpoint.jsonl" -Encoding UTF8 |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
).Count

if ($CheckpointCount -ne 12) {
    throw "冒烟 checkpoint 不是12条：$CheckpointCount"
}
```

该命令首次执行必须显示 `Completed checkpoint records: 0` 和 `Pending projection records: 12`。若中断，保持 `smoke_v2` 标签再次执行即可续传。

### 7.2 冒烟通过后启动 4,093 条全量投影

```powershell
$FullBase = "D:\Users\王鑫020420\Desktop\refgrader-public-datasets\sas_bench\protocol_comparison\refgrader_full4093"
$FullTag = "RefGrader_3WD_Deepseek_t0_official_v1"

& $Python scripts\benchmarks\project_sasbench_official_protocol.py `
    --source-dir $Source `
    --compare-csv $Compare `
    --protocol-base-dir $FullBase `
    --save-type-name $FullTag `
    --method 3wd `
    --text-provider deepseek `
    --thinking-mode disabled `
    --temperature 0 `
    --workers 2 `
    --expected-records 4093

if ($LASTEXITCODE -ne 0) {
    throw "全量协议投影未完成；保留相同标签后断点续传"
}
```

中断后执行同一段全量命令即可续传；不要删除输出目录，不要更换 `$FullTag`、模型、温度或思考模式。
