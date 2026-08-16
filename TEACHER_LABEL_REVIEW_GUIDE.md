# 教师标签审核界面使用指南

本界面用于人工核对“教师分数与模型分数差异较大”的候选样本。它会同时展示学生作答图片、题目与官方参考答案、当前生效评分准则、教师分、模型均分、3WD-Core、最终校准分、三类风险和初筛提示，并把人工决定保存为可追踪、可撤销的 JSONL 文件。

当候选报告的 `summary.json` 记录了 `source_run_id` 时，界面还会按需读取该运行归档的 grading checkpoint，展示模型三次独立判决、逐评分项得分、学生证据、满分期望、给分理由、三支路由和 BND 仲裁说明。这里展示的是实验时保存的原始轨迹，不会重新调用模型生成解释。

## 1. 先理解两个原则

1. **候选不等于噪声。** 模型与教师分数差异较大，也可能是模型评分、OCR 提取、参考答案或评分准则存在问题。只有人工确认后，才能标记为噪声。
2. **原始数据永远不移动。** 点击“确认噪声”不会删除或移动学生图片，也不会改写 `teacher_scores.json`。界面只记录一条人工决定；只有显式生成并启用样本策略后，该样本才会在后续实验中被排除。

## 2. 数据保存在哪里

候选报告来自：

```text
data/csbench/quality_control/reports/<报告批次>/
```

人工决定默认保存到：

```text
data/csbench/quality_control/reviews/<报告批次>_decisions.jsonl
```

启用后的统一样本策略保存到：

```text
data/csbench/quality_control/policies/active_sample_policy.json
```

这些文件位于 `refgrader-main`，可以通过 Git 在多台设备之间同步。学生图片仍保存在：

```text
data/csbench/student_images/<题号>/<答案ID>.<扩展名>
```

## 3. 启动界面

在 PowerShell 中进入 `refgrader-main`：

```powershell
cd "C:\Users\wx\Desktop\refgrader-main"
```

审核全 43 道题的可疑教师标签时，直接启动即可。程序会优先选择最新的
`teacher_label_audit_all43_*` 全 43 题报告，并合并模型均分和 3WD-Core 两套候选，
按 `question_id + answer_id` 去重：

```powershell
.\venv\Scripts\python.exe scripts\review_teacher_labels.py
```

当前全 43 题报告共有 1026 条宽召回统计候选，其中 P1 为 288 条、P2 为
738 条。由于已存在同名 `_initial_screening.csv`，无参数启动时会默认显示
73 条二次初筛候选：教师标签高可疑 24 条、模型或提取问题 20 条、评分规则歧义
29 条。这些候选都不代表已经确认存在问题；任何样本都不会在未经人工决定的情况下自动排除。

需要查看全部 1026 条统计候选时，增加 `--all-candidates`：

```powershell
.\venv\Scripts\python.exe scripts\review_teacher_labels.py --all-candidates
```

也可以显式指定当前 43 题审计报告：

```powershell
.\venv\Scripts\python.exe scripts\review_teacher_labels.py `
  --report-run teacher_label_audit_all43_20260727_160210
```

程序会自动打开浏览器，默认地址为：

```text
http://127.0.0.1:8765
```

即使当前终端显示 `(.venv-ocr)`，上述命令也会显式使用主环境 `venv`，不受当前激活环境影响。

### 3.1 仅诊断三支路由时启动 33 条 unsafe POS 专项批次

当前审核批次为：

```text
evidence_verifier_v2_train31_20260812_212808_unsafe_pos_review
```

审核服务停止后，重新打开 PowerShell，完整执行以下命令即可恢复审核：

```powershell
cd "C:\Users\wx\Desktop\refgrader-main"

.\venv\Scripts\python.exe scripts\review_teacher_labels.py `
  --report-run evidence_verifier_v2_train31_20260812_212808_unsafe_pos_review
```

启动成功后，终端应显示 `Candidates: 33`，浏览器会自动打开
`http://127.0.0.1:8765`。审核期间需要保持该 PowerShell 窗口运行；关闭窗口或按
`Ctrl+C` 会停止审核服务，但已经保存的决定不会丢失。再次执行同一命令即可继续。

如果页面显示“候选报告加载失败”或 `fetch` 失败，先确认启动命令所在终端仍在运行，
然后在浏览器按 `Ctrl+F5`；服务已经停止时，不要只刷新旧页面，应重新执行上述命令。

如果存在同名的 `_initial_screening.csv`，界面默认只加载已经完成初筛的候选，避免把全部自动候选一次性推给人工。需要检查全部自动候选时，显式增加：

```powershell
.\venv\Scripts\python.exe scripts\review_teacher_labels.py `
  --report-run teacher_label_audit_all43_20260727_160210 `
  --all-candidates
```

如果浏览器没有自动打开，手动访问终端打印的地址。如果端口被占用，可以更换端口：

```powershell
.\venv\Scripts\python.exe scripts\review_teacher_labels.py `
  --report-run teacher_label_audit_all43_20260727_160210 `
  --port 8766
```

不指定 `--report-run` 时，程序优先选择最新的全 43 题教师标签审计报告；只有不存在全量报告时，才回退到最近修改且包含候选 CSV 的报告。专项诊断仍应显式填写批次名。

## 4. 界面区域

### 4.1 顶部筛选栏

- **题目**：只查看某一道题。
- **优先级**：`P1` 通常表示差异大且当前混杂风险较低，应优先人工复核；`P2` 表示还可能存在提取、评分稳定性或准则映射问题。
- **审核状态**：切换待审核、已审核或全部候选。
- **初筛类别**：使用已有初筛结果缩小范围。
- **复核人**：填写姓名或设备标识，保存后写入决定记录。
- **生成并启用策略**：把已经保存的决定编译为实验流程能够统一读取的活动策略。

### 4.2 左侧学生图片

左侧显示原始学生作答图片。右上角提供缩小、恢复、放大和旋转按钮。图片只读，界面不会修改原文件。

### 4.3 中间审核依据

中间区域分为三个页签，切换样本后会自动加载该题对应的内容。

#### 分数与证据

- 教师分数及教师分与候选参考分的差值；
- 模型三次评分均分；
- 仅含三支决策机制的 `3WD-Core` 分数；
- 包含可选残差校正的最终校准分；
- POS/BND/NEG 路由；
- 证据质量风险 `U_E`、评分稳定性风险 `U_S`、评分准则映射风险 `U_R`；
- OCR/文本提取结果；
- 自动初筛结论与建议。

初筛提示只用于排序和辅助判断，不能自动替代人工决定。

教师分旁的差值统一定义为：

```text
差值 = 教师分 - 候选报告采用的参考模型分
```

正值表示教师分更高，负值表示模型参考分更高。候选报告可能采用模型均分、基础选定分或 3WD-Core 作为参考，具体以“模型给分依据”中的分数链路为准。

#### 模型给分依据

- 三次独立语义判决及其总分；
- 每个评分项实际采用的学生证据、满分期望、得分和理由；
- 三次判决均分、基础选定分、3WD-Core 和最终校准分；
- POS/BND/NEG 路由原因、BND 动作和仲裁说明；
- 独立证据验证器保存的支持度、置信度和矛盾标记。

如果界面提示 checkpoint 不可用，应先同步 `refgrader-artifacts`，或在启动时通过 `--grading-source-dir` 显式指定包含 `<题号>_grading_checkpoint.json` 的目录。缺少原始轨迹时不要仅凭分差确认教师噪声。

#### 题目与参考答案

- 当前题目的题干；
- 题目图片；
- 数据集中保存的官方参考答案；
- 数据集中保存的原始官方评分说明。

参考答案只能证明学生结论是否正确，不能替代评分准则。对于过程题，仍需结合学生推导过程和当前评分准则判断应得分数。

#### 当前评分准则

该页签展示 `data/csbench/rubrics/active_rubric_set.json` 当前指定的优化评分准则，包括：

- 每个评分项的编号、名称和分值；
- 满分证据或标准答案；
- 核心得分项、支撑项等评分层级；
- 题目语义、拆分策略和依赖关系；
- 当前准则来源及文件路径。

界面优先读取当前激活准则；如果激活配置缺失，才会显示题库记录中的准则路径，并明确提示加载状态。历史归档准则不会被自动当作当前评分依据。

### 4.4 右侧人工决定

四种决定与项目现有质量策略完全一致：

| 按钮 | 内部决定 | 后续作用 |
| --- | --- | --- |
| 保留原标签 | `retained_hard_case` | 教师标签有效；样本继续参与实验 |
| 修正教师分 | `corrected` | 样本保留，后续读取时使用人工复核分覆盖原教师分 |
| 确认噪声 | `confirmed_noise` | 启用策略后，从优化、validation、校准、正式评估和断点续传中排除 |
| 暂缓复核 | `ambiguous` | 证据不足；样本暂时保留，等待以后继续复核 |

选择“修正教师分”时必须填写新分数，系统会按照该题满分检查范围。选择“确认噪声”时会再次弹出确认框，防止误操作。

建议同时填写原因和复核备注。论文复现、后续争议处理和多设备协作都需要这些记录。

## 5. 推荐审核流程

1. 先按 `P1` 和“仅待审核”开始核对。
2. 对照原图检查模型提取文本是否可靠。
3. 检查参考答案和评分准则是否可能造成模型误判。
4. 只有明确确认教师标签不可用时，选择“确认噪声”。
5. 教师分有误但可以给出可靠新分数时，优先选择“修正教师分”，不要删除样本。
6. 无法立即判断时选择“暂缓复核”。
7. 点击“保存决定并进入下一份”。每次保存都会立即写入决定文件，关闭界面不会丢失已保存进度。
8. 完成一批审核并检查决定后，再点击“生成并启用策略”。

## 6. 关闭与继续

在启动界面的终端按 `Ctrl+C` 即可停止服务。已经保存的决定仍在 JSONL 文件中。

下次使用同一个 `--report-run` 启动时，界面会读取原决定文件并恢复审核进度，不会创建第二套决定。

### 6.1 撤销单条决定

顶部“审核状态”选择“仅已审核”，定位样本后点击右下角“撤销”。界面会从决定 JSONL 中删除该样本的决定，并把它重新放回待审核队列；原始图片、教师分和候选报告均不修改。

### 6.2 整批重新审核

如果分差定义理解错误，不应直接删除旧记录。先关闭审核界面，再把决定文件改名归档：

```powershell
$Run = "evidence_verifier_v2_train31_20260812_212808_unsafe_pos_review"
$Review = "data\csbench\quality_control\reviews\${Run}_decisions.jsonl"
$Backup = "data\csbench\quality_control\reviews\${Run}_decisions_before_rereview_$(Get-Date -Format 'yyyyMMdd_HHmmss').jsonl"

if (Test-Path $Review) {
    Move-Item -LiteralPath $Review -Destination $Backup
}
```

重新使用同一个 `--report-run` 启动后，审核计数会恢复为 `0 / 总数`。旧记录保留在带时间戳的备份中，便于追溯；新审核仍写回原来的 `_decisions.jsonl`。

## 7. 启用策略后的影响

活动策略生效后：

- `confirmed_noise` 在评分准则优化、validation、A3WA/残差校准、正式 test 评估和断点续传中统一排除；
- `corrected` 使用人工复核分，但原始 `teacher_scores.json` 不变；
- `retained_hard_case` 和 `ambiguous` 继续保留；
- 策略路径与哈希会写入实验配置和运行签名；
- 策略发生变化后，不允许把旧 checkpoint 当成相同配置继续运行。

需要复现实验的原始标签版本时，可临时使用：

```powershell
$env:REFGRADER_SAMPLE_POLICY_MODE = "raw"
```

完成原始标签实验后，移除该临时设置：

```powershell
Remove-Item Env:REFGRADER_SAMPLE_POLICY_MODE
```

## 8. 多设备同步

审核或启用策略后，先检查：

```powershell
git status --short
```

主要应出现：

```text
data/csbench/quality_control/reviews/<报告批次>_decisions.jsonl
data/csbench/quality_control/policies/active_sample_policy.json
```

确认无误后提交并推送 `refgrader-main`。另一台设备拉取后会获得相同决定和活动策略。不要在两台设备上同时编辑同一个决定文件；如果确实需要多人并行审核，应按不同报告或题目分别保存决定，再统一合并。

## 9. 常见问题

### 页面打不开

检查启动终端是否仍在运行，并访问终端打印的地址。端口冲突时使用 `--port 8766`。

### 看不到学生图片

确认已经执行 `git lfs pull`，并检查对应文件是否存在于 `data/csbench/student_images`。服务只允许读取该目录中的图片，不接受任意外部路径。

### 保存修正分失败

修正分必须是有限数字，并位于 `0` 到该题满分之间。

### 保存后为什么实验还没有排除噪声

保存决定与启用策略是两个独立步骤。必须点击顶部“生成并启用策略”，生成 `active_sample_policy.json` 后才会影响后续实验。

### 关闭浏览器会不会丢失结果

已经点击保存的决定不会丢失；尚未保存的表单内容会丢失。
