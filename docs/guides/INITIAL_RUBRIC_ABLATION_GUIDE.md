# 第二台电脑：初始评分准则消融实验

## 实验定义与隔离

本实验只删除阶段1中的模型驱动准则优化/候选选择。使用项目已有 `rubrics/initial`；
该层已经包含项目的数据预处理和结构化，不等同于 `rubrics/source` 的原始评分文字。
模型、重复评分、证据提取、三支决策和残差校正的方法保持相同。
初始准则改变了评分分布，必须仅用该分支的验证集重新校准阶段4/5，不能沿用主实验参数。

初始准则、原始教师标签、元数据、划分和85条审核快照已冻结于：
`experiments/initial_rubric_ablation/reviewed85_20261019/`。
该版本对应主实验 `reviewed45_glm52_20261019_221414_317274`，有效
calibration/validation/test 数量为340/339/2675。本消融不执行准则优化，因此不使用
calibration样本去优化准则；保留其划分以确保不能流入validation或test。

独立入口：`scripts/run_initial_rubric_ablation.py`。它显式使用初始准则、独立事实缓存、
独立验证与测试目录、独立校准配置；不覆盖共享活动准则/配置，不自动提交或推送。
不创建伪造的“优化准则通过清单”。所有请求仍使用 Coding Plan 地址。

## 1. 当前电脑：先发布代码和冻结输入

下面只发布指定文件，不提交实时审核变更、运行中的优化准则产物或全部工作区。
提交和Git分支指针操作不会修改运行中的代码字节；不要在运行期间切换已有分支、拉取合并
或执行 `reset --hard`、`clean`、`stash`。

先检查当前分支和待提交文件：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
git branch --show-current
git status --short
```

下面命令提交明确列出的代码和冻结快照，在当前工作区内容不变的前提下建立发布分支。
若发布分支已存在，先检查它；不要强制覆盖。

```powershell
$PublishBranch = 'codex/initial-rubric-ablation-20261020'
$PublishPaths = @(
  'AGENTS.md',
  'scripts/run_csbench.py',
  'step0_extract_ground_truth.py',
  'scripts/run_reviewed_full_workflow.py',
  'scripts/verify_coding_plan.py',
  'scripts/run_initial_rubric_ablation.py',
  'test_reviewed_full_workflow.py',
  'test_csbench_grade_policy_paths.py',
  'test_coding_plan_verification.py',
  'test_initial_rubric_ablation.py',
  'REVIEWED_FULL_RUN_GUIDE.md',
  'CODING_PLAN_VERIFICATION.md',
  'INITIAL_RUBRIC_ABLATION_GUIDE.md',
  'experiments/initial_rubric_ablation/reviewed85_20261019'
)
git add -- $PublishPaths
if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
git diff --cached --stat -- $PublishPaths
git commit --only -m 'Add isolated initial-rubric ablation with frozen reviewed inputs' -- $PublishPaths
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect the message before continuing' }
git branch $PublishBranch HEAD
if ($LASTEXITCODE -ne 0) { throw 'Branch exists or could not be created; do not overwrite it' }
git push -u origin $PublishBranch
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not start the other-computer setup yet' }
git rev-parse HEAD
```

记下最后的提交号。此操作没有切换当前工作区，也不把运行中的准则产物或实时审核文件
混进提交。专用远程分支包含准备所需的新脚本，`origin/main` 未必包含它们。

## 2. 另一台Windows电脑：更新旧仓库与创建独立实验目录

旧仓库可先只获取远端信息，保留本地文件：

```powershell
# 将此路径改为另一台电脑上原有代码仓库的真实路径。
$ExistingRepo = 'C:\Users\你的用户名\Desktop\refgrader-main'
git -C $ExistingRepo fetch origin
git -C $ExistingRepo status --short
git -C $ExistingRepo log --oneline HEAD..origin/main
```

只有旧仓库无未提交文件、没有正在运行的实验、当前分支为main时，再执行：

```powershell
git -C $ExistingRepo pull --ff-only origin main
git -C $ExistingRepo lfs pull
```

有本地修改或本地独有提交时，不强行重置或自动暂存；直接使用下面的新目录。
`fetch`只更新远端引用；`pull --ff-only`只允许快进，不能解决未提交文件或分叉。

为消融新克隆一个目录，不在旧工作区切换准则：

```powershell
$Workspace = Join-Path $env:USERPROFILE 'RefGraderLabs'
$AblationRepo = Join-Path $Workspace 'refgrader-initial-ablation'
$PublishBranch = 'codex/initial-rubric-ablation-20261020'
New-Item -ItemType Directory -Force -Path $Workspace | Out-Null
git lfs install
git clone --branch $PublishBranch https://github.com/wangxin020420-byte/refgrader.git $AblationRepo
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; do not delete or overwrite an existing directory' }
Set-Location $AblationRepo
git lfs install --local
git lfs pull
if ($LASTEXITCODE -ne 0) { throw 'Git LFS download failed' }
git rev-parse HEAD
git status --short
```

提交号应与当前电脑发布的提交一致；首次新克隆应干净。LFS必须下载实际图片，不能仅有指针文件。
无需拉取或更改SAS-Bench作者仓库，也无需重新准备CSBench数据集。

## 3. 环境安装与离线验收

使用Windows x64 Python 3.11；不要跨机器复制 `venv` 或 `.venv-ocr`。

```powershell
Set-Location $AblationRepo
py -3.11 -m venv venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed' }
.\venv\Scripts\python.exe -m pip install -r requirements-main.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Main dependency installation failed' }
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_paddle_ocr.ps1 -PythonCommand .\venv\Scripts\python.exe
if ($LASTEXITCODE -ne 0) { throw 'OCR setup failed' }
.\venv\Scripts\python.exe scripts\verify_coding_plan.py
if ($LASTEXITCODE -ne 0) { throw 'Endpoint verification failed' }
.\venv\Scripts\python.exe scripts\run_initial_rubric_ablation.py --check
if ($LASTEXITCODE -ne 0) { throw 'Ablation preflight failed' }
```

离线检查应显示45题、`rubric_source: initial`、有效人数340/339/2675，真实模型调用0次。
检查初始准则解析、命令参数、冻结数据哈希、模型契约及图片内容。
安装OCR环境可能下载依赖和模型文件；离线预检本身不调用评分API。
若看到 `optimized rubric missing ... using initial rubric`，这是显式选择初始准则的
预期提示；最终应以 `offline_preflight: passed` 为准。

## 4. 后台执行、查询、日志和停止

固定本消融Run ID；后续断点续跑不要改名：

```powershell
$AblationRunId = 'initial85_glm52_20261020_pc2'
.\venv\Scripts\python.exe scripts\run_initial_rubric_ablation.py --background --run-id $AblationRunId
```

缺少环境Key时会隐藏输入；已有Key则继承。Key不写入文件、命令参数或Git。
执行顺序：初始准则验证集评分→仅用验证集校准三支决策与残差→固定配置测试集评分→评估。
未通过效果门禁时按主实验相同政策标记experimental并继续，不伪称正式通过。

```powershell
.\venv\Scripts\python.exe scripts\run_initial_rubric_ablation.py --status --run-id $AblationRunId
Get-Content ".\results_runs\initial_rubric_ablation\$AblationRunId\console.log" -Tail 40 -Wait
```

退出日志查看按Ctrl+C，不会停止实验。停止实验时输入本次启动输出的新PID：

```powershell
$AblationProcessId = [int](Read-Host '输入本次消融启动输出的PID')
taskkill /PID $AblationProcessId /T /F
```

中断后重复原启动命令即可；完成的阶段和答案保留。两台电脑共用账户/Key时共享接口限额，
可能出现限流；不要因此修改模型、温度、重复评分次数或覆盖检查点。

## 5. 结果对比与保存

结果目录：`results_runs/initial_rubric_ablation/<RunID>/`。

- `validation/`：初始准则验证结果。
- `a3wa_config.json`：本消融独立校准配置。
- `test/`：测试评分，checkpoint包括NEG样本的暂存分和复核状态。
- `reports/score_comparison.csv`、`reports/summary.json`：平均分、核心分、最终分等比较。
- `workflow.json`、`input_sha256.json`：状态与本机不可变输入记录。
- `cache/`：本消融独立缓存，不提交到代码仓库。

先比较两分支的 `model_avg_score`，更直接衡量准则优化对重复评分平均分的作用；
再比较 `three_way_core_score` 和 `final_calibrated_score`，分别反映保留后续方法时的整体变化。
两分支各自仅在validation重标定，同属w/o rubric optimization的完整流程消融。
按(question_id, answer_id)配对比较，核对同一policy ID和有效test人数2675；
不能与未审核版本2723人结果直接当成严格配对消融。

不把结果直接写回代码仓库或覆盖主实验run。可以在实验历史仓库的专用分支归档：

```powershell
$ArtifactsRepo = Join-Path $Workspace 'refgrader-artifacts-initial'
git clone https://github.com/wangxin020420-byte/refgrader-artifacts.git $ArtifactsRepo
if ($LASTEXITCODE -ne 0) { throw 'Artifacts clone failed; existing directory must be inspected first' }
$RunRoot = Join-Path $AblationRepo "results_runs\initial_rubric_ablation\$AblationRunId"
$RunState = Get-Content -LiteralPath (Join-Path $RunRoot 'workflow.json') -Raw | ConvertFrom-Json
if ($RunState.status -ne 'completed') { throw 'Experiment is not complete; do not archive as a completed run' }
$Archive = Join-Path $ArtifactsRepo "csbench\ablations\$AblationRunId"
if (Test-Path -LiteralPath $Archive) { throw 'Archive exists; do not overwrite it' }
New-Item -ItemType Directory -Path $Archive -Force | Out-Null
foreach ($Name in @('workflow.json','input_sha256.json','a3wa_config.json','exam_database.json','sample_policy.json','selection.json','review_decisions.jsonl','validation','test','reports','calibration_diagnostics')) {
  Copy-Item -LiteralPath (Join-Path $RunRoot $Name) -Destination $Archive -Recurse
}
Copy-Item -LiteralPath (Join-Path $AblationRepo 'experiments\initial_rubric_ablation\reviewed85_20261019') -Destination (Join-Path $Archive 'frozen_inputs') -Recurse
$ResultBranch = 'codex/initial-rubric-results-20261020'
git -C $ArtifactsRepo switch -c $ResultBranch
if ($LASTEXITCODE -ne 0) { throw 'Result branch creation failed; do not reuse it blindly' }
git -C $ArtifactsRepo add -- "csbench/ablations/$AblationRunId"
git -C $ArtifactsRepo commit -m "Archive initial-rubric ablation $AblationRunId"
if ($LASTEXITCODE -ne 0) { throw 'Artifact commit failed' }
git -C $ArtifactsRepo push -u origin $ResultBranch
```

主电脑后续在实验历史仓库 `fetch`该结果分支后，统一合并这一个独立run目录。
本方案不并发修改主实验的 `csbench/index.json`；历史索引由汇总端统一更新。
跨机器可同步完整证据，但不能把包含绝对路径的本机运行状态直接当成另一台的续跑状态。

## 6. 不能使用的操作

- 不运行主实验的 `--new-run` 作为本消融；那仍会执行准则优化。
- 不直接将initial文件覆盖到optimized目录，不伪造优化验收清单。
- 不删除主实验准则、策略、缓存和检查点。
- 不用 `--split all` 规避初始准则检查；应保持validation/test隔离。
- 不执行 `git add .`、`git reset --hard` 或清理命令来解决另一台的未提交文件。
- 不同时在两台机器写同一run目录或共享活动配置，不在运行期更新其源码。
