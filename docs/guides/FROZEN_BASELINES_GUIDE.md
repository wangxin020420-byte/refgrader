# 两轮基线冻结与多机同步

## 为什么PC2源代码管理没有显示结果

1. 代码仓库忽略`results_runs/`、`logs/`和缓存。这些文件仍在磁盘，不属于未保存的编辑器文档。
2. `run_initial_rubric_ablation.py`只执行validation、calibrate、test、evaluate，没有自动发布到历史仓库。
3. PC2结果在`refgrader-initial-ablation`独立checkout；只打开`refgrader-main`不会显示其文件视图。
4. `refgrader-artifacts`只有在显式复制完成结果后才会出现Git更改。

不要删除忽略规则或强行把`results_runs`加入代码Git。冻结入口不调用API、不重新评分、不修改活跃审核政策，也不要求主实验环境依赖。

## 结果布局

```text
refgrader-artifacts/csbench/frozen_baselines/<run_id>/
├─ baseline_manifest.json        完成标记、源文件指纹、归档文件指纹、真实门禁状态
├─ .gitattributes                保留原始字节，禁止Git换行转换
├─ workflow/                     原始workflow/config/policy/selection/input SHA
├─ inputs/                       原始标签、元数据、题目、划分、initial rubric
│  └─ rubrics/optimized/         只在PC1存在；按已完成准则证明核对
├─ validation/                   45个完成检查点
├─ test/                         45个完成检查点
├─ evaluation/compare.csv        统一文件名；内容按原字节复制
├─ evaluation/summary.json       原始评估汇总
├─ diagnostics/                 已有校准诊断
├─ metrics.json                 标准库补算宏/加权/合并原始分QWK等
└─ README.md
```

该目录是可校验快照，不是新实验；原来`results_runs`和旧逐题历史目录不删除。
图片不重复复制，保留原始图片指纹；跨机如需图像，使用版本化数据/LFS。
原工作流的绝对路径保留，不能把它当作另一台设备的续跑状态。
不要在冻结目录内重算指标或加临时文件，另写分析目录。

## PC1：已经完成的标准冻结

Run ID：`reviewed45_glm52_20261019_221414_317274`。
本机2026-10-22建立标准快照并校验。部署门禁仍为false，保留experimental，不能因归档改成formal。

重新验证：

```powershell
Set-Location 'C:\Users\wx\Desktop\refgrader-main'
.\venv\Scripts\python.exe scripts\freeze_experiment_baseline.py verify --snapshot ..\refgrader-artifacts\csbench\frozen_baselines\reviewed45_glm52_20261019_221414_317274
```

## PC2：先获取工具，不改消融输入或既有环境

推荐在原`refgrader-main`中获取归档工具，而不更新已完成消融checkout中的核心代码。
如旧仓库有未提交更改，不清空、不reset，也不强行pull。
可直接把提供的工具包放在PC2原`refgrader-main`根目录，解压到本地维护目录；工具包只有归档脚本和说明。

```powershell
# 在PC2现有refgrader-main根目录运行；不是本机的C:\Users\wx路径。
$CodeRepo = (Get-Location).Path
$Tools = Join-Path $CodeRepo '.local-maintenance\freeze-tools-20261022'
$Package = Join-Path $CodeRepo 'refgrader-freeze-tools_20261022.zip'
if (-not (Test-Path -LiteralPath $Package -PathType Leaf)) {
    throw '请先将工具包放到当前refgrader-main根目录。'
}
if (-not (Test-Path -LiteralPath $Tools)) {
    Expand-Archive -LiteralPath $Package -DestinationPath $Tools
}
$ExperimentRepo = Join-Path (Split-Path -Parent $CodeRepo) 'refgrader-initial-ablation'
$ArtifactsRepo = Join-Path (Split-Path -Parent $CodeRepo) 'refgrader-artifacts'

powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Tools 'freeze_pc2_baseline.ps1') -ExperimentRepo $ExperimentRepo -ArtifactsRepo $ArtifactsRepo -DryRun
if ($LASTEXITCODE -ne 0) { throw '预检未通过，不要重跑实验。请提供归档报错。' }

powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Tools 'freeze_pc2_baseline.ps1') -ExperimentRepo $ExperimentRepo -ArtifactsRepo $ArtifactsRepo -Stage
if ($LASTEXITCODE -ne 0) { throw '归档或暂存未通过。原实验文件保持原状。' }
```

如果工具已随代码拉取到` scripts/`，可直接调用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\freeze_pc2_baseline.ps1 -Stage
```

脚本优先使用`py -3.11`，再寻找已有Python>=3.8；只用标准库，不要求`openai`、OCR、numpy或sklearn。
`-Stage`仅暂存本轮快照；不自动commit/push，不夹带主仓库的2条更改或教师审核记录。

预检必须确认：工作流completed、数据与政策指纹一致、验证/测试ID完整、检查点符合完成证明、教师分与冻结政策一致、CSV分数与检查点一致、逐题QWK/全局MAE/RMSE可复算。
若失败，先核查具体差异；不能直接把状态改为completed、改SHA或重新批改来绕过。

## PC2：查看更改、提交和推送独立结果分支

结果仓库是公开仓库；push会公开本次指定快照中的作答证据和实验记录。先审阅暂存范围。
以下只操作历史仓库，不提交代码仓库的本地改动：

```powershell
$RunId = 'initial85_glm52_20261020_pc2'
$Relative = "csbench/frozen_baselines/$RunId"
$ResultBranch = 'codex/freeze-pc2-initial85-20261022'
git -C $ArtifactsRepo status --short
git -C $ArtifactsRepo diff --cached --stat -- $Relative

# 第一次使用该分支时创建；如提示已存在，检查后使用已有分支，不重复创建。
git -C $ArtifactsRepo switch -c $ResultBranch
if ($LASTEXITCODE -ne 0) { throw '分支未切换成功；先检查，不要reset。' }
git -C $ArtifactsRepo commit -m 'Freeze PC2 reviewed85 initial-rubric baseline' -- $Relative
if ($LASTEXITCODE -ne 0) { throw '提交失败或已经提交；先查看git log/status。' }
git -C $ArtifactsRepo push -u origin $ResultBranch
if ($LASTEXITCODE -ne 0) { throw '推送失败；本地快照和提交仍保留，只需重试同一分支推送。' }
```

如果VS Code仍不显示结果仓库，先用上述`git status`判断磁盘Git状态，再把实际`refgrader-artifacts`目录添加到工作区。

## PC1：导入PC2后配对验证

在PC1历史仓库保存本机未提交内容后，获取并检查PC2分支。不要在脏工作区强行merge。

```powershell
git -C ..\refgrader-artifacts fetch origin codex/freeze-pc2-initial85-20261022
git -C ..\refgrader-artifacts log --oneline HEAD..origin/codex/freeze-pc2-initial85-20261022
git -C ..\refgrader-artifacts merge --no-edit origin/codex/freeze-pc2-initial85-20261022
if ($LASTEXITCODE -ne 0) { throw '有冲突时逐项核对，不覆盖两份结果。' }

.\venv\Scripts\python.exe scripts\freeze_experiment_baseline.py verify --snapshot ..\refgrader-artifacts\csbench\frozen_baselines\initial85_glm52_20261020_pc2
if ($LASTEXITCODE -ne 0) { throw 'PC2快照SHA检查失败。' }
.\venv\Scripts\python.exe scripts\freeze_experiment_baseline.py compare --optimized-snapshot ..\refgrader-artifacts\csbench\frozen_baselines\reviewed45_glm52_20261019_221414_317274 --initial-snapshot ..\refgrader-artifacts\csbench\frozen_baselines\initial85_glm52_20261020_pc2
```

配对检查验证ID、有效教师分、划分、审核政策、initial准则及模型合同；实际拟合阈值和残差参数允许不同。
它不证明两台设备的OCR缓存、依赖安装、随机调用及服务端模型修订完全一致，仍需记录这些限制。
验证完成后才更新PC2登记状态，进入[下一阶段清单](../../experiments/ROADMAP.md)。

## 不覆盖与失败处理

相同设备重复freeze时，先验证源文件与已有快照；完全一致则返回`already_frozen_verified`。
内容不同或目标不完整时拒绝覆盖。复制失败保留现场，不自动递归删除。
另一台电脑只需`verify`导入的快照；没有原始运行目录时不能用`freeze`代替验证。
