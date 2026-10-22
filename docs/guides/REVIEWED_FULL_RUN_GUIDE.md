# 审核数据版本：45题本地夜间一键重测

入口：`scripts/run_reviewed_full_workflow.py`。

本轮沿用当前方法，文本模型 GLM-5.2、thinking disabled，视觉模型 GLM-4.6v，Windows OCR 使用 CPU。它是审核数据版本下的现有方法重测；未实施研究方案中尚待验证的三支决策与BND门禁改进。

## 1. 晚上启动：整段粘贴到 PowerShell

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\run_reviewed_full_workflow.py --background
```

这是实际启动命令，会调用模型并消耗API额度。本次准备工作没有执行它。请在准备好当晚审核版本后运行。

若终端已有 `ZHIPUAI_API_KEY` 或 `ZHIPU_API_KEY`，直接沿用；否则启动时提示输入API key，输入隐藏，仅在父子进程内存中传递，不写入脚本、命令行或日志。不要将key发到聊天中。

脚本打印 PID、run ID、日志位置和状态文件。后台独立运行，可关闭启动终端；保持电脑供电、联网，不关机、不手动休眠。运行时请求Windows阻止自动睡眠，结束后恢复；屏幕可正常关闭。总用时受API响应、重试和样本复杂度影响，不保证次日早晨全部结束。

## 2. 查看进度

```powershell
.\venv\Scripts\python.exe scripts\run_reviewed_full_workflow.py --status
```

显示进程锁状态、当前阶段、逐题验证/测试进度、失败信息和最近25行日志。

## 3. 只检查环境和命令，不启动实验

```powershell
.\venv\Scripts\python.exe scripts\run_reviewed_full_workflow.py --check
```

检查45题、审核决定格式、修正分范围、排除后的样本数、划分隔离、初始准则、OCR环境，以及每个计划命令与现有参数解析器的兼容性。不调用API、不启用活动策略、不创建正式运行。

## 4. 一次启动实际执行什么

1. 冻结启动时已保存的审核决定，生成本轮不可变的样本策略与逐题计数；备份原活动准则、清单、A3WA配置和旧样本策略。
2. 记录数据、图片、评分代码和提示的SHA-256；进行原始快照及数据划分审计。
3. 执行少量GLM API连通性检查。
4. 对45题逐题执行准则优化。沿用当前方法，每题从自己的准则校准集合最多取5份作答；不是将整个校准池都当作模型训练数据。各题的方差进度放在本轮独立目录。
5. 统一激活本轮准则，复制准则证据到实验仓库。
6. 对全部有效验证答案重新评分，固定验证run ID；结果完整后复制验证证据。
7. 仅使用该验证run重新校准风险模型、A3WA阈值、BND门禁、误差区间与残差，保存配置和诊断；随后激活并复制校准结果。
8. 使用固定校准配置，对45题全部有效测试答案重新评分，固定测试run ID。
9. 统一使用本轮样本策略计算指标，导出结果并复制测试证据；保存整轮审核快照、策略及版本记录。

准则优化通过项目现有 `main_pipeline.py --mode VARIANCE_OPT --resume-optimization` 执行，使用本轮独立进度目录。这样既保留原有候选选择逻辑，又避免旧活动准则使新的整轮重测被误判为“已完成”。其余阶段调用现有 `run_csbench.py`，不重写评分算法。

## 5. 审核版本如何使用

- `corrected`：保留答案，读取人工修订分。
- `confirmed_noise`：按已保存决定排除，不重新自动判断是否应删除。
- `retained_hard_case`：保留原教师分。
- `ambiguous`：沿用现有项目规则，仍保留原分；单独记录其未裁定状态。
- 未审核答案：保留原教师分。因此必须称为“应用当前审核决定的数据版本”，不能称为“全部3411份已完成审核”。

程序不覆盖原始 `teacher_scores.json`，不改图片，不改正在编辑的审核JSONL。实际运行开始时，将本轮策略原子写入活动策略路径；所有子进程仍显式绑定独立的冻结策略，防止后续审核改变本轮标签口径。之后可以继续保存人工审核决定，但这些新增决定不进入本轮。

2026-10-19本地准备时的读取快照为85条决定：15条修分、57条排除、13条保留。对应准则校准池340份、验证339份、测试2675份。**这不是夜间固定人数**：实际启动时重新冻结与统计，终端和 `selection.json` 才是本轮依据。

## 6. 中断后的续跑

重新执行同一条启动命令：

```powershell
.\venv\Scripts\python.exe scripts\run_reviewed_full_workflow.py --background
```

默认恢复最近一轮，沿用同一个run ID和冻结标签；已完成阶段不重新执行，失败阶段自动退避重试，评分只补未完成答案。重启电脑、网络中断或达到重试上限后都使用这条入口，不加 `--force`，不删除checkpoint。

指定历史运行：

```powershell
.\venv\Scripts\python.exe scripts\run_reviewed_full_workflow.py --background --run-id <启动时打印的run_ID>
```

只有明确要进行另一轮实验、或纳入后来新增的审核决定时，才创建新版本：

```powershell
.\venv\Scripts\python.exe scripts\run_reviewed_full_workflow.py --background --new-run
```

运行期间不要修改题库、初始准则、图片、评分代码和提示，也不要并行启动另一套会更换活动准则/校准配置的实验。输入或已完成产物变化会停止，不能通过改签名绕过。单实例锁会阻止重复启动本入口。

## 7. 门禁与异常

- 限流、连接超时和未完成评分：固定run ID重试，等待从60秒增长至最多600秒；评分阶段每次启动最多尝试36次。
- 校准的部署指标未通过：保留全部诊断，标记 `experimental`，继续本轮开发性测试；不会因此半夜停在校准后。不能将这种结果称为通过安全验收。
- 残差校正因数据门禁未启用：保留核心分数并记录，不强行生成校正项。
- 数据范围、标签版本、模型契约、代码或产物哈希不一致：停止，保护实验有效性。
- 本轮全部有效样本完成且评估/证据复制成功后，状态才为 `completed`。

## 8. 输出位置与第二天查看

- 总状态：`results_runs/reviewed_csbench45_glm52/<run_ID>/workflow.json`
- 数据范围：同目录 `selection.json`
- 审核快照/策略：`review_decisions.jsonl`、`sample_policy.json`
- 输入哈希：`input_sha256.json`
- 旧活动产物备份：`previous_products/`
- 每题准则优化进度：`rubric_work/<题号>/`
- 固定校准配置：`a3wa_config.json`，以及同目录的诊断结果目录
- 验证/测试目录：总状态中的 `paths.validation` / `paths.test`
- 后台总日志：`logs/<run_ID>.log`
- 实验仓库：`../refgrader-artifacts/csbench/<题号>/...`；整轮版本说明在 `csbench/workflows/<run_ID>/`。

第二天先运行 `--status`，查看是否 `completed`、是否存在未完成答案，以及 `result_class` 和部署门禁结果。完成后再检查代码仓库的活动配置及实验仓库的产物，并按批次提交。夜间脚本不会自动Git提交或推送，也不会自动运行SAS-Bench等其他模型/基线。
