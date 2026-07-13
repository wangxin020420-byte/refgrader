# RefGrader 正式实验命令

CSBench 实验统一使用 `scripts/run_csbench.py`。更换题目时只需要修改题号，例如把 `CO_2` 改成 `CO_3`。

## 1. 可用题目 ID

运行统一命令时，只需把示例中的 `CO_3` 替换为下表中的题目 ID。题目 ID 不区分输入大小写，脚本会自动转换为大写。

| 题目 ID | 数据集科目 | 主要考察内容 |
| --- | --- | --- |
| `CO_1` | 计算机组成原理（CO） | 机器指令格式、直接寻址与间接寻址、操作数访问 |
| `CO_2` | 计算机组成原理（CO） | 中断响应优先级、中断处理优先级、屏蔽字与中断执行顺序图 |
| `CO_3` | 计算机组成原理（CO） | 8位补码运算、移位/除法、十六进制补码结果与溢出判断 |
| `CO_4` | 计算机组成原理（CO） | 组相联 Cache 映像、地址字段划分与命中判断 |
| `CO_5` | 计算机组成原理（CO） | 定长指令、扩展操作码编码、零/一/二地址指令数量 |
| `CO_6` | 计算机组成原理（CO） | 水平微指令、控制字段编码、条件转移、控制存储器容量 |
| `CO_7` | 计算机组成原理（CO） | 指令数、CPI、主频与程序运行时间 |
| `CO_8` | 计算机组成原理（CO） | 程序查询 I/O、DMA 与 CPU 时间占用率 |
| `CO_9` | 计算机组成原理（CO） | ROM/RAM 存储器扩展、芯片数量、地址译码与片选逻辑 |
| `CO_10` | 计算机组成原理（CO） | 4路组相联 Cache、标记区容量与相等比较器数量 |
| `CO_11` | 计算机组成原理（CO） | R/I/J 型指令格式、操作码扩展与最大指令数量 |
| `CO_12` | 计算机组成原理（CO） | CPU 数据通路、寄存器识别、取指/访存过程与微程序控制器设计 |
| `CO_13` | 计算机组成原理（CO） | CPU 中断响应、屏蔽字设计与中断处理轨迹 |
| `CPL_1` | C语言/程序设计（CPL） | 自增、自减运算符与表达式求值顺序 |
| `CPL_2` | C语言/程序设计（CPL） | for 循环、条件判断、continue 与累加结果 |
| `CPL_3` | C语言/程序设计（CPL） | 二维数组初始化、行和与列和计算 |
| `CPL_4` | C语言/程序设计（CPL） | 指针移动、数组首尾元素交换与循环条件 |
| `CPL_5` | C语言/程序设计（CPL） | 多项式计算程序改错、函数、数组、文件输入与命令行参数 |
| `CPL_6` | C语言/程序设计（CPL） | 变位词判断、字符处理、计数数组与程序填空 |
| `CPL_7` | C语言/程序设计（CPL） | 5×5矩阵转置、数组输入与文本文件输出 |
| `CPL_8` | C语言/程序设计（CPL） | 动态二维数组、指针、最高/最低分删除与平均分计算 |
| `DM_1` | 离散数学（DM） | 偏序集、整除关系、哈斯图、极值元、上下确界 |
| `DM_2` | 离散数学（DM） | 二元关系与等价关系的自反、对称、传递性证明 |
| `DM_3` | 离散数学（DM） | 谓词逻辑翻译、自然推理系统与严格推导 |
| `ISC_1` | 计算机导论（ISC） | 集成电路、摩尔定律挑战与延续计算性能的新技术 |
| `ISC_2` | 计算机导论（ISC） | 操作系统功能、进程管理与多任务调度 |
| `ISC_3` | 计算机导论（ISC） | 数据库系统组成、现代信息系统与 SQL 作用 |
| `ISC_4` | 计算机导论（ISC） | 网络协议概念与 HTTP 协议构成 |
| `ISC_5` | 计算机导论（ISC） | ASCII 十六进制解码与字符类型识别 |
| `ISC_6` | 计算机导论（ISC） | 十进制整数/小数转换为二进制 |
| `ISC_7` | 计算机导论（ISC） | 地址总线、数据总线、最大寻址空间与字长 |
| `ISC_8` | 计算机导论（ISC） | 原码、反码、补码表示与补码减法 |
| `ISC_9` | 计算机导论（ISC） | 图像分辨率、灰度位深与存储容量计算 |
| `ML_1` | 数字逻辑（ML） | 真值表填写 |
| `ML_2` | 数字逻辑（ML） | 组合逻辑电路与最简与或表达式 |
| `ML_3` | 数字逻辑（ML） | 同步时序电路、激励函数与输出函数 |
| `ML_4` | 数字逻辑（ML） | Mealy 序列检测器、状态图与状态表 |
| `POC_1` | 编译原理（POC） | 根据上下文无关文法描述其生成的语言 |
| `POC_2` | 编译原理（POC） | 语法分析中的短语、直接短语与句柄 |
| `POC_3` | 编译原理（POC） | 给定文法和符号串的短语、直接短语与句柄 |
| `POC_4` | 编译原理（POC） | 构造非0开头正偶数的2型文法 |
| `POC_5` | 编译原理（POC） | LR(1) 项目集规范族构造 |
| `POC_6` | 编译原理（POC） | 属性文法类型判断与标注语法树 |

说明：

- 当前兼容题库共包含 43 道题。
- `OS_1`、`OS_2` 目前没有完整 question 定义，因此未进入兼容题库，不能传给统一入口。
- 本项目将 `ML` 题组按实际题目内容统一称为“数字逻辑”。

## 2. 进入服务器环境

```bash
conda activate ref-grader
```

```bash
cd /home/E125221219/projects/refgrader
```

## 2.1 正式 test 完成后的自动评估与 artifacts 复制

`grade --split test` 在完整批改成功后会自动执行以下操作，不需要再单独调用
`evaluate` 或 `publish`：

```text
校验 checkpoint 与 test split 完全一致
-> 评估 single / avg / selected / 3WD
-> 导出 results_runs/csbench_<题目集合>_full/evaluation/compare.csv
   和 evaluation/summary.json
-> 复制完整实验产物到同级 refgrader-artifacts
```

默认只复制到本地 artifacts 仓库，不执行 Git commit 或 push。因此 VS Code 的
`refgrader-artifacts` 源代码管理会显示待检查的新增文件，由用户手动提交和推送。
只有显式添加 `--push-artifacts` 才会自动提交和推送。

自动复制内容与既有 artifacts 目录结构一致：

```text
csbench/<题号>/grading_runs/<run_id>/
  calibration/a3wa_config.json       # 使用 --a3wa-config 时保存
  rubrics/initial_rubric.json
  rubrics/optimized_rubric.json
  rubrics/optimization_manifest.json
  rubric_optimization/variance_checkpoint.json
  rubric_optimization/progress.json
  grading/grading_checkpoint.json
  grading/graded_results.json
  grading/rejected.json               # 存在 NEG 时保存
  grading/progress.json
  evaluation/compare.csv
  evaluation/summary.json            # 各指标的机器可读 JSON
  dataset/question_split.json         # 本次使用的固定 split
  logs/experiment.log
  run_manifest.json
```

断点续跑时，如果某道题在正式命令启动前已经拥有完整 test checkpoint，该题会被跳过，
也不会被错误标记为使用本次新传入的 A3WA config。`run_manifest.json` 会将其记录为
`preexisting_completed_checkpoint`；新 config 只复制到本次实际新批改的题目目录。

validation/calibration 使用独立的 `validation_runs`/`calibration_runs`，不会冒充正式
test artifacts。`--limit` 调试运行、checkpoint 不完整、存在未解决 failed、split 污染
或重复 ID 时不会发布。`--include-facts` 和 `--include-raw-ocr` 仍为可选项，默认不复制
大体积逐答案缓存。

示例：正式批改后自动评估并复制，但由用户手动 Git 推送：

```bash
python scripts/run_csbench.py grade CO_2 --split test --a3wa-config results_runs/a3wa_config.json
```

仅在明确不需要自动评估和复制时才使用：

```bash
python scripts/run_csbench.py grade CO_2 --split test --a3wa-config results_runs/a3wa_config.json --no-artifacts
```

## 3. CO_1 到 CO_7 四种常用执行方式

下面四种命令都以 CO_1 到 CO_7 为例。每种流程都同时给出后台模式和前台模式。

说明：

- `--force` 表示覆盖已有优化准则或旧 checkpoint。
- 后台模式适合服务器长时间正式实验；命令启动后终端会很快返回，可以关闭 VS Code 窗口和本地电脑。
- 前台模式适合调试；终端会持续显示运行过程，关闭窗口会中断任务。

### 3.1 重新生成兼容视图 → 重新优化评分准则 → 正式批改

适用场景：明确决定把外部 `CSBench_new` 的新版本重新导入内嵌快照。普通实验不要使用该命令；当前 `data/csbench` 已随 `refgrader-main` 版本化，不再依赖外部数据集。导入时会在 prepare 后自动补齐参考图并改写为可移植路径。

后台模式：

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --dataset-root /home/E125221219/CSBench_new --force --background
```

作用：一条命令完成完整实验。执行顺序是：重新生成 `data/csbench` 兼容视图 → 覆盖重跑评分准则优化 → 正式批改。由于使用了 `--background`，整条链路都在服务器后台运行。

前台模式：

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --dataset-root /home/E125221219/CSBench_new --force
```

作用：执行同样的完整流程，但全部在当前终端前台运行。适合观察详细输出和调试，不适合关机前长时间运行。

### 3.2 重新优化评分准则 → 正式批改

适用场景：`data/csbench` 已经是最新的，不需要重新生成兼容视图，但需要覆盖旧评分准则并重新批改。

后台模式：

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force --background
```

作用：跳过兼容视图生成，直接基于当前 `data/csbench` 覆盖重跑评分准则优化，然后正式批改。由于使用了 `--background`，优化和批改两个阶段整体都在服务器后台运行。

前台模式：

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force
```

作用：执行同样的“优化准则 → 正式批改”流程，但全部在当前终端前台运行，适合确认流程是否正常。

### 3.3 只重新优化评分准则

适用场景：只想重新生成 optimized rubric，先不正式批改。例如你要先检查优化后的评分准则是否合理。

后台模式：

```bash
python scripts/run_csbench.py optimize CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force --background
```

作用：只覆盖重跑 CO_1 到 CO_7 的评分准则优化，并把优化阶段放到服务器后台运行。

前台模式：

```bash
python scripts/run_csbench.py optimize CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force
```

作用：只覆盖重跑 CO_1 到 CO_7 的评分准则优化，不进入正式批改阶段。该模式会在当前终端持续输出优化过程。

### 3.4 只正式批改

适用场景：评分准则已经优化完成，只想重新批改 test 答案。

后台模式：

```bash
python scripts/run_csbench.py grade CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --background --force
```

作用：只批改 CO_1 到 CO_7 的全部 test 答案，并把正式批改放到服务器后台运行。运行前会检查每道题是否已有 optimized rubric 和 optimization manifest。

前台模式：

```bash
python scripts/run_csbench.py grade CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force
```

作用：只批改 CO_1 到 CO_7 的全部 test 答案，但在当前终端前台运行。适合短测试或排查错误。

### 3.5 查看状态、日志、进度监控和评估

```bash
python scripts/run_csbench.py status
```

作用：查看后台批改任务是否仍在运行，以及日志文件和进度文件位置。

```bash
python scripts/run_csbench.py tail
```

作用：实时查看后台批改日志。按 `Ctrl+C` 只会退出日志查看，不会停止后台批改任务。

```bash
python scripts/run_csbench.py monitor CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7
```

作用：实时查看 CO_1 到 CO_7 联合实验的结构化进度，包括当前运行状态、各题完成数量、失败数量和耗时。按 `Ctrl+C` 只会退出监控界面，不会停止后台任务。题目集合需要与启动 `run` 或 `grade` 时一致。

```bash
python scripts/run_csbench.py evaluate CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --export
```

作用：批改结束后评估 CO_1 到 CO_7 的结果，并导出逐答案对比 CSV。评估表现在默认包含 `SER(>2)`，即误差超过 2 分的严重错误率；失败样本会保存在 failed 文件中，正式分析时可按实验设计排除。

### 3.6 validation 重校准 A3WA/3WD → test 正式批改

适用场景：需要让三支决策逻辑基于新数据集重新校准，而不是继续使用旧数据集或旧 checkpoint 上得到的参数。该流程会先跑 validation，基于教师分数学习 A3WA 参数和 route/score-band 分数校准表，然后固定配置跑 test。

后台模式：

```bash
cd /home/E125221219/projects/refgrader
mkdir -p logs results_runs

nohup bash -lc '
set -e
source /home/E125221219/anaconda3/etc/profile.d/conda.sh
conda activate ref-grader

Q="CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7"
CFG="results_runs/csbench_co1_co2_co3_co4_co5_co6_co7_a3wa_calibration.json"

python scripts/run_csbench.py grade $Q --split validation --force

python scripts/run_csbench.py calibrate $Q --output "$CFG"

python scripts/run_csbench.py grade CO_1 --split test --force --a3wa-config "$CFG"
' > logs/csbench_route_score_calibrated_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

作用：

- validation 工作目录为 `results_runs/csbench_<题目集合>_validation`，与 test 目录隔离。
- validation 完整结束后立即发布一份不含 A3WA config 的原始中间产物，因此
  `refgrader-artifacts` 会马上显示 `validation_runs/<run_id>/` 更改；不完整结果不发布。
- `calibrate` 会精确校验 validation ID，并将 checkpoint、rubric、split 和 A3WA config
  再发布到新的 `validation_runs/<run_id>/`。前一份记录原始 validation 完成时点，后一份
  记录由它派生的校准配置，两者不可变且不计入 test 指标。
- `scripts/calibrate_a3wa.py` 现在会同时输出 `loss_params`、`risk_weights` 和 `score_calibration`。`score_calibration` 是基于 validation 残差学习出的题目/route/分数段加性校准表。
- test 批改通过 `--a3wa-config "$CFG"` 固定使用该配置。运行时不会读取 test 教师分数。
- test 完成后自动导出逐样本 `compare.csv` 和指标汇总 `summary.json`，再复制到
  `grading_runs/<run_id>/`，不会自动 Git push。

如果要在 validation/calibrate 完成后切换到另一台设备，先提交 artifacts：

```bash
cd ../refgrader-artifacts
git add csbench
git commit -m "Publish CO_1-CO_7 validation and A3WA calibration"
git push origin main
```

另一设备同时拉取 `refgrader-main` 和 `refgrader-artifacts`，再使用第 12.1 节的
`restore_csbench_artifacts.py`，不能只拉取代码仓库后直接运行 test。
- BND 降分逻辑已收紧：没有核心矛盾、允许的 agent over-evidence 或 direct-only 且核心支持不足时，不再自动 lower。
- fact mapping 失败时会保留原始转写或 OCR 文本作为降级事实，减少 `OCR fact mapping failed` 直接导致整条样本失败。

## 4. 评分准则优化

```bash
python scripts/run_csbench.py optimize CO_3
```

作用：使用 CO_3 的 5 份 calibration 答案优化评分准则。脚本自动设置题库、答案元数据、初始准则、优化准则、OCR缓存、结果目录和进度文件。

优化结果必须通过评分语义契约门禁。门禁不仅检查总分，还检查父项分值守恒、拆分策略、层次评分的唯一最终答案触发项和 `fallback_cap`。当前语义契约版本为 `2`；更新代码后，旧 manifest 会被正式批改入口拒绝，因此受影响题目需要先使用 `--force` 重新优化。CO_1 的初始准则已改为“最终答案正确直接满分、错误时按过程兜底”的层次评分，不能再压缩回单一 5 分条款。

层次评分覆盖整题且至少两次有效探测的严格多数由规范化器确定性命中最终答案时，满分属于准则硬约束，不再接受 BND Agent 或 validation 残差的数值下调；路由和人工复核标记仍保留用于风险审计。

优化成功后，脚本会自动把可移植的 CO_3 准则产物写入同级 `refgrader-artifacts` 仓库。服务器 VS Code 的 `refgrader-artifacts` 源代码管理会直接显示新增文件，你只需在该仓库点击暂存、提交和推送。

优化完成后自动提交并推送产物：

```bash
python scripts/run_csbench.py optimize CO_3 --push-artifacts
```

作用：优化完成后自动写入、提交并推送 `refgrader-artifacts`，无需在 VS Code 手动提交。

一次优化多个题目：

```bash
python scripts/run_csbench.py optimize CO_2 CO_3 CO_4
```

作用：在同一次前台任务中依次优化 CO_2、CO_3、CO_4；每道题独立使用自己的 calibration 答案，并分别生成 optimized 准则和 manifest。

如果 CO_3 已经生成过优化准则，并且确定需要重新优化：

```bash
python scripts/run_csbench.py optimize CO_3 --force
```

修改优化样本数：

```bash
python scripts/run_csbench.py optimize CO_3 --sample-size 10 --force
```

只查看脚本将执行的完整底层命令：

```bash
python scripts/run_csbench.py optimize CO_3 --dry-run
```

## 5. 正式后台批改

```bash
python scripts/run_csbench.py grade CO_3 --background --force
```

作用：后台批改 CO_3 全部 test 答案。脚本自动检查优化准则与 manifest，并使用 `csbench_hybrid`：普通答案读取 `raw_text`，视觉答案按条件调用 PaddleOCR 和 GLM-4.6V。
批改成功结束后，脚本会自动把完整评分产物复制到同级 `refgrader-artifacts` 仓库，源代码管理会直接显示该仓库的新增/修改文件；不需要再额外执行发布命令。

批改成功后自动复制、提交并推送产物：

```bash
python scripts/run_csbench.py grade CO_3 --background --force --push-artifacts
```

作用：后台批改完成后自动发布到 `refgrader-artifacts`，并自动提交、推送远程产物仓库。默认不加 `--push-artifacts` 时只产生本地 Git 更改，方便检查后手动提交。

一次后台批改多个题目：

```bash
python scripts/run_csbench.py grade CO_2 CO_3 CO_4 --background --force
```

作用：在同一个后台实验中依次批改 CO_2、CO_3、CO_4 的全部 test 答案，共享一个进度文件和结果目录；运行前会逐题检查优化准则与 manifest。

多题输出隔离规则：

- 优化准则始终逐题独立，例如 `CO_2_rubric_standard.json`、`CO_3_rubric_standard.json`。
- manifest 始终逐题独立，例如 `CO_2_optimization.json`、`CO_3_optimization.json`。
- 多题批改共享一个运行目录和 `progress.json`，但 checkpoint、graded、rejected、failed 均按题号独立命名。
- 例如联合批改 `CO_2 CO_3` 时，结果目录是 `results_runs/csbench_co2_co3_full/`，其中同时包含 `CO_2_*.json` 和 `CO_3_*.json`，不会把两道题的学生结果写入同一个 JSON。

只测试前 5 份 test 答案：

```bash
python scripts/run_csbench.py grade CO_3 --limit 5 --force
```

只查看完整底层命令：

```bash
python scripts/run_csbench.py grade CO_3 --background --force --dry-run
```

## 6. 查看运行状态

```bash
python scripts/run_csbench.py status
```

作用：查看后台实验是否运行。

```bash
python scripts/run_csbench.py tail
```

作用：实时查看后台日志；按 `Ctrl+C` 只退出日志查看。

```bash
python scripts/run_csbench.py monitor CO_3
```

作用：实时查看 CO_3 的结构化进度；按 `Ctrl+C` 只退出监控。

查看多题联合实验进度：

```bash
python scripts/run_csbench.py monitor CO_2 CO_3 CO_4
```

作用：打开 CO_2、CO_3、CO_4 联合批改任务对应的共享进度文件。题目集合需要与启动 `grade` 时一致，题目输入顺序可以不同。

## 7. 停止与断点续跑

```bash
python scripts/run_csbench.py stop
```

作用：优雅停止当前后台实验。

```bash
python scripts/run_csbench.py grade CO_3 --background
```

作用：断点续跑 CO_3。未使用 `--force`，因此保留 checkpoint 并跳过已经完成的答案。

多个题目断点续跑：

```bash
python scripts/run_csbench.py grade CO_2 CO_3 CO_4 --background
```

作用：继续同一组多题实验，逐题跳过已经写入 checkpoint 的答案，只处理未完成部分。

## 8. 正式评估

```bash
python scripts/run_csbench.py evaluate CO_3
```

作用：评估 CO_3 完整 checkpoint，对比 single、avg、selected 和最终 3WD 分数。

评估成功后，脚本会自动把完整实验产物写入 `refgrader-artifacts`。默认只产生可见的 Git 更改，方便你在源代码管理中检查后手动提交。

联合评估多个题目：

```bash
python scripts/run_csbench.py evaluate CO_2 CO_3 CO_4
```

作用：读取同一次多题 `grade` 生成的共享结果目录，分别评估 CO_2、CO_3、CO_4，并计算跨题目的全局对比指标。

注意：多个题目的 `grade`、`monitor` 和 `evaluate` 应使用相同的题目集合。若这些题目此前是分别单独批改的，则应分别执行单题评估，而不是使用联合评估命令。

```bash
python scripts/run_csbench.py evaluate CO_3 --export
```

作用：评估并导出逐答案对比 CSV。

评估后自动提交并推送精简产物：

```bash
python scripts/run_csbench.py evaluate CO_3 --export --push-artifacts
```

作用：评估完成后自动发布准则、评分结果、CSV和日志，并推送到远程产物仓库。默认不发布逐答案 facts 和原始 OCR 缓存，避免产物仓库出现大量缓存文件。

联合评估并导出 CSV：

```bash
python scripts/run_csbench.py evaluate CO_2 CO_3 CO_4 --export
```

作用：评估多个题目，并将各题 single、avg、selected、3WD 分数导出到同一个 CSV。

```bash
python scripts/run_csbench.py evaluate CO_3 --detail
```

作用：增加逐答案误差明细。

## 9. 查看输出文件位置

```bash
python scripts/run_csbench.py outputs CO_3
```

作用：不运行模型，直接显示 CO_3 的初始准则、优化准则、manifest、方差检查点、正式批改结果、OCR/事实缓存、评估 CSV 和日志位置。

查看多个题目联合实验的输出位置：

```bash
python scripts/run_csbench.py outputs CO_2 CO_3 CO_4
```

作用：显示多题共享运行目录，以及每道题各自独立的准则和结果文件。

### CO_3 单题完整输出示例

```text
data/csbench/rubrics/initial/CO/CO_3_rubric_standard.json
data/csbench/rubrics/optimized/CO/CO_3_rubric_standard.json
data/csbench/rubrics/manifests/CO/CO_3_optimization.json

results_runs/csbench_co3_rubric_opt/CO_3_variance_checkpoint.json
results_runs/csbench_co3_rubric_opt/progress.json

results_runs/csbench_co3_full/CO_3_grading_checkpoint.json
results_runs/csbench_co3_full/CO_3_graded_results.json
results_runs/csbench_co3_full/CO_3_rejected.json
results_runs/csbench_co3_full/CO_3_failed.json
results_runs/csbench_co3_full/progress.json

ocr_cache/csbench/variance_facts/CO_3/<answer_id>.json
ocr_cache/csbench/facts/CO_3/<answer_id>.json
ocr_cache/csbench/CO_3/<answer_id>.json

outputs/csbench_co3_compare.csv
logs/experiment_<run_id>.log
```

说明：

- `CO_3_grading_checkpoint.json` 保存该题全部成功完成评分的答案。
- `CO_3_graded_results.json` 保存非 NEG 答案。
- `CO_3_rejected.json` 仅在存在 NEG 答案时生成。
- `CO_3_failed.json` 仅在存在程序失败答案时生成。
- CO_3 当前准则不要求图形证据，因此正式 test 预计不会触发 PaddleOCR；`ocr_cache/csbench/CO_3/` 可能不存在或为空。
- `facts/CO_3/` 仍会保存由 `raw_text` 经 GLM-5.1 映射得到的评分点事实缓存。

## 10. 日常顺序

首次运行某道题：

```text
optimize → grade --background --force → status/tail → evaluate
```

准则已经优化完成：

```text
grade --background --force → status/tail → evaluate
```

实验中断：

```text
grade --background → evaluate
```

更换题目时只改题号：

```bash
python scripts/run_csbench.py optimize DM_2
```

```bash
python scripts/run_csbench.py grade DM_2 --background --force
```

```bash
python scripts/run_csbench.py evaluate DM_2 --export
```

## 11. refgrader-artifacts 发布目录说明

从本次版本开始，发布到 `refgrader-artifacts` 的结果按阶段分目录：

- 评分准则优化结果：`csbench/<题目ID>/rubric_optimizations/<run_id>/`
- validation及A3WA校准：`csbench/<题目ID>/validation_runs/<run_id>/`
- calibration split结果：`csbench/<题目ID>/calibration_runs/<run_id>/`
- 正式批改结果：`csbench/<题目ID>/grading_runs/<run_id>/`

例如：

```text
refgrader-artifacts/csbench/CO_2/rubric_optimizations/20260624_153000/
refgrader-artifacts/csbench/CO_2/validation_runs/20260713_010000/
refgrader-artifacts/csbench/CO_2/grading_runs/20260624_170000/
```

这样在 VS Code 中可以直接通过目录名区分“评分准则优化”和“正式批改”。历史结果如果仍在 `csbench/<题目ID>/runs/<run_id>/` 下，不需要迁移；新发布的结果会自动使用新结构。

## 12. 已有结果的补充发布

正常情况下不需要单独执行 `publish`：

- `optimize` 成功后自动发布准则阶段。
- 完整 validation/calibration 成功后发布到各自的 split 目录。
- `calibrate` 将 validation 与生成的 A3WA config 一起发布。
- 完整 test 成功后自动评估并发布正式评分产物。
- `evaluate` 成功后自动发布完整实验阶段。

`publish` 只用于迁移以前已经生成、但尚未进入产物仓库的历史结果。

仅发布已经存在的评分准则优化结果：

```bash
python scripts/run_csbench.py publish CO_2 --stage rubric --push
```

作用：将服务器已有的 CO_2 初始准则、优化准则、optimization manifest、方差检查点、优化事实缓存和日志复制到同级目录 `../refgrader-artifacts`，转换绝对路径后自动提交并推送。

正式批改和评估完成后发布完整结果：

```bash
python scripts/run_csbench.py publish CO_2 --stage full --push
```

补充发布 validation 与 A3WA 配置：

```bash
python scripts/run_csbench.py publish CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 \
  --stage validation \
  --a3wa-config results_runs/csbench_co1_co2_co3_co4_co5_co6_co7_a3wa_calibration.json
```

作用：除准则优化产物外，同时发布 grading checkpoint、graded/rejected/failed、评估CSV和日志。默认不发布 Stage1 facts 和原始 OCR 缓存。

自动判断当前可发布阶段：

```bash
python scripts/run_csbench.py publish CO_2 --push
```

作用：如果正式 checkpoint 已存在则发布完整实验，否则只发布准则优化阶段。

一次发布多个题目：

```bash
python scripts/run_csbench.py publish CO_2 CO_3 CO_4 --stage full --push
```

作用：使用同一个 run ID 将多个题目分别发布到 `csbench/CO_2/grading_runs/`、`csbench/CO_3/grading_runs/`、`csbench/CO_4/grading_runs/`，每道题的文件保持独立。

本地同步服务器产物：

```powershell
git -C C:\Users\wx\Desktop\refgrader-artifacts pull origin main
```

作用：将服务器已发布的实验产物拉取到本地，供 VS Code 和 Codex 分析。

### 12.1 在另一台设备恢复 validation 并继续实验

拉取两个仓库后执行：

```powershell
cd D:\Users\王鑫020420\Desktop\refgrader-main

python scripts\restore_csbench_artifacts.py `
  CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 `
  --stage validation `
  --config-output results_runs\csbench_co1_co2_co3_co4_co5_co6_co7_a3wa_calibration.json `
  --force
```

Linux/实验室服务器对应命令：

```bash
cd /home/E125221219/projects/refgrader
python scripts/restore_csbench_artifacts.py \
  CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 \
  --stage validation \
  --config-output results_runs/csbench_co1_co2_co3_co4_co5_co6_co7_a3wa_calibration.json \
  --force
```

该命令恢复 optimized rubric、optimization manifest、七题 validation checkpoint、
graded/rejected/failed、progress 和 A3WA config，并校验本机初始 rubric 与固定 split。
恢复成功后可以直接使用该 config 在另一设备运行 test。


## 13. 本地评估从 refgrader-artifacts 拉取的 CO_1 到 CO_7 结果

`refgrader-artifacts` 的发布结果按题号分目录保存，例如：

```text
refgrader-artifacts/csbench/CO_1/grading_runs/<run_id>/grading/grading_checkpoint.json
```

`evaluate.py` 需要一个扁平结果目录，文件名必须是：

```text
CO_1_grading_checkpoint.json
CO_2_grading_checkpoint.json
...
```

因此本地评估前先把 artifacts 中的 checkpoint 汇总到 `refgrader-main/results_runs/...`。

### 13.1 拉取服务器 artifacts

```powershell
git -C C:\Users\wx\Desktop\refgrader-artifacts pull origin main
```

### 13.2 汇总 CO_1 到 CO_7 的 checkpoint

把 `$run` 改成实际 artifacts 目录名。本次新数据集 A3WA 重校准后的 run id 示例是 `20260628_081803`。

```powershell
cd C:\Users\wx\Desktop\refgrader-main

$run = "20260628_081803"
$questions = "CO_1","CO_2","CO_3","CO_4","CO_5","CO_6","CO_7"
$srcRoot = "..\refgrader-artifacts\csbench"
$out = "results_runs\csbench_co1_co2_co3_co4_co5_co6_co7_full"

New-Item -ItemType Directory -Force $out | Out-Null

foreach ($q in $questions) {
  Copy-Item "$srcRoot\$q\grading_runs\$run\grading\grading_checkpoint.json" "$out\${q}_grading_checkpoint.json" -Force
  Copy-Item "$srcRoot\$q\grading_runs\$run\grading\graded_results.json" "$out\${q}_graded_results.json" -Force
  if (Test-Path "$srcRoot\$q\grading_runs\$run\grading\rejected.json") {
    Copy-Item "$srcRoot\$q\grading_runs\$run\grading\rejected.json" "$out\${q}_rejected.json" -Force
  }
  if (Test-Path "$srcRoot\$q\grading_runs\$run\grading\failed.json") {
    Copy-Item "$srcRoot\$q\grading_runs\$run\grading\failed.json" "$out\${q}_failed.json" -Force
  }
}
```

### 13.3 本地重新评估并导出 CSV

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null

python evaluate.py `
  --questions CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 `
  --results-dir results_runs\csbench_co1_co2_co3_co4_co5_co6_co7_full `
  --result-source checkpoint `
  --teacher-db data\csbench\teacher_scores.json `
  --database-path data\csbench\exam_database.json `
  --compare `
  --compare-score-keys single avg selected 3wd `
  --compare-output outputs\csbench_co1_co2_co3_co4_co5_co6_co7_compare_local.csv
```

作用：在本地重新计算 CO_1 到 CO_7 的 single、avg、selected 和 3WD 指标，并导出逐答案对比 CSV。

只评估单题，例如 CO_7：

```powershell
python evaluate.py `
  --questions CO_7 `
  --results-dir results_runs\csbench_co1_co2_co3_co4_co5_co6_co7_full `
  --result-source checkpoint `
  --teacher-db data\csbench\teacher_scores.json `
  --database-path data\csbench\exam_database.json `
  --compare `
  --compare-score-keys single avg selected 3wd `
  --detail
```

### 13.4 推荐短命令：直接评估 artifacts 结果

如果结果已经从服务器推送到 `refgrader-artifacts`，本地先拉取一次：

```powershell
git -C C:\Users\wx\Desktop\refgrader-artifacts pull origin main
```

然后在 `refgrader-main` 目录执行一条命令即可评估 CO_1 到 CO_7，并导出逐答案 CSV。这里显式使用 `.\venv\Scripts\python.exe`，避免误用 PaddleOCR 专用的 `.venv-ocr` 环境：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --run-id 20260630_134044 --score-key single avg selected 3wd --export
```

如果不写 `--run-id`，脚本会自动使用 `CO_1` 下最新的 grading run：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --score-key single avg selected 3wd --export
```

只看最终三支决策分数：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --score-key 3wd
```

自由搭配任意几种分数形式。传入多个 `--score-key` 时会自动进入对比模式：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --score-key single avg 3wd
```

例如只对比 selected 和最终 3WD：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --score-key selected 3wd
```

对比 single、avg、selected、3WD 四种分数：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --compare
```

查看单题明细，例如 CO_7：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_7 --run-id 20260630_134044 --detail --compare
```

如果当前 PowerShell 提示符显示 `(.venv-ocr)`，说明你之前在这个终端里执行过 `.venv-ocr\Scripts\Activate.ps1`。该状态会一直保留到执行 `deactivate` 或关闭终端为止。`.venv-ocr` 只用于 PaddleOCR，不用于评估；评估请使用上面的 `.\venv\Scripts\python.exe` 命令，或者先执行：

```powershell
deactivate
```

脚本会自动把 `refgrader-artifacts/csbench/<题号>/grading_runs/<run_id>/grading/` 下的结果复制到 `results_runs/artifacts_<题号集合>_<run_id>/`，再调用原来的 `evaluate.py`。原来的 `python scripts/run_csbench.py evaluate ...` 仍然用于服务器端刚批改完后的评估；本地评估已经拉取的 artifacts 时，优先使用上面的短命令。
