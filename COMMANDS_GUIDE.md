# RefGrader 正式实验命令

CSBench 实验统一使用 `scripts/run_csbench.py`。更换题目时只需要修改题号，例如把 `CO_2` 改成 `CO_3`。

> 当前默认模型契约（2026-07-21）：文本评分统一使用 `glm-4.7`，并显式发送
> `thinking.type=disabled`；视觉提取仍使用 `glm-4.6v`。该契约同时覆盖语义评分、
> OCR 事实映射和评分准则优化中的文本裁判。每个 validation/test 运行及 A3WA 配置
> 都记录模型签名，模型或思考模式不一致时脚本会拒绝复用旧结果。

## 0. 日常最短命令

本节是日常实验的默认入口。评分准则、七题 validation 和 A3WA 配置已经生成并提交后，当前批准的配置位于 `data/csbench`。任何设备拉取 `refgrader-main` 后都使用同一份哈希校验配置，不再从本机 `results_runs` 猜测“最近一个”配置。

只正式批改任意题目，例如 CO_5、CO_6：

```powershell
.\venv\Scripts\python.exe scripts\run_csbench.py grade CO_5 CO_6 --split test --force
```

以后只需要替换 `CO_5 CO_6`。该命令在前台运行，本地电脑必须保持开机、联网且不能休眠。
完整结束后自动评估并复制到同级 `refgrader-artifacts`，但不会自动 Git 提交或推送。
`--force` 会自动创建 `runs/<时间戳>/`，不会覆盖以前相同题目组合的结果。

如果任务中断，使用同一个题目集合和配置断点续跑，**不要使用 `--force`**：

```powershell
.\venv\Scripts\python.exe scripts\run_csbench.py grade CO_5 CO_6 --split test
```

程序会从批次目录的 `active_run.json` 找到刚才的 `run_id`。如需续跑指定历史实验：

```powershell
.\venv\Scripts\python.exe scripts\run_csbench.py grade CO_5 CO_6 --split test --run-id 20260716_210000
```

Linux/服务器已经有固定配置时，同样只需要一条核心命令：

```bash
python scripts/run_csbench.py grade CO_5 CO_6 --split test --force
```

这些 test 命令会自动读取 `data/csbench/calibration/active_a3wa_config.json`，并用 `data/csbench/rubrics/active_rubric_set.json` 验证其题目覆盖、rubric 哈希和数据快照。配置缺失、过期或不覆盖待测题目时会直接停止，避免静默使用默认参数。只有进行历史对照实验时才显式传 `--a3wa-config <文件>`；明确执行无校准配置消融时传 `--no-active-a3wa`。

`optimize` 成功后会更新 Git 可见的 `rubrics/optimized`、`rubrics/manifests` 和 `active_rubric_set.json`；`calibrate` 成功后还会更新 `calibration/active_a3wa_config.json`。这两个阶段结束后应提交 `refgrader-main`，历史运行证据仍单独提交到 `refgrader-artifacts`。

只有以下情况才进入后文的完整流程：数据快照发生变化、评分语义或 rubric 发生变化、
三支决策校准逻辑发生变化，或者需要重新生成 A3WA 配置。Windows 隐藏后台、日志重定向和
PID 管理只是长时间运行的可选包装，不是批改必须步骤。

### 0.1 切换到 GLM-4.7 后的首次实验

旧 validation/A3WA 配置由其他模型生成，不能直接用于新模型。评分准则不变时，按以下
顺序重新建立模型匹配的校准链，然后再跑 test。当前仓库还需先从 artifacts 恢复哈希
一致的已批准评分准则（不会重新调用模型优化准则）：

```powershell
$Q = @("CO_1","CO_2","CO_3","CO_4","CO_5","CO_6","CO_7")
.\venv\Scripts\python.exe scripts\restore_csbench_artifacts.py CO_1 CO_2 --stage rubric --run-id 20260719_035814 --force
.\venv\Scripts\python.exe scripts\restore_csbench_artifacts.py CO_3 CO_4 CO_5 CO_6 CO_7 --stage rubric --run-id 20260720_125147 --force
.\venv\Scripts\python.exe scripts\run_csbench.py grade @Q --split validation --force
.\venv\Scripts\python.exe scripts\run_csbench.py calibrate @Q --score-calibration
.\venv\Scripts\python.exe scripts\run_csbench.py grade CO_4 --split test --force
```

上述命令无需重复写模型参数，因为默认值已经是 `--text-provider glm47
--thinking-mode disabled --vlm-provider glm4v`。需要做对照实验时才显式覆盖，例如：

```powershell
.\venv\Scripts\python.exe scripts\run_csbench.py grade CO_4 --split test --force --text-provider glm5 --thinking-mode enabled
```

不同模型契约必须分别运行 validation 和 calibrate，不能共用 A3WA 配置。

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

`grade --split test` 在批次正常收尾后会自动执行以下操作，不需要再单独调用
`evaluate` 或 `publish`：

```text
校验 checkpoint 与 test split 的结构一致性
-> 写入 complete/partial、覆盖率、missing/failed ID
-> 对成功 checkpoint 评估 single / avg / selected / 3wd-core / 3wd
-> 导出 results_runs/csbench_<题目集合>_full/runs/<run_id>/evaluation/compare.csv
   和 evaluation/summary.json
-> 使用同一个 run_id 复制实验产物到同级 refgrader-artifacts
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
  grading/failed.json                 # 存在失败样本时保存
  grading/completion_report.json      # complete/partial、覆盖率和失败ID
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
test artifacts。缺失或 failed 样本允许以 `partial` 发布，续跑完成后更新同一 run_id；
但 calibration 只接受完整 validation。`--limit` 调试运行仍不自动发布，split 污染、
重复 ID 或 checkpoint/graded/rejected 冲突会直接阻止评估和发布。`--include-facts` 和
`--include-raw-ocr` 仍为可选项，默认不复制大体积逐答案缓存。

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

- rubric 优化中的 `--force` 表示重新生成准则；grade 中的 `--force` 表示创建新的时间戳实验，不再覆盖旧 checkpoint。
- `--force` 不会无条件重算原始 PaddleOCR；图片 SHA-256 与现有 OCR JSON 一致时复用缓存，只重跑准则、事实映射和评分。需要重新识别原图时，应单独删除对应 OCR JSON 或显式运行 `paddle_ocr_worker.py --force`。
- PaddleX 模型文件在每台设备上只使用一个权威缓存。Windows 默认是项目所在盘根目录的 `paddlex_cache`（例如 `D:\paddlex_cache`），Linux 默认是 `~/.cache/refgrader/paddlex`；可用 `PADDLE_PDX_CACHE_HOME` 显式覆盖。主流程和直接调用 OCR worker 使用同一规则。
- `ocr_cache/csbench` 是逐图片的可审计 OCR 证据缓存，不是 PaddleX 模型缓存。它包含图片 SHA-256、识别文本和置信度，不应在清理重复模型文件时删除。
- 主流水线发生 OCR、模型或文件异常时返回非零状态，后续 artifact 发布和下一实验阶段不会继续执行。
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

如果上一次批量优化在中途失败，使用断点续传模式，不要再次添加
`--force`：

```bash
python scripts/run_csbench.py optimize CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --resume
```

`--resume` 会校验每道题的 optimized rubric、manifest、语义契约版本和
SHA-256。已经完整有效的题目直接跳过；未完成或版本过期的题目继续处理，
并复用同一批次 `results_runs/..._rubric_opt` 中已经保存的 calibration
checkpoint。`--resume` 与 `--force` 互斥：前者保留进度，后者明确从初始准则
重新开始全部题目。

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

适用场景：需要让三支决策逻辑基于新数据集重新校准，而不是继续使用旧数据集或旧 checkpoint 上得到的参数。该流程会先跑 validation，基于教师分数学习单调安全隶属度、conformal 不确定性区间并选择 A3WA 非对称损失参数，然后固定配置跑 test。默认不启用残差改分。

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

- validation 批次根目录为 `results_runs/csbench_<题目集合>_validation`，实际结果位于其 `runs/<run_id>/`，与 test 目录隔离。
- validation 完整结束后立即发布一份不含 A3WA config 的原始中间产物，因此
  `refgrader-artifacts` 会显示 `validation_runs/<run_id>/` 更改；不完整结果标记为 partial，但不能用于 calibrate。
- `calibrate` 会精确校验 validation ID，并将 checkpoint、rubric、split 和 A3WA config
  再发布到新的 `validation_runs/<run_id>/`。前一份记录原始 validation 完成时点，后一份
  记录由它派生的校准配置，两者不可变且不计入 test 指标。
- `scripts/calibrate_a3wa.py` 输出 `membership_model`、`score_uncertainty`、`loss_params`、`boundary_policy`、跨题 LOQO 诊断和 `deployment_gate`。
- `score_calibration.enabled` 默认是 `false`。若要在同一次 test 中同时评估纯三支决策和残差层，给 `calibrate` 增加 `--score-calibration`；每个 test checkpoint 会同时保存 `three_way_core_score` 和 `final_calibrated_score`，不需要重复批改。
- `deployment_gate.passed=false` 表示该配置未同时满足跨题非劣、路由预算和 validation BND 正收益；配置仍可用于研究性实验，但不能宣称已经通过部署验证。
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

核心机制与残差消融的短评估命令：

```powershell
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --score-key avg 3wd-core 3wd --export
```

其中 `3wd-core` 是纯三支路由与结构化 BND action 的结果；`3wd` 是可选残差层之后的最终结果。默认校准下二者应相同。
- 正式自动评估默认比较 `single`、`avg`、`selected`、`3wd-core` 和 `3wd`，并额外输出两段配对消融：`avg -> 3wd-core` 与 `3wd-core -> 3wd`。
- 配对消融表包含两端 MAE、平均绝对误差增益、改善/不变/恶化数量、平均分数改变量和 Wilcoxon 配对检验 p 值；`summary.json` 的 `score_ablation` 保存同一口径的机器可读结果。
- BND 降分逻辑已收紧：没有核心矛盾、允许的 agent over-evidence 或 direct-only 且核心支持不足时，不再自动 lower。
- fact mapping 失败时会保留原始转写或 OCR 文本作为降级事实，减少 `OCR fact mapping failed` 直接导致整条样本失败。

Windows 本地完整流程如下。该示例使用 CO_1 到 CO_7 做 validation 和校准，只正式批改 CO_3、CO_4；以后只修改 `$TestQuestions` 即可选择 CO_3 到 CO_7 中任意题目。命令在当前 PowerShell 前台连续执行，当前提示符即使显示 `.venv-ocr` 也没有影响，因为每一步都显式调用主环境 `venv`：

```powershell
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path ".\scripts\run_csbench.py")) {
    throw "请先在 refgrader-main 项目根目录执行"
}

$Python = (Resolve-Path ".\venv\Scripts\python.exe").Path
$ValidationQuestions = @("CO_1", "CO_2", "CO_3", "CO_4", "CO_5", "CO_6", "CO_7")
$TestQuestions = @("CO_3", "CO_4")
$RubricRun = "20260714_022104"
$Config = "results_runs\csbench_all7_a3wa_core_residual.json"

& $Python scripts\audit_csbench_snapshot.py --prepared-dir data\csbench
if ($LASTEXITCODE -ne 0) { throw "数据快照检查失败" }

& $Python -m unittest test_a3wa_theory.py test_evaluate_ablation.py test_csbench_artifact_sync.py -q
if ($LASTEXITCODE -ne 0) { throw "代码测试失败" }

# 本轮没有修改评分准则；恢复已经核验的七题共同 rubric 批次。
& $Python scripts\restore_csbench_artifacts.py @ValidationQuestions --stage rubric --run-id $RubricRun --force
if ($LASTEXITCODE -ne 0) { throw "恢复既有评分准则失败" }

& $Python scripts\run_csbench.py grade @ValidationQuestions --split validation --dry-run --no-artifacts
if ($LASTEXITCODE -ne 0) { throw "评分准则或 manifest 校验失败" }

& $Python scripts\run_csbench.py grade @ValidationQuestions --split validation --force
if ($LASTEXITCODE -ne 0) { throw "validation 批改失败" }

& $Python scripts\run_csbench.py calibrate @ValidationQuestions --output $Config --score-calibration
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Config)) { throw "A3WA 校准失败" }

$Calibration = Get-Content $Config -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $Calibration.score_calibration.enabled) {
    throw "残差层没有启用，停止 test"
}

& $Python scripts\run_csbench.py grade @TestQuestions --split test --force --a3wa-config $Config
if ($LASTEXITCODE -ne 0) { throw "test 正式批改或自动评估失败" }

Write-Host "全部完成。正式评估已自动比较 avg、3wd-core 和 3wd。"
git -C "..\refgrader-artifacts" status --short
```

最后一条 `grade` 只进行一次 test 模型调用。完成后会自动运行新版评估、生成 `evaluation/compare.csv` 与 `evaluation/summary.json`，并复制 validation、配置、准则和正式结果到同级 `refgrader-artifacts`；不会自动提交或推送。
历史 checkpoint 如果没有 `three_way_core_score`，评估器不会用最终 `3wd` 冒充 core；要得到有效的两段消融，必须使用本版本代码重新执行 test。
本轮 A3WA/评估修改没有改变 rubric 生成算法或评分语义契约，因此不执行 `optimize`。只有初始评分准则、准则优化算法或语义契约发生变化，或者 artifacts 中没有可恢复的有效 rubric 时，才需要重新优化评分准则。

## 4. 评分准则优化

```bash
python scripts/run_csbench.py optimize CO_3
```

作用：使用 CO_3 的 5 份 calibration 答案优化评分准则。脚本自动设置题库、答案元数据、初始准则、优化准则、OCR缓存、结果目录和进度文件。

优化结果必须通过评分语义契约门禁。当前版本为 `5`：高分父项先分类为结果充分、正交结果、组成部分、过程主导或严格原子；正交/组成部分才允许等权，复杂过程题要求过程至少 80% 且结论不超过 20%，短过程题要求过程至少 65% 且结论不超过 35%。门禁还检查父项分值守恒、角色完整性、事实锚点守恒、层次评分唯一触发项和 `fallback_cap`，并在 calibration 教师分上执行候选非劣回放。旧 manifest 会被正式批改入口拒绝，因此本次必须对 CO_1 至 CO_7 使用 `--force` 重优化。CO_1 的显式结果充分层次评分会保留。

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
- 例如联合批改 `CO_2 CO_3` 时，批次根目录是 `results_runs/csbench_co2_co3_full/`，每次实验位于 `runs/<run_id>/`；同一运行中同时包含 `CO_2_*.json` 和 `CO_3_*.json`，不会把两道题的学生结果写入同一个 JSON。

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

results_runs/csbench_co3_full/runs/<run_id>/CO_3_grading_checkpoint.json
results_runs/csbench_co3_full/runs/<run_id>/CO_3_graded_results.json
results_runs/csbench_co3_full/runs/<run_id>/CO_3_rejected.json
results_runs/csbench_co3_full/runs/<run_id>/CO_3_failed.json
results_runs/csbench_co3_full/runs/<run_id>/progress.json

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
  --compare-score-keys single avg selected 3wd-core 3wd `
  --compare-output outputs\csbench_co1_co2_co3_co4_co5_co6_co7_compare_local.csv
```

作用：在本地重新计算 CO_1 到 CO_7 的 single、avg、selected、3WD-Core 和最终 3WD 指标，并导出逐答案对比 CSV。

只评估单题，例如 CO_7：

```powershell
python evaluate.py `
  --questions CO_7 `
  --results-dir results_runs\csbench_co1_co2_co3_co4_co5_co6_co7_full `
  --result-source checkpoint `
  --teacher-db data\csbench\teacher_scores.json `
  --database-path data\csbench\exam_database.json `
  --compare `
  --compare-score-keys single avg selected 3wd-core 3wd `
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
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --run-id 20260630_134044 --score-key single avg selected 3wd-core 3wd --export
```

如果不写 `--run-id`，脚本会自动使用 `CO_1` 下最新的 grading run：

```powershell
cd C:\Users\wx\Desktop\refgrader-main
.\venv\Scripts\python.exe scripts\evaluate_artifacts.py CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --score-key single avg selected 3wd-core 3wd --export
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

对比 single、avg、selected、3WD-Core、最终 3WD 五种分数：

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
