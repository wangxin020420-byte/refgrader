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

## 3. CO_1 到 CO_7 四种常用执行方式

下面四种命令都以 CO_1 到 CO_7 为例。`--force` 表示覆盖已有优化准则或旧 checkpoint；`--background` 表示正式批改阶段在服务器后台运行。

### 3.1 重新生成兼容视图 → 重新优化评分准则 → 正式后台批改

适用场景：数据集刚更新过、刚修改过 `CSBench_new/question/question_CO.json`、刚拉取过数据集，或者不确定 `data/csbench` 是否最新。推荐用于这次 CO_1 到 CO_7 全量重跑。

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --dataset-root /home/E125221219/CSBench_new --force --background
```

作用：一条命令完成 CO_1 到 CO_7 的完整实验。执行顺序是：先根据 `/home/E125221219/CSBench_new` 重新生成 `data/csbench` 兼容视图，再覆盖重跑 CO_1 到 CO_7 的评分准则优化，最后后台启动 CO_1 到 CO_7 的正式批改。`--background` 只作用于正式批改阶段，评分准则优化会先在前台完成，避免准则没生成就开始批改。

### 3.2 重新优化评分准则 → 正式后台批改

适用场景：`data/csbench` 已经是最新的，不需要重新生成兼容视图，但需要覆盖旧评分准则并重新批改。

```bash
python scripts/run_csbench.py run CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force --background
```

作用：跳过兼容视图生成，直接基于当前 `data/csbench` 覆盖重跑 CO_1 到 CO_7 的评分准则优化，然后后台启动正式批改。

### 3.3 只重新优化评分准则

适用场景：只想重新生成 optimized rubric，先不正式批改。例如你要先检查优化后的评分准则是否合理。

```bash
python scripts/run_csbench.py optimize CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --force
```

作用：只覆盖重跑 CO_1 到 CO_7 的评分准则优化，不进入正式批改阶段。

### 3.4 只正式后台批改

适用场景：评分准则已经优化完成，只想重新批改 test 答案。

```bash
python scripts/run_csbench.py grade CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --background --force
```

作用：只后台批改 CO_1 到 CO_7 的全部 test 答案。运行前会检查每道题是否已有 optimized rubric 和 optimization manifest。

### 3.5 查看状态、日志和评估

```bash
python scripts/run_csbench.py status
```

作用：查看后台批改任务是否仍在运行，以及日志文件和进度文件位置。

```bash
python scripts/run_csbench.py tail
```

作用：实时查看后台批改日志。按 `Ctrl+C` 只会退出日志查看，不会停止后台批改任务。

```bash
python scripts/run_csbench.py evaluate CO_1 CO_2 CO_3 CO_4 CO_5 CO_6 CO_7 --export
```

作用：批改结束后评估 CO_1 到 CO_7 的结果，并导出逐答案对比 CSV。失败样本会保存在 failed 文件中，正式分析时可按实验设计排除。

## 4. 评分准则优化

```bash
python scripts/run_csbench.py optimize CO_3
```

作用：使用 CO_3 的 5 份 calibration 答案优化评分准则。脚本自动设置题库、答案元数据、初始准则、优化准则、OCR缓存、结果目录和进度文件。

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
- 正式批改结果：`csbench/<题目ID>/grading_runs/<run_id>/`

例如：

```text
refgrader-artifacts/csbench/CO_2/rubric_optimizations/20260624_153000/
refgrader-artifacts/csbench/CO_2/grading_runs/20260624_170000/
```

这样在 VS Code 中可以直接通过目录名区分“评分准则优化”和“正式批改”。历史结果如果仍在 `csbench/<题目ID>/runs/<run_id>/` 下，不需要迁移；新发布的结果会自动使用新结构。

## 12. 已有结果的补充发布

正常情况下不需要单独执行 `publish`：

- `optimize` 成功后自动发布准则阶段。
- `grade --background` 成功结束后自动发布完整评分产物阶段。
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

