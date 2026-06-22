# RefGrader 常用命令顺序说明

本文档用于记录本项目从代码检查、评分准则生成、正式批改到结果评估的常用命令。后续新对话或服务器实验时，优先参考本文档。

## 1. 每次实验前的代码检查

```bash
python -m py_compile calibration_utils.py step4_vlm_grader.py main_pipeline.py step3_rrd_generator.py evaluate.py scripts/replay_calibration.py
```

作用：检查主要 Python 文件是否存在语法错误。该命令不调用大模型，不会产生新的实验结果。

## 2. 本地前台运行

```bash
python main_pipeline.py --mode FULL --questions Q2 Q3 --progress-file results_rrd_vlm/progress_q2_q3_local.json
```

作用：在本地前台运行 Q2、Q3 的正式批改。终端关闭后任务会停止，适合本机短任务或调试。

```bash
python main_pipeline.py --mode FULL --questions Q2 Q3 --force-rerun --progress-file results_rrd_vlm/progress_q2_q3_local.json
```

作用：强制重跑 Q2、Q3。会清理对应题目的旧 `checkpoint / graded_results / rejected / failed` 文件，避免新旧结果混用。

## 3. 实验室服务器后台运行

```bash
./run_experiment.sh run --mode FULL --questions Q4 Q5 --progress-file results_rrd_vlm/progress_q4_q5_server.json
```

作用：在实验室服务器后台运行 Q4、Q5。`run_experiment.sh` 内部已经使用 `nohup`，SSH 断开或本地电脑关闭后，服务器任务仍会继续。

```bash
./run_experiment.sh run --mode FULL --questions Q2 Q3 Q4 --force-rerun --progress-file results_rrd_vlm/progress_q2_q4_clean.json
```

作用：服务器后台强制重跑 Q2、Q3、Q4。适合修复代码后重新生成干净结果。

```bash
./run_experiment.sh status
```

作用：查看后台实验是否仍在运行，以及当前 PID 和日志文件位置。

```bash
./run_experiment.sh tail
```

作用：实时查看最新实验日志。

```bash
./run_experiment.sh stop
```

作用：优雅停止后台实验。优先发送 `SIGTERM`，让当前样本处理结束后退出。

## 4. 评分准则生成与优化

```bash
python main_pipeline.py --mode VARIANCE_OPT --questions Q2 Q3 Q4 Q5 --sample-size 5 --force-rerun --progress-file results_rrd_vlm/progress_rubric_q2_q5_clean.json
```

作用：对 Q2-Q5 重新执行评分准则方差优化。`--sample-size 5` 表示每题抽取 5 个样本用于高方差检查和 rubric 修正。

```bash
./run_experiment.sh run --mode VARIANCE_OPT --questions Q2 Q3 Q4 Q5 --sample-size 5 --force-rerun --progress-file results_rrd_vlm/progress_rubric_q2_q5_server.json
```

作用：在服务器后台生成或优化 Q2-Q5 的评分准则。适合耗时较长的 rubric 优化任务。

## 5. 正式结果评估

```bash
python evaluate.py --questions Q4 Q5 Q6 Q7
```

作用：评估默认最终三支决策分数 `final_calibrated_score`。如果不加 `--result-source checkpoint`，默认读取 `Qx_graded_results.json`，通常不包含 NEG/rejected 样本。

```bash
python evaluate.py --result-source checkpoint --compare --questions Q2 Q3 Q4 Q5 --compare-score-keys single avg selected 3wd
```

作用：正式推荐评估命令。读取完整 `Qx_grading_checkpoint.json`，纳入 POS、BND、NEG 全部样本，并对比四种分数来源：

- `single`：第一次模型评分 `single_first_score`
- `avg`：三次模型评分均分 `model_avg_score`
- `selected`：三支决策选择后的基础分 `selected_baseline_score`
- `3wd`：最终三支决策分数 `final_calibrated_score`

```bash
python evaluate.py --result-source checkpoint --compare --questions Q6 Q7 --compare-score-keys single avg selected 3wd
```

作用：只评估 Q6、Q7，适合对论文主力题进行快速对比。

```bash
python evaluate.py --result-source checkpoint --compare --questions Q2 Q3 Q4 Q5 --compare-score-keys single avg selected 3wd --compare-output outputs/q2_q5_clean_compare.csv
```

作用：除终端输出指标表外，额外导出逐学生 CSV 明细，便于分析每个学生的 single / avg / selected / 3WD 差异。

```bash
python evaluate.py --questions Q2 Q3 --detail --result-source checkpoint
```

作用：输出逐学生误差明细，并按绝对误差从大到小排序，适合定位严重高估或低估样本。

## 6. 独立结果目录运行与评估

```bash
python main_pipeline.py --mode FULL --questions Q4 --force-rerun --results-dir results_runs/q4_verify --rubric-dir results_rrd_vlm --progress-file results_runs/q4_verify/progress.json
```

作用：把 Q4 的新实验结果写入独立目录 `results_runs/q4_verify`，但评分准则仍从 `results_rrd_vlm` 读取。适合单题验证，不污染主结果目录。

```bash
python evaluate.py --result-source checkpoint --results-dir results_runs/q4_verify --compare --questions Q4 --compare-score-keys single avg selected 3wd
```

作用：评估独立 run 目录中的 Q4 结果。

## 7. 离线 replay 验证

```bash
python scripts/replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm/Q6_grading_checkpoint.json results_rrd_vlm/Q7_grading_checkpoint.json
```

作用：不重新调用 VLM，只基于已有 checkpoint 重放后校准和三支决策逻辑。适合快速检查代码层策略是否可能改善，但不能验证视觉重提取效果。

```bash
python scripts/replay_calibration.py --results-dir results_rrd_vlm --files results_rrd_vlm/Q2_grading_checkpoint.json results_rrd_vlm/Q3_grading_checkpoint.json results_rrd_vlm/Q4_grading_checkpoint.json
```

作用：离线重放 Q2-Q4 的校准逻辑，适合正式重跑前做安全检查。

## 8. 推荐的完整实验顺序

```bash
python -m py_compile calibration_utils.py step4_vlm_grader.py main_pipeline.py step3_rrd_generator.py evaluate.py scripts/replay_calibration.py
```

第一步：先确认代码无语法错误。

```bash
./run_experiment.sh run --mode FULL --questions Q2 Q3 Q4 --force-rerun --progress-file results_rrd_vlm/progress_q2_q4_clean.json
```

第二步：在服务器后台强制重跑目标题目，生成干净 checkpoint。

```bash
./run_experiment.sh status
```

第三步：查看实验是否仍在运行。

```bash
./run_experiment.sh tail
```

第四步：实时查看日志，确认是否有失败样本或一致性警告。

```bash
python evaluate.py --result-source checkpoint --compare --questions Q2 Q3 Q4 --compare-score-keys single avg selected 3wd --compare-output outputs/q2_q4_clean_compare.csv
```

第五步：实验结束后评估完整结果，并导出逐学生对比 CSV。

## 9. 注意事项

1. `--questions` 优先级最高，会覆盖 `main_pipeline.py` 配置区中的题目设置。
2. 服务器后台实验优先使用 `./run_experiment.sh run`，不要再手动套一层 `nohup`。
3. 正式论文分析建议使用 `--result-source checkpoint`，因为它包含 POS、BND、NEG 全部样本。
4. 如果使用 `--force-rerun`，当前代码会清理对应题目的旧结果文件，防止新旧结果混用。
5. 如果出现 `Qx_failed.json`，说明有样本运行失败，需要结合日志和 failed 文件分析原因。
6. replay 只能验证已有结果上的后校准逻辑，不能验证重新视觉提取和重新评分的真实效果。

## 10. PaddleOCR 本地独立测试

首次安装或重建独立 OCR 环境：

```powershell
.\scripts\setup_paddle_ocr.ps1
```

作用：在项目目录创建 `.venv-ocr`，安装 CPU 版 PaddlePaddle 和 PaddleOCR。
该环境与现有 `venv` 相互独立，不会修改正式评分环境。

测试单张正常作答图片：

```powershell
.\scripts\run_paddle_ocr.ps1 `
  -InputPath cleaned_patches\Q5\E01914115_Q5.jpg `
  -OutputDir ocr_cache\q5_test
```

测试已确认的空白图片：

```powershell
.\scripts\run_paddle_ocr.ps1 `
  -InputPath cleaned_patches\Q5\E02014181_Q5.jpg `
  -OutputDir ocr_cache\q5_blank_test
```

批量识别一个题目的全部答卷：

```powershell
.\scripts\run_paddle_ocr.ps1 `
  -InputPath cleaned_patches\Q2 `
  -OutputDir ocr_cache\Q2
```

作用：把每张图片的识别文字、置信度、坐标和图片哈希保存为独立 JSON。
当前只用于评估 OCR 能力，尚未替换正式 Stage1 视觉提取。

## 11. 可选 OCR 正式后端与分阶段模式

只提取并写缓存：

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode OCR_ONLY --questions Q2 --student-ids E01914115 `
  --extraction-backend paddle_glm5 `
  --results-dir results_runs\q2_ocr_backend_test `
  --rubric-dir results_rrd_vlm
```

只读取缓存评分：

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode GRADE_ONLY --questions Q2 --student-ids E01914115 `
  --extraction-backend paddle_glm5 `
  --results-dir results_runs\q2_grade_only_test `
  --rubric-dir results_rrd_vlm --force-rerun
```

完整实验后端：

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode FULL --questions Q2 `
  --extraction-backend paddle_glm5
```

旧提取器消融基线：

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode FULL --questions Q2 `
  --extraction-backend glm_vlm
```

`glm_vlm` 仍是默认值。当前 Q2 测试表明新后端的图形关系优于纯 OCR，
但五位屏蔽字识别仍弱，不能直接替换旧后端。
