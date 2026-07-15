# PaddleOCR 本地测试说明

## 1. 当前安装状态

项目已经建立独立 OCR 环境：

```text
.venv-ocr
```

当前版本：

```text
Python       3.11
PaddlePaddle 3.3.1 CPU
PaddleOCR    3.7.0
```

独立环境只负责视觉文字识别，不修改现有 `venv`，也不参与评分。
当前 Windows CPU 环境默认关闭 oneDNN/MKL-DNN，以规避
PaddlePaddle 3.3.x 的执行器兼容问题。

## 2. 单张图片识别

在项目根目录执行：

```powershell
.\scripts\run_paddle_ocr.ps1 `
  -InputPath cleaned_patches\Q5\E01914115_Q5.jpg `
  -OutputDir ocr_cache\q5_test
```

输出文件：

```text
ocr_cache/q5_test/E01914115_Q5.json
```

JSON中包含：

```text
识别文字
识别置信度
文字框坐标
检测多边形
图片 SHA-256
OCR/Paddle版本
```

默认使用 `0.5` 作为汇总文本的最低置信度。原始 token 仍会完整保存在
JSON 中，便于后续分析低置信度误识别。

## 3. 测试空白答卷

```powershell
.\scripts\run_paddle_ocr.ps1 `
  -InputPath cleaned_patches\Q5\E02014181_Q5.jpg `
  -OutputDir ocr_cache\q5_blank_test
```

注意：`ocr_empty_candidate=true` 只表示 PaddleOCR 没有识别到可靠文字，
不能单独作为“确认空白”的最终结论。后续还需要结合笔迹像素检测。

本地初步测试发现：空白答卷仍可能识别出印刷题号和低置信度噪声，
因此正式空白检测还必须剔除固定印刷区域，并结合图像笔迹特征。

## 4. 批量识别一个题目

```powershell
.\scripts\run_paddle_ocr.ps1 `
  -InputPath cleaned_patches\Q2 `
  -OutputDir ocr_cache\Q2
```

强制覆盖已有缓存：

```powershell
.\scripts\run_paddle_ocr.ps1 `
  -InputPath cleaned_patches\Q2 `
  -OutputDir ocr_cache\Q2 `
  -Force
```

## 5. 直接调用独立环境

不使用 PowerShell 包装脚本时：

```powershell
.\.venv-ocr\Scripts\python.exe ocr\paddle_ocr_worker.py `
  --input cleaned_patches\Q3 `
  --output-dir ocr_cache\Q3 `
  --device cpu
```

## 6. 正式流水线接入

已实现：

```text
原始 OCR JSON 与 SHA-256 缓存
confirmed_blank / confirmed_nonblank / uncertain 三态空白证据
GLM-5.1 脱敏事实映射
仅对 diagram rubric 条目调用 GLM-4.6V
OCR_ONLY / GRADE_ONLY / FULL
```

Q2 单样本提取：

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode OCR_ONLY --questions Q2 --student-ids E01914115 `
  --extraction-backend paddle_glm5 `
  --results-dir results_runs\q2_ocr_backend_test `
  --rubric-dir results_rrd_vlm
```

只读取事实缓存评分：

```powershell
.\venv\Scripts\python.exe main_pipeline.py `
  --mode GRADE_ONLY --questions Q2 --student-ids E01914115 `
  --extraction-backend paddle_glm5 `
  --results-dir results_runs\q2_grade_only_test `
  --rubric-dir results_rrd_vlm --force-rerun
```

当前边界：Q2 的顺序图关系已经能提取 `C→D→C`，但 PaddleOCR 对手写
五位屏蔽字仍有漏识别。因此 `glm_vlm` 仍是默认正式后端，
`paddle_glm5` 用于实验和消融对比。

## 7. 两类缓存及唯一模型目录

项目包含两类用途不同的缓存：

- PaddleX 模型缓存保存 `PP-OCRv6_medium_det`、`PP-OCRv6_medium_rec` 等下载模型。每台设备只使用一个权威目录；Windows 默认使用项目所在盘根目录的 `paddlex_cache`，Linux 默认使用 `~/.cache/refgrader/paddlex`。
- `ocr_cache/csbench` 保存逐图片 OCR 证据，包括图片 SHA-256、文本、坐标和置信度。它用于断点续传与审计，不能当作重复模型缓存删除。

主批改流程和直接执行 `paddle_ocr_worker.py` 都会在导入 PaddleOCR 前应用同一模型缓存规则。需要覆盖默认位置时设置：

```powershell
$env:PADDLE_PDX_CACHE_HOME = "D:\paddlex_cache"
```

Windows 安装脚本会创建该目录并写入用户环境变量：

```powershell
.\scripts\setup_paddle_ocr.ps1
```

worker 启动日志中的 `model_cache=...` 是本次实际使用的权威模型目录。确认新目录中的单图 OCR 测试成功后，可以人工删除旧的用户目录 `.paddlex`；程序不会自动递归删除可能被其他项目共用的目录。
