# RefGrader 在 SCALE 四个公开数据集上的主实验

本包运行本项目 RefGrader：参考答案重建初始评分规则 → 训练侧规则校准样本的三次试评与方差优化 → 验证集评分与 A3WA 校准 → 测试集最终评分。**不运行 SCALE 的候选生成、代理标注器或 SAE，也不把已有 SCALE ADS 模型当成本方法结果。**

## 固定实验设置

- 文本模型 glm-5.3，Coding Plan 接口，thinking=disabled。文本规则生成也使用 glm-5.3，不调用视觉模型。保留现有评分、规则语义校验、三支决策与校准算法。
- GLM-5.3 独立 profile；原项目默认 glm-5.2 不变。仅新 profile 将学科限定提示改为“当前题目所属学科”，支持物流与健康信息学。
- 公开数据固定版本：aleversn/SAS-Datasets，commit `9e1a866d58edaa0d78af374d2f3529a10d72239a`。原文数据仓库：https://github.com/aleversn/SAS-Datasets 。逐文件验证 Git blob 哈希，数据已随包携带。
- 上游训练/测试边界不变。从每个数据集上游测试侧按 Python Random(42) 统一抽 50 条验证，其余正式测试。ADS 与此前 SCALE v2 的行序划分规则相同；双方样本 ID 命名不同。
- 按“题目＋有序参考答案”分组，每组训练侧最多 5 条作为规则校准样本；其余训练记录保留但不参与参数训练。RefGrader 是 API 评分方法，不需要训练 BERT/SAE。不能把训练池行数称为全部被模型训练使用。
- A3WA 参数及分数校准只使用 50 条验证记录；不依据测试结果调参。每轮独立生成/优化规则、校准、评分，共 5 次 API 重复，不声称 API 提供可复现 seed。
- 只报告最终 `final_calibrated_score` 的主实验，不单独运行消融。R、RMSE 基于连续预测；QWK 将预测与真值裁剪到 0–10 后 half-up 取整，固定 11 档。该 QWK 口径是冻结实现选择，未声称与作者一致。
- 全部任务线性映射到 0–10 供本方法使用：ADS 原值；LE/SR ×10；ASAG ×2。保存原始分数。同时报告原始 0–10 RMSE 与 RMSE÷10，禁止只因数值接近认定原文归一化口径相同。

| 数据集 | 上游训练池 | 其中规则校准 | 验证 | 正式测试 | 公开总数 |
|---|---:|---:|---:|---:|---:|
| ADS | 1,107 | 75 | 50 | 425 | 1,582 |
| LE | 409 | 394 | 50 | 126 | 585 |
| ASAG | 440 | 105 | 50 | 139 | 629 |
| SR | 231 | 144 | 50 | 50 | 331 |

公开文件中 ASAG/SR 是中文译版，不是英语/意大利语原版。ASAG 该版本只有 629 条，不补造论文介绍中缺少的一条。SR 不提供完整题干，使用公开主题和参考答案，并保留“原始题干未提供”标记。参考答案不等于官方细粒度评分细则，本包将初始规则明确标为重建规则。有训练侧上下文的组才做方差优化，无训练样本组保留无标签的重建规则并记入日志，不借用测试答案优化。空白答案不删除，模型输入使用明确空白标记并保留原始空字符串。跨划分重复文本只审计，不删样本。

## 隔离部署

完整包解压到 **`/data0/E125221219/projects/refgrader-scale4`**。
不要解压到 `/data0/E125221219/projects/refgrader-reproductions/scale`，不覆盖旧项目，不复用其 env/scale-bert（Python 3.8）。

本方法依赖 GLM API，不需要 CUDA/PyTorch/PaddleOCR。setup 使用独立 Python 3.11–3.13 venv；机器没有兼容解释器时尝试 conda 创建本目录 `.runtime-python`，不会升级旧环境。无 Python/conda 时明确停止，需管理员提供 Python 3.11。

把 `refgrader-scale4.zip` 上传到服务器 `/data0/E125221219/incoming/` 后，在服务器终端执行：

```bash
cd /data0/E125221219/projects
test ! -e refgrader-scale4 || { echo '目标已存在；续跑时跳过解压，不要覆盖。'; exit 1; }
unzip /data0/E125221219/incoming/refgrader-scale4.zip -d /data0/E125221219/projects
cd /data0/E125221219/projects/refgrader-scale4
bash scripts/benchmarks/setup_scale4.sh
tmux new -s refgrader-scale4
```

然后在新 tmux 会话内执行：

```bash
cd /data0/E125221219/projects/refgrader-scale4
bash scripts/benchmarks/launch_scale4.sh
```

按提示输入 API key（隐藏输入），不写入脚本、不粘贴到聊天。整个四数据集五次重复将依次运行。GLM 请求超时上限 900 秒、最多 12 次传输尝试、SDK 隐式重试关闭；外层每个未完成流程最多续跑 3 次。失败后保留进度，继续尝试其他独立重复，最后汇总标为不完整而不是虚报完成。

Ctrl+B 然后 D 可退出 tmux 界面而保持运行。重新连接：

```bash
tmux attach -t refgrader-scale4
```

## 续跑和查看结果

使用**同一条 launch 命令**续跑。成功规则和已完成校准/评分阶段会校验后跳过；不传 force，不删缓存，不改 tag。文件锁防止同一 tag 并发执行。模型、代码、提示词、数据哈希变化时拒绝混入原 run。

```bash
cd /data0/E125221219/projects/refgrader-scale4
bash scripts/benchmarks/launch_scale4.sh
```

汇总文件（每个数据集五次的均值、样本标准差、有效轮数）：

```bash
cat /data0/E125221219/projects/refgrader-scale4/results_runs/scale4_refgrader/refgrader_glm53_scale4_v1/summary.json
```

若要仅重建汇总，不调用 API：

```bash
cd /data0/E125221219/projects/refgrader-scale4
.venv/bin/python scripts/benchmarks/run_scale4.py summarize --tag refgrader_glm53_scale4_v1 --repeats 5
```

结果保留目录：`results_runs/scale4_refgrader/`（协议、规则、校准和指标）；`results_runs/public_benchmarks/`（逐样本评分检查点）；`ocr_cache/public_benchmarks/`（文本事实缓存，目录名不表示调用 OCR）；`reports/` 和 `logs/`（日志）。备份这些目录及 BUNDLE_SHA256.json。

完成以 summary.json 的 `complete: true` 且四个数据集 `completed_runs: 5` 为准。A3WA 验证样本不足导致的 experimental 标记会保留，不自动终止剩余实验；数据完整性或输入混用则必须停止该轮。

## 检查边界

本地执行了真实数据解析、划分/标签隔离检查、参数预演和相关回归测试；未连接实验室服务器、未调用用户 GLM API，因而不能声称已完成真实 API 实验或保证服务端永不超时。当前是可续跑的部署方案，正式数值以服务器完整输出为准。
