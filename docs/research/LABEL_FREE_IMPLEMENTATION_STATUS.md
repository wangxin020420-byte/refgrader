# 无目标教师分改造：当前实现状态

状态：基础计算及评分路径解耦已实现；严格无标签评分入口、输入和配置溯源、审计与正式实验尚未完成。不要将当前结果表述为无教师分端到端验证，也不要据此启动实验。

## 已实现

- `scripts/run_csbench.py grade` 通过 `--sample-policy-root` 指定数据集目录，不再仅为定位策略而传递 `--teacher-db`。`main_pipeline.py` 未传该选项时保留旧的路径回退行为；审核策略本身仍可能读取教师分或筛选样本，因此旧流程不是无标签模式。
- `label_free_a3wa.py` 提供独立计算：等权风险隶属度、A3WA双阈值和三域判定、给定无标签隶属度及先验系数和网格下的信息损失搜索。拒绝缺失评分、非法风险、非法参数；输出不是真实评分正确概率。该模块尚未接到批改入口。
- 不更改教师分审核记录和既有实验，不训练或调用模型。
- `scripts/preflight_label_free.py` 对已准备数据做只读结构核验，并明确返回 `ready_for_live_scoring: false`；它不是评分入口，也不能证明文本与图片中不存在教师标记。
- `label_free_action.py` 为独立保守基线：POS接受基础分；BND和NEG保留暂存分并交人工。BND证据改分门禁与端到端评分尚未接入，也未用目标评分标签验证有效性。

## 离线检查

```
python -m unittest test_label_free_a3wa test_label_blind_grading test_csbench_grade_policy_paths test_sample_quality test_a3wa_theory
python -m unittest test_model_runtime test_coding_plan_verification
```

此前上述两组分别44项、17项通过。`scripts/run_csbench.py grade CO_1 --split validation --dry-run --no-artifacts` 只打印现有普通评分路径，不执行评分；它依然可能使用审核策略和旧优化准则，不可作为无标签运行。

## 必须完成的剩余事项

1. 独立的严格无标签模式：仅初始准则、raw样本、无教师分或有标签配置、明确禁止自动评估或回写；测试有效评分次数不足、随机标签置换/禁止读取以及输入快照和续跑隔离。
2. 审计元数据内容、原图教师批注、OCR和事实缓存；3411条元数据的字段名称中未见teacher/label/grade/score/review/human，但这不能证明正文或图像无标签。
3. 将独立A3WA计算接入严格模式，明确无标签参数来源、BND与NEG动作并测试，冻结配置与推理输出，再独立评估。不可用旧的教师分拟合参数冒充无标签参数。
4. 按两台电脑的实际目录核对数据、Git LFS、独立artifacts仓库及所有需要同步的运行产物；生成逐文件清单和校验，不声称普通Git已同步忽略内容。用户正在修改的审核JSONL不属于代码提交。
5. 在新实验前离线核对智谱文本和视觉最终地址为 `https://open.bigmodel.cn/api/coding/paas/v4`。不启动付费测试用于验证URL。
