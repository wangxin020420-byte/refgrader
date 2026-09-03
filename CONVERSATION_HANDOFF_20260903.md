# RefGrader 新对话交接摘要

> 更新时间：2026-09-03
> 用法：在新对话开始时要求先阅读本文件、`PROJECT_STATUS.md`、S-GRADES 的
> `REPRODUCTION_STATUS.md`，再继续工作。不要仅凭旧聊天记忆重新推断路径或运行编号。

## 1. 用户当前目标

用户的科研方向是智慧教育中的计算机学科主观题自动批改。当前需要完成三条相互隔离但最终可比较的工作线：

1. 维护并验证私有计算机学科数据上的 RefGrader 三支决策自动评分流程。
2. 完成专利《一种基于非对称三支决策的计算机学科主观题自动评分方法及系统》初稿并交导师审阅。
3. 先完成公共论文基线复现，再使用 RefGrader 方法运行相同公共数据，形成论文对比实验。

科研表述必须客观。代码正确不等于效果提升；预实验、正式实验、隐藏测试和原论文同模型复现必须区分。

## 2. 目录和仓库边界

| 内容 | 路径 |
| --- | --- |
| RefGrader 主代码 | `C:\Users\wx\Desktop\refgrader-main` |
| 可移植实验结果 | `C:\Users\wx\Desktop\refgrader-artifacts` |
| S-GRADES 独立复现 | `C:\Users\wx\Desktop\refgrader-reproductions\sgrades` |
| S-GRADES 固定原始数据 | `C:\Users\wx\Desktop\refgrader-public-datasets\sgrades\raw_hf_pinned_20260901` |
| S-GRADES 规范化数据 | `C:\Users\wx\Desktop\refgrader-public-datasets\sgrades\prepared_protocol_20260901` |
| 专利当前桌面稿 | `C:\Users\wx\Desktop\一种基于非对称三支决策的计算机学科主观题自动评分方法及系统-草稿.docx` |

`refgrader-main` 保存主方法和活动配置；`refgrader-artifacts` 保存结果；S-GRADES 官方仓库及复现控制脚本位于独立复现目录。不要把公共数据结果写入私有数据活动目录，不要修改固定的官方仓库提交。

## 3. RefGrader 主方法

当前流程：题目/数据契约 -> 评分准则 -> Stage 1 事实与视觉证据提取 -> Stage 2 三次独立语义评分 -> 风险建模 -> A3WA 的 POS/BND/NEG 路由 -> BND 证据仲裁 -> 3WD-Core -> 可选残差校正 -> 最终分数与评估。

默认模型合同：文本请求别名 `glm-5.2`、服务端实际返回 GLM-5.3、`thinking=disabled`；视觉模型 `glm-4.6v`。Coding Plan 地址为 `https://open.bigmodel.cn/api/coding/paas/v4/`。密钥已经配置为 Windows 用户级 `ZHIPUAI_API_KEY`；旧终端若未继承，需从 User 环境加载到当前进程。

已修复过的关键问题包括：评分阶段教师标签隔离、`real_diff` 未定义、Windows 并发写 `progress.json`、编码乱码、严格准则优化失败回滚、Coding Plan URL/密钥/模型合同和断点续跑兼容。不得重新引入教师分数进入正式评分提示。

## 4. 私有数据最新结果

私有数据包含 45 道计算机学科题，答案级正式测试集为 2,723 份。最新正式结果：

| 方法 | N | MAE | RMSE | Pearson | 误差不超过2分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 多探针平均分 | 2,723 | 1.4003 | 2.3362 | 0.9157 | 77.34% |
| 3WD-Core | 2,723 | 1.3481 | 2.2787 | 0.9174 | 78.11% |

当前只能说三支决策核心分有小幅正向变化。后续论文还需要分学科结果、配对 bootstrap/显著性分析、POS/BND/NEG 分域结果、风险分量消融、边界仲裁消融和教师噪声敏感性分析。

## 5. 专利状态

专利正式初稿已完成，内容以最终系统方法为主，不写中间调试、未来计划、教师标签审核工作流或用于防止用户误解的说明。文档顺序为：说明书、说明书附图、权利要求书、说明书摘要、摘要附图。当前还需插入 6 张最终附图，核对图号和说明书引用，并由导师或专利代理人审查权利要求范围。

## 6. SAS-Bench 状态

已完成 4,093 条有效记录的 RefGrader 整体评分。3WD 的 MAE 为 `2.0794`、RMSE 为 `3.2971`、Pearson 为 `0.8265`；12任务宏平均QWK为 `76.71%`，答案数加权QWK为 `77.02%`。官方 CCS/ECS 需要分步得分和错因，整体分完成不代表官方协议完成。最后有证据的状态仍记录在 `SAS_BENCH_REPRODUCTION_PROGRESS.md`，新对话不得擅自写成已完成。

## 7. S-GRADES 复现性质

论文官方模型为 GPT-4o-mini、Gemini-2.5-Flash 和 LLaMA-4-Scout。用户决定统一使用 GLM，因此当前工作应称为“在 GLM-5.3 上复用 S-GRADES 协议的模型替换/适配复现”，不能称为原论文数值的严格同模型复现。

作者公开结果中的 AES/ASAG 稳定性比值已复算：GPT-4o-mini `1.59x`、Gemini `1.65x`、LLaMA `1.96x`，与论文四舍五入口径一致。GLM 请求使用 `glm-5.2` 别名，服务端返回 `glm-5.3`。GLM 会把 `reasoning_content` 计入输出预算，因此有效最小 completion 上限改为256；该兼容差异已写入运行清单。

## 8. S-GRADES 数据审计

- 14 个固定源仓库展开为 23 个评估单元。
- 固定版本实际测试总数为 31,739，而上游 README 写 35,873。
- 当前 SciEntSBank 两个配置各 5,835 条；README 写 4,969，但 `test_sizes.json` 与固定文件均为 5,835。
- BEEtlE 测试CSV有两行答案中的逗号未转义，导致后半句移入 label 列。规范化脚本在派生副本中合并回答案，并清空全部测试标签；原始固定文件未修改。
- 公共 `D_` 测试集不提供真实标签。完整预测后必须通过 S-GRADES 官方平台取得隐藏测试指标。

关键清单：

- `C:\Users\wx\Desktop\refgrader-artifacts\public_benchmarks\sgrades\reproduction\protocol_freeze_20260901\protocol_manifest.json`
- `C:\Users\wx\Desktop\refgrader-artifacts\public_benchmarks\sgrades\reproduction\protocol_freeze_20260901\protocol_units.csv`
- `C:\Users\wx\Desktop\refgrader-public-datasets\sgrades\prepared_protocol_20260901\prepared_protocol_manifest.json`

## 9. S-GRADES 已完成实验

1. Coding Plan 连接测试通过，服务端返回 GLM-5.3。
2. 代表性演绎路径 5/5 有效。
3. 六策略各5条全部有效。
4. BEEtlE 100条带标签训练留出预实验完成。归纳策略最佳：Accuracy `0.5100`、Macro-F1 `0.4446`。这不是官方隐藏测试结果。
5. 23单元×6策略的138组合真实API冒烟全部通过，0异常。
6. R5正式矩阵的13个小单元全部完成：78/78组合、9,462/9,462条有效预测、0无效。

R5根目录：

`C:\Users\wx\Desktop\refgrader-artifacts\public_benchmarks\sgrades\reproduction\r5_glm53_formal_matrix`

## 10. S-GRADES 正在运行

剩余10个大单元合计30,162条/策略，六策略共180,972条。已按原始行序每200条分片，共942个任务。状态快照：`D_ASAP_plus_plus / inductive / shard_00000` 已完成200/200，`shard_00001`正在运行。实际状态会继续变化，必须读取所有 `run_manifest.json`，不能只依赖本段文字。

活动进程使用：

- `run_glm_protocol_unit.py`
- `--shard-size 200`
- `--seed 42`
- `--api-model glm-5.2`
- `--model-label glm-5.3`
- `--resume`

恢复时必须继续使用同一个 R5 根目录。不要改模型、种子、分片大小、prepared manifest 或输出目录。当前分片中断时，最多重跑当前200条；已完成分片会跳过。

## 11. R5 完成门禁

1. 大单元必须有 942 个分片清单。
2. 大单元有效预测总数必须为 180,972，invalid 为0。
3. 使用 `merge_protocol_shards.py` 合并后，应得到60个大单元策略组合并通过ID无重复、无缺失和源顺序校验。
4. 加上已完成的78个小单元组合，完整矩阵为138个组合、190,434条预测。
5. 将预测提交官方平台并保存返回指标。
6. 生成23单元明细、六策略宏平均、分类/回归分组和失败率表。

满足以上条件后，才能说“GLM-5.3上的S-GRADES协议适配复现完成”。只跑完CSV但没有官方指标，不算完整闭环。

## 12. 后续顺序

1. 让R5分片队列继续运行，定期审计并使用相同命令续传。
2. 完成大单元合并和完整矩阵索引。
3. 通过官方平台取得隐藏测试指标。
4. 再建立独立的 `S-GRADES -> RefGrader` 适配层。
5. RefGrader必须使用同一数据版本、样本ID和GLM骨干；结果写入独立目录，不复用S-GRADES基线checkpoint。
6. 最后比较官方六策略基线与RefGrader 3WD，并增加Average、3WD-Core、去风险分量、去BND仲裁等消融。

## 13. 新对话首条建议

可以在新对话中直接发送：

> 请先阅读 `C:\Users\wx\Desktop\refgrader-main\CONVERSATION_HANDOFF_20260903.md`、`PROJECT_STATUS.md` 和 `C:\Users\wx\Desktop\refgrader-reproductions\sgrades\reproduction-control\REPRODUCTION_STATUS.md`。先检查当前R5分片运行状态，只基于最新manifest给出续跑或合并操作；不要重新下载数据、不要重跑已完成的R0-R5小单元，也不要开始RefGrader公共数据实验，除非GLM协议复现已经通过完整门禁。
