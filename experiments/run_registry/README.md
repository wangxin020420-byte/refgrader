# 实验登记表

| 身份 | Run ID | 当前可核实的状态 |
|---|---|---|
| PC1优化准则 | `reviewed45_glm52_20261019_221414_317274` | 标准冻结快照已建立并校验；experimental |
| PC2初始准则 | `initial85_glm52_20261020_pc2` | 待PC2导出及本机导入核验；不能用终端表替代完整快照 |
| 历史对照 | `csbench45_full_experimental_glm52_20260825_163432` | 旧数据2723份，不是严格配对消融 |

标准结果位置：`../refgrader-artifacts/csbench/frozen_baselines/<run_id>/`（相对代码仓库根目录）。
旧PC1专题档案：`reviewed85_glm52_optimized_pc1_20261021.json`，继续保留；标准快照与旧专题档案各自有来源。
两臂跟踪：`reviewed85_glm52_pair.json`。

对齐命令见[冻结与同步指南](../../docs/guides/FROZEN_BASELINES_GUIDE.md)。
索引只保存身份及验证状态；不要在此放大型运行结果。
