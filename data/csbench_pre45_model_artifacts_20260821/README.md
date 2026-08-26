# 43题模型配置归档

本目录保存45题数据快照切换前使用的43题模型侧活动文件：

- `rubrics/optimized`：旧优化评分细则；
- `rubrics/manifests`：旧评分细则来源与哈希；
- `rubrics/active_rubric_set.json`：旧活动评分细则集合；
- `calibration/active_a3wa_config.json`：旧A3WA校准配置。

这些文件的输入契约和哈希不再匹配当前45题快照，只用于历史追溯，不得复制回 `data/csbench` 作为当前配置。45题快照必须重新完成评分细则优化、验证和A3WA校准后再生成新的活动配置。
