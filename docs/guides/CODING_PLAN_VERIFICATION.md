# 智谱 Coding Plan 地址核验（2026-10-19 更新）

用户已确认：先前的账户余额扣费来自其他服务，与本次实验无关。
此前因计费疑虑设置的暂停要求已撤销，不再要求官方确认或账单证明。
当前要求仅为：所有智谱模型请求使用指定的 Coding Plan 地址。

## 固定地址

- Base URL：`https://open.bigmodel.cn/api/coding/paas/v4`
- Chat Completions 最终地址：`https://open.bigmodel.cn/api/coding/paas/v4/chat/completions`
- SDK 为 Base URL 添加末尾斜杠属于正常行为。
- 不允许普通 `/api/paas/v4` 地址、默认地址、第三方代理或地址回退。

## 已检查的入口

| 入口 | 结果 |
| --- | --- |
| 准则生成与优化 | Coding Plan 地址 |
| 文本评分 | Coding Plan 地址 |
| 三处视觉客户端 | Coding Plan 地址 |
| SAS-Bench 的 GLM 对比入口 | Coding Plan 地址 |
| 旧教师分数提取入口 | 已改为显式 Coding Plan 地址 |

共检查 7 处客户端构造，生成 8 次本地模拟请求。
旧教师分数提取入口不在本次夜间工作流中；本次未修改审核记录、评分算法或运行快照。

## 离线验证

在项目根目录执行：

```powershell
.\venv\Scripts\python.exe scripts\verify_coding_plan.py
```

工具使用假 Key 和本地 MockTransport，并禁止真实 socket 连接。
以下结果表示指定地址检查通过：

```json
{
  "offline_url_verification": "passed",
  "real_api_calls_made": 0,
  "real_credentials_used": false,
  "endpoint_requirement_satisfied": true,
  "billing_verification_required": false
}
```

此检查不连接在线模型，不核查账户账单，也不会启动或恢复实验。
它只覆盖上述入口；以后新增 API 客户端时，需要扩展检查范围。
实验启动与断点恢复仍遵循原有运行 ID、输入完整性及用户授权要求。
