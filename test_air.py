from zhipuai import ZhipuAI

# 填入你的真实 API KEY
API_KEY = "2ffa5de4683b4fcf90228c237d2f3fff.XY1MC10RtYaaoXni"
client = ZhipuAI(api_key=API_KEY)

print("🚀 正在单独测试 glm-4.5-air 连通性...")

try:
    response = client.chat.completions.create(
        model="glm-4.5-air",  # 确保全是小写
        messages=[
            {"role": "user", "content": "你好，请计算 1+1 等于几？直接输出数字。"}
        ],
        timeout=30
    )
    print("✅ 测试成功！模型回复:", response.choices[0].message.content)
except Exception as e:
    print("❌ 测试失败！抛出异常:", type(e).__name__)
    print("详细报错:", e)