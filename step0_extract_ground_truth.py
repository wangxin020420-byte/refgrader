import os
import json
import base64
import time
import re
from zhipuai import ZhipuAI

# ==================== 配置区 ====================
API_KEY = "be992a8955834b3ab91e708576da5089.7mYD9wLWTZh4AFY2"  
client = ZhipuAI(api_key=API_KEY)
VLM_MODEL_NAME = "glm-4.6v"

INPUT_DIR = "./cropped_with_scores"       # 步骤一裁剪出来的带分数的图片目录
OUTPUT_JSON = "./database/teacher_scores.json" # 提取结果的保存路径


# ==================== 工具函数 ====================
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_and_parse_json(text):
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match: text = match.group(1)
    else:
        text = text.strip().strip('`').strip()
        if text.startswith('json'): text = text[4:].strip()
    try: return json.loads(text)
    except: return None

# ==================== 核心提取逻辑 ====================
def extract_teacher_score_from_image(img_path):
    """召唤视觉大模型提取教师红笔打分"""
    prompt = """
    你是一个考卷数据录入员。这是一张学生答卷的切片图。
    图中包含了学生原本的书写内容，以及阅卷老师后来用【红笔】批改的痕迹和打分。
    
    你的任务：
    请仔细寻找图中的【红笔数字】（通常带有下划线、圈圈，或者写在题目旁边）。这代表老师给这道题的分数。
    
    🚨注意：
    1. 绝对不要把学生写的黑色数字或计算结果当成分数！
    2. 如果图中找不到任何明确的老师打分痕迹，请返回 -1。
    
    请严格输出纯 JSON 格式：
    {
        "teacher_score": 提取到的分数(数字格式)
    }
    """
    
    b64_img = encode_image_to_base64(img_path)
    content_list = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
    ]
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=VLM_MODEL_NAME, 
                messages=[{"role": "user", "content": content_list}], 
                temperature=0.1, 
                timeout=60
            )
            result = response.choices[0].message.content.strip()
            parsed = extract_and_parse_json(result)
            if parsed and "teacher_score" in parsed:
                return float(parsed["teacher_score"])
        except Exception as e:
            print(f"      ⏳ 提取受阻重试 ({attempt+1}/3): {e}")
            time.sleep(2)
    return -1.0

# ==================== 主控流程 ====================
def main():
    print("🚀 启动教师分数提取引擎...")
    
    # 初始化成绩单字典
    # 结构: {"stu001": {"Q1": 8.0, "Q2": 5.0}, "stu002": {...}}
    scores_db = {}
    
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            scores_db = json.load(f)
        print("⚡ 发现已有成绩单数据库，将进行增量提取。")

    # 遍历题库文件夹 (Q1, Q2...)
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入目录 {INPUT_DIR}")
        return

    for q_folder in os.listdir(INPUT_DIR):
        q_path = os.path.join(INPUT_DIR, q_folder)
        if not os.path.isdir(q_path): continue
        
        print(f"\n📂 正在扫描题目文件夹: {q_folder}")
        
        for file_name in os.listdir(q_path):
            if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            
            # 解析文件名 (假设格式为: E01914115_Q1.jpg)
            # 你可以根据自己的实际命名规则修改这里的拆分逻辑
            student_id = file_name.split('_')[0]
            q_id = q_folder 
            
            # 检查是否已经提取过
            if student_id in scores_db and q_id in scores_db[student_id] and scores_db[student_id][q_id] != -1:
                continue
                
            img_path = os.path.join(q_path, file_name)
            print(f"   👁️ 正在识别: {file_name}...")
            
            score = extract_teacher_score_from_image(img_path)
            print(f"      ✅ 提取结果: {score} 分")
            
            # 存入字典
            if student_id not in scores_db:
                scores_db[student_id] = {}
            scores_db[student_id][q_id] = score
            
            # 实时保存，防止中断
            os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(scores_db, f, indent=4, ensure_ascii=False)
            
            time.sleep(1) # VLM 频率保护

    print(f"\n🎉 全部提取完成！Ground Truth 成绩单已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()