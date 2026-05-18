import cv2
import numpy as np
import os
import glob

# ==========================================
# 核心配置：试卷题目固定坐标 (x, y, w, h)
# ==========================================
QUESTION_COORDS = {
    #"Q1": (78, 618, 951, 292),  
    #"Q2": (75, 907, 984, 546),  
    #"Q3": (1182, 141, 981, 694),  
    "Q4": (1182, 826, 969, 633),  
    #"Q5": (78, 1657, 963, 724),  
    #"Q6": (75, 2411, 981, 600),  
    #"Q7": (1170, 1750, 975, 586),  
}

valid_extensions = ('.jpg', '.jpeg', '.png')

# ==========================================
# 步骤一：仅裁剪（保留红笔，用于提取教师分数）
# ==========================================
def step1_batch_crop(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    if not image_files:
        print(f"❌ 步骤一失败：在 {input_dir} 下未找到图片！")
        return

    print(f"\n✂️ [步骤一] 开始裁剪 {len(image_files)} 份原始试卷 (保留红笔批注)...")

    for file_name in image_files:
        student_id = os.path.splitext(file_name)[0]
        image_path = os.path.join(input_dir, file_name)
        
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"   ⚠️ 跳过无法读取的图片: {file_name}")
            continue
            
        # 根据坐标进行空间切片并分类归档
        for q_id, (x, y, w_box, h_box) in QUESTION_COORDS.items():
            question_patch = original_img[y:y+h_box, x:x+w_box]
            
            q_folder = os.path.join(output_dir, q_id)
            os.makedirs(q_folder, exist_ok=True)
            
            output_path = os.path.join(q_folder, f"{student_id}_{q_id}.jpg")
            cv2.imwrite(output_path, question_patch)
            
        print(f"   ✅ 原卷 [{file_name}] 已成功裁剪并分发。")

    print(f"🎉 步骤一完成！带分数的切片已保存在: {output_dir}")


# ==========================================
# 步骤二：独立去敏（抹除红笔，用于模型客观盲评）
# ==========================================
def step2_batch_desensitize(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历 input_dir 下的所有子文件夹 (如 Q1, Q2) 和图片
    search_pattern = os.path.join(input_dir, "**", "*.*")
    image_paths = [p for p in glob.glob(search_pattern, recursive=True) if p.lower().endswith(valid_extensions)]
    
    if not image_paths:
        print(f"❌ 步骤二失败：在 {input_dir} 及其子文件夹下未找到图片！")
        return

    print(f"\n🧼 [步骤二] 开始对 {len(image_paths)} 张切片进行红笔去敏处理...")

    for img_path in image_paths:
        # 获取文件名和它所属的题目文件夹名 (例如: Q1/stu001_Q1.jpg)
        file_name = os.path.basename(img_path)
        q_id_folder = os.path.basename(os.path.dirname(img_path))
        
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue
            
        # --- 核心去敏算法 (HSV 提取红色并覆盖为白色) ---
        hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_img, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv_img, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.dilate(mask1 + mask2, np.ones((3, 3), np.uint8), iterations=1)
        
        desensitized_img = original_img.copy()
        desensitized_img[red_mask > 0] = [255, 255, 255]
        # ----------------------------------------------
        
        # 确保输出目录有对应的题号子文件夹
        out_q_folder = os.path.join(output_dir, q_id_folder)
        os.makedirs(out_q_folder, exist_ok=True)
        
        output_path = os.path.join(out_q_folder, file_name)
        cv2.imwrite(output_path, desensitized_img)

    print(f"🎉 步骤二完成！去敏后的纯净切片已保存在: {output_dir}")


# ==========================================
# 执行入口 (通过注释自由控制)
# ==========================================
if __name__ == "__main__":
    
    # 路径配置
    DIR_RAW_EXAMS = "./raw_exams"                      # 1. 你最初扫描的整页试卷原图
    DIR_CROPPED_WITH_SCORE = "./cropped_with_scores"   # 2. 裁剪后（带红笔分数）的图片存放地
    DIR_CROPPED_CLEANED = "./cleaned_patches"          # 3. 去敏后（纯净客观）的图片存放地

    # ---------------------------------------------------------
    # 【操作指南】
    # 需要跑哪一步，就取消哪一步的注释。
    # ---------------------------------------------------------

    # 🔴 第一步：将整卷裁剪成单题切片（保留老师分数）
    # 用于：你自己写脚本去提取老师的分数，存成 JSON/Excel。
    #step1_batch_crop(DIR_RAW_EXAMS, DIR_CROPPED_WITH_SCORE)

    # 🔵 第二步：读取带分数的切片，进行批量去红笔脱敏
    # 用于：提供给你的 VLM 模型进行 Stage 1 的盲提取事实。
    step2_batch_desensitize(DIR_CROPPED_WITH_SCORE, DIR_CROPPED_CLEANED)