import cv2
import numpy as np

def process_exam_paper_high_res(image_path):
    # 1. 读取原始图像
    print("正在读取试卷图像...")
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError("找不到图片，请检查路径是否正确！")
    
    # 2. 视觉脱敏：去除红笔批改痕迹
    print("正在执行视觉脱敏...")
    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    
    # 红色 HSV 范围
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)
    
    desensitized_img = original_img.copy()
    desensitized_img[red_mask > 0] = [255, 255, 255]
    
    # [核心修改 1]：保存 1:1 原分辨率的脱敏图像
    cv2.imwrite("HighRes_Desensitized.jpg", desensitized_img)
    print("已保存高清脱敏全图: HighRes_Desensitized.jpg")
    
    # 3. 交互式空间切片
    print("\n>>> 请在弹出的窗口中，用鼠标框选出计组大题的完整解答区域！")
    print(">>> 框选好后，按 'SPACE' 或 'ENTER' 键确认。按 'c' 取消重新框。\n")
    
    cv2.namedWindow("Select Question Patch", cv2.WINDOW_NORMAL)
    h, w = desensitized_img.shape[:2]
    cv2.resizeWindow("Select Question Patch", int(w/3), int(h/3)) 
    
    x, y, w_box, h_box = cv2.selectROI("Select Question Patch", desensitized_img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Question Patch")
    
    # 按照鼠标框选的坐标进行精准裁切
    if w_box > 0 and h_box > 0:
        question_patch = desensitized_img[y:y+h_box, x:x+w_box]
        
        # [核心修改 2]：保存 1:1 原分辨率的局部切片图
        cv2.imwrite("HighRes_Spatial_Patch.jpg", question_patch)
        print(f"提取成功！已保存高清切片图: HighRes_Spatial_Patch.jpg")
        print(f"目标区域坐标: x={x}, y={y}, w={w_box}, h={h_box}")
    else:
        print("未检测到有效框选，已取消切片保存。")

# 运行函数
process_exam_paper_high_res("C124301068.jpg")