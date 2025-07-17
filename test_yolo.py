import cv2
from ultralytics import YOLO
import numpy as np

# 配置参数
MODEL_PATH = "model/best.pt"  # YOLO模型路径（可替换为你的自定义模型）
CAMERA_ID = 0  # 摄像头ID（0为默认摄像头，可改为视频文件路径）
CONF_THRESH = 0.5  # 置信度阈值（0-1）
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
COLOR_BGR = (0, 255, 0)  # 标注框颜色（BGR格式）

# 加载YOLO模型
model = YOLO(MODEL_PATH)

# 初始化摄像头
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise ValueError("无法打开摄像头或视频文件")

try:
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 目标检测
        results = model(frame, conf=CONF_THRESH, half=False, device="cpu", vid_stride=2)[0]  # 获取第一帧结果，使用半精度
        
        # 遍历检测结果
        for box in results.boxes:
            # 提取边界框坐标和置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = np.round(box.conf[0].item(), 2)
            cls_id = int(box.cls[0].item())
            label = f"{model.names[cls_id]} {conf}"
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BGR, THICKNESS)
            
            # 绘制标签文本
            (text_width, text_height), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
            cv2.rectangle(frame, (x1, y1 - text_height - 2), 
                        (x1 + text_width, y1), COLOR_BGR, -1)  # 填充文本背景
            cv2.putText(frame, label, (x1, y1 - 2), FONT, 
                       FONT_SCALE, (0, 0, 0), THICKNESS, cv2.LINE_AA)
        
        # 显示结果
        cv2.imshow("YOLO Object Detection", frame)
        
        # 按Q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()