import cv2
import numpy as np
from openvino.runtime import Core

# --------- 参数配置 ---------
MODEL_PATH = "model/best_openvino_model/best.xml"  # 替换为你的模型路径
DEVICE = "CPU"
INPUT_SIZE = (640, 640)
CONF_THRESH = 0.3

# --------- 初始化 OpenVINO 模型 ---------
core = Core()
model = core.read_model(model=MODEL_PATH)
compiled_model = core.compile_model(model=model, device_name=DEVICE)
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# --------- 摄像头初始化 ---------
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

# --------- 实时循环 ---------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ------ 图像预处理 ------
    img = cv2.resize(frame, INPUT_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_transposed = img_rgb.transpose(2, 0, 1)  # HWC → CHW
    img_input = np.expand_dims(img_transposed, axis=0).astype(np.float32) / 255.0

    # ------ 模型推理 ------
    outputs = compiled_model([img_input])[output_layer][0]  # shape: (8400, 12) for example

    # ------ 解析输出 ------
    for det in outputs:
        x1, y1, x2, y2, conf, *cls_conf = det
        class_id = int(np.argmax(cls_conf))
        score = conf * cls_conf[class_id]
        if score > CONF_THRESH:
            # 坐标缩放回原始尺寸（如果模型预处理有变化需同步调整）
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{class_id} {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

    # ------ 显示画面 ------
    cv2.imshow("OpenVINO YOLOv11 Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
