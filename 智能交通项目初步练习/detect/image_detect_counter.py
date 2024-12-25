import numpy as np
from ultralytics import YOLO
import cv2
from collections import defaultdict

model = YOLO("../weights/yolo11n.pt")

cap = cv2.VideoCapture("../data/output_video.mp4")
# 绘制需要监测的区域
part = np.array([[[373, 380], [337, 446], [867, 446], [784, 380]]])
# 追踪的历史记录
track_history = defaultdict(lambda: [])

total_count = 0
# 记录经过指定区域的总量
records = []
while cap.isOpened():
    status, frame = cap.read()
    # 按帧处理图像：
    results = model.track(source=frame, persist=True)
    for result in results:
        boxes = result.boxes
        cls = result.boxes.cls
        conf = result.boxes.conf
        # 获取当前所有目标的id
        track_ids = result.boxes.id.int().cpu().tolist()
        for i, (x1, y1, x2, y2) in enumerate(boxes.xyxy):
            track_id = track_ids[i]
            if int(cls[i].item()) not in [2, 5]:
                continue
            obj_conf = conf[i].item()
            x1 = int(x1.item())
            y1 = int(y1.item())
            x2 = int(x2.item())
            y2 = int(y2.item())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            reval = cv2.pointPolygonTest(part, [cx, cy], measureDist=False)
            if reval > 0 and track_id not in records:
                records.append(track_id)

            # 将进入的目标id进行销毁
            track = track_history[track_id]
            track.append((float(cx), float(cy)))
            if len(track) > 30:
                track.pop(0)

            total_count = len(records)
            color_bbox = (0, 255, 0) if reval < 0 else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color_bbox, thickness=1)
            cv2.circle(frame, (cx, cy), 2, color=color_bbox, thickness=1)
            cv2.putText(frame, f"id:{track_id}", (x1, y1), 0, 0.6, color=color_bbox, thickness=1)
    if status is False:
        break
    cv2.putText(frame, f"total_count:{total_count}", (10, 60), 0, 1, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame, part, isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imshow("frame", frame)
    # 原视频是30帧/秒（fps），因此此处需要设置延时时间（1000 / 30）
    if cv2.waitKey(33) == ord(" "):
        break
