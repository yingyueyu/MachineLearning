import numpy as np
from ultralytics import YOLO
import cv2
import sqlite3
from time import time

model = YOLO("../weights/yolo11n.pt")

# 设置黑色掩膜区域
image = cv2.imread("../data/test_img/test1.jpg")
results = model(image)

# 绘制需要监测的区域
part = np.array([[[55, 120], [16, 250], [323, 250], [256, 120]]])

# 创建并访问数据库
con = sqlite3.connect("cars.db")
# 创建表格
# cursorObj = con.cursor() # 获取可以执行下列语句的游标（缓存）
# cursorObj.execute("create table cars(id integer,start integer,end integer)")
# con.commit() # 让上述的执行语句生效

for result in results:
    boxes = result.boxes
    cls = result.boxes.cls
    conf = result.boxes.conf
    for i, (x1, y1, x2, y2) in enumerate(boxes.xyxy):
        if int(cls[i].item()) not in [2, 5]:
            continue
        obj_conf = conf[i].item()
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        reval = cv2.pointPolygonTest(part, [cx, cy], measureDist=False)
        if reval > 0:
            cv2.imshow("crop", image[y1:y2, x1:x2])
            # 写入记录
            # cursorObj.execute(f"insert into cars values({1},{int(time())},{0})")
            # con.commit()

            # 放大整个车辆图片 --> 图像分割获取车牌的分割区域 --> 监测车牌信息 --> 录入信息
            # sqlite 轻量级数据库
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
        cv2.circle(image, (cx, cy), 2, color=(0, 255, 0), thickness=1)
        cv2.putText(image, f"{obj_conf:.4f}", (x1, y1), 0, 0.6, color=(0, 255, 0), thickness=1)

cv2.polylines(image, part, isClosed=True, color=(0, 0, 255), thickness=2)
cv2.imshow("image", image)
cv2.waitKey(0)
