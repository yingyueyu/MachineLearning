from ultralytics import YOLO

# YOLO("yolo8n.yaml") 训练的网络文件
# load("yolo8n.pt") 加载预训练权重（如果不使用load，就相当于从0开始训练）
# 下载 v8网络结构，需要访问：https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8
model = YOLO("yolov8n.pt")

# workers=0 指定训练时候的并发线程的数量，workers > 0 的时候可以在多GPU下使用
results = model.train(data="banana.yaml", epochs=100, imgsz=640, workers=0)
