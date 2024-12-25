from ultralytics import YOLO

# 加载官方的预训练权重
# 预训练权重是在COCO数据集下训练的，因此会有80个分类可以预测
model = YOLO("weights/yolov8n.pt")

# 预测一张图中的信息
# results 返回每张图中的信息
results = model(["test1.jpg","test1.jpg"])  # return a list of Results objects

# 遍历每一张图中信息
for result in results:
    # boxes 边框参数（boxes 中可以生成想要的任何格式的坐标，目标检测）
    boxes = result.boxes
    # masks 图像分割中掩码的单通道图
    masks = result.masks
    # keypoints 返回图像中的关键信息
    keypoints = result.keypoints
    # probs 分类预测
    probs = result.probs
    # obb 旋转角度的目标检测结果
    obb = result.obb
    result.show() # 展示图片（以当前电脑中的默认方式打开图片）
    result.save(filename="result.jpg")  # 保存结果图片