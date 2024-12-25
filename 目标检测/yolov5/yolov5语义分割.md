# yolov5 的语义分割



> 为什么yolo可以做语义分割
>
> + yolo 本身具有FPN、PAN 结构
> + yolo可以接上任何形式输出（边框回归、图像输出、语义输出....）



## 标注图像分割数据集

+ labelme

通过labelme标注之后的数据有json文件

![image-20241121102729947](./assets/image-20241121102729947.png)

但是对于yolo训练的数据集来说，不需要json文件，其二，希望将信息改变格式



针对YOLO图像分割的数据集要求，处理数据：

<img src="./assets/image-20241121110844157.png" alt="image-20241121110844157" style="zoom:50%;" />

```python
import os
import json
import cv2

root = "./data/panels"
dist_root = "./data/yolo/panels"
dict = {"panel": 0}

for file in os.listdir(root):
    # 获取文件的名称
    filename = file.split(".")[0]
    # 获取图像文件
    if file.endswith("jpg"):
        # 复制文件到指定目录
        img = cv2.imread(os.path.join(root, file))
        cv2.imwrite(os.path.join(dist_root, "images", file), img)
    # 定义张图像输出的内容
    content = ""
    # 获取图像的标注文件
    if file.endswith("json"):
        with open(os.path.join(root, file)) as f:
            data = json.load(f)
            shapes = data['shapes']
            imageHeight = data['imageHeight']
            imageWidth = data['imageWidth']
            for shape in shapes:
                label = shape['label']
                content += f"{dict[label]}"
                points = shape['points']
                for x, y in points:
                    x = x / imageWidth
                    y = y / imageHeight
                    content += f" {x} {y}"
                content += "\n"
            with open(os.path.join(dist_root, "labels", f"{filename}.txt"), "w") as f:
                f.write(content)
        f.close()

```



关于yaml文件的设置参照：data/coco128-seg.yaml

```
# Ultralytics YOLOv3 🚀, AGPL-3.0 license
# COCO128-seg dataset https://www.kaggle.com/datasets/ultralytics/coco128 (first 128 images from COCO train2017) by Ultralytics
# Example usage: python train.py --data coco128.yaml
# parent
# ├── yolov5
# └── datasets
#     └── coco128-seg  ← downloads here (7 MB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128-seg # dataset root dir
train: images/train2017 # train images (relative to 'path') 128 images
val: images/train2017 # val images (relative to 'path') 128 images
test: # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```







## YOLO图像分割的格式要求



打开yolo项目中`data/coco128-seg.yaml`文件，选择最底部的链接，进行下载

```
https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip
```

对于训练的分割数据集主要为

![image-20241121103159539](./assets/image-20241121103159539.png)

然后images下，主要是训练使用图像数据集（images）

而labels下，是我们图像分割的目标数据集（labels）



关于labels下的数据格式：

+ labels 针对每一个图像，都有一个同名的txt文件

![image-20241121103341283](./assets/image-20241121103341283.png)

+ 文件中包含  区域类别数值，以及该区域中所定义图像分割区域的所有的两点坐标位置（归一化值）



## 训练语义分割模型

```
# Train YOLOv5s on COCO128 for 3 epochs
!python segment/train.py --img 640 --batch 16 --epochs 3 --data coco128-seg.yaml --weights yolov5s-seg.pt --cache
```

+ img  设置图像的大小（所有图像都要调整到一致大小）
+ batch  设置批次大小
+ epochs 训练次数
+ data  设置训练数据的配置yaml
+ weights 权重，预训练权重（如果是从头训练，则去掉weights）
+ cache 将训练过程进行缓存



官方权重下载链接

```
https://objects.githubusercontent.com/github-production-release-asset-2e65be/264818686/da150349-1509-4b6e-9756-456f4783831c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241121%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241121T031900Z&X-Amz-Expires=300&X-Amz-Signature=042da3d7a27d4b9869548efd37cbc7a2aadb726e34fe2669bdbcfdb9377e4a54&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dyolov5n-seg.pt&response-content-type=application%2Foctet-stream
```





相关的语义分割项目：

+ 自动驾驶汽车：自动驾驶汽车需要精确区分路面、行人、交通标志等元素，以确保行驶安全。语义分割技术可以实现对这些元素的实时识别和分割，为自动驾驶汽车提供重要的环境感知信息。
+ 医学影像分析：在医学影像领域，语义分割技术可以对组织器官进行精准分割，为医生提供详细的解剖结构和病理信息，辅助疾病诊断和手术规划。
+ 无人机航拍图像解析：无人机航拍图像通常包含复杂的场景和多个目标。语义分割技术可以对这些图像进行场景解析，识别出建筑物、道路、植被等不同的地物类别，为城市规划、环境监测等领域提供有力支持。
+ 工业缺陷检测：在工业制造过程中，语义分割技术可以用于检测产品表面的缺陷，如裂纹、划痕、污渍等。通过训练模型对正常产品和缺陷产品进行区分，可以实现自动化质量检测，提高生产效率和产品质量。
+ 人像分割：人像分割技术可以将图像中的人物与背景进行分离，为图像编辑、虚拟现实等应用提供便利。例如，在虚拟试衣间中，人像分割技术可以将用户与背景分离，然后将用户“穿”上虚拟服装，实现试衣效果。
+ 道路分割：道路分割是自动驾驶和交通监控中的重要应用。通过语义分割技术，可以将图像中的道路区域与其他区域进行分割，为车辆提供精确的行驶路径和障碍物信息。
+ 农作物分割：在农业领域，语义分割技术可以用于农作物监测和农业生产。通过对农田图像进行分割，可以识别出不同种类的农作物，监测其生长情况和病虫害情况，为农业生产提供决策支持。
+ 城市场景分割：城市街景图像中通常包含建筑物、车辆、行人等多个物体。语义分割技术可以对这些物体进行分割和识别，为智能交通、城市规划等领域提供重要的数据支持。
+ 安防监控：在安防监控领域，语义分割技术可以用于识别和分割监控视频中的异常事件和人员。例如，在公共场所的监控视频中，可以识别出携带危险物品的人员或异常行为，及时发出警报并采取相应的安全措施。
+ 虚拟现实与增强现实：在虚拟现实（VR）和增强现实（AR）应用中，语义分割技术可以用于识别和分割虚拟场景中的物体和人物。通过精确的分割和识别，可以实现更加真实的虚拟体验和交互效果。



