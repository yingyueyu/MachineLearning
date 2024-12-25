### YOLOv8（2023）

（代码地址：https://github.com/ultralytics/ultralytics）

#### 模型介绍

YOLOv8 与YOLOv5出自同一个团队，是一款前沿、最先进（SOTA）的模型，基于先前 YOLOv5版本的成功，引入了新功能和改进，进一步提升性能和灵活性。

YOLOv8是一种尖端的、最先进的 (SOTA) 模型，它建立在以前成功的 YOLO 版本的基础上，并引入了新的功能和改进，以进一步提高性能和灵活性。YOLOv8 旨在快速、准确且易于使用，这也使其成为对象检测、图像分割和图像分类任务的绝佳选择。具体创新包括一个新的骨干网络、一个新的 Ancher-Free 检测头和一个新的损失函数，还支持YOLO以往版本，方便不同版本切换和性能对比。

YOLOv8 有 5 个不同模型大小的预训练模型：n、s、m、l 和 x。关注下面的参数个数和COCO mAP（准确率），可以看到准确率比YOLOv5有了很大的提升。特别是 l 和 x，它们是大模型尺寸，在减少参数数量的同时提高了精度。

![Ultralytics YOLOv8](./assets/yolov8-comparison-plots.avif)

#### 网络结构

![img](./assets/8a491cf03e0327a70812f111412235af.png)


整体结构上与YOLOv5类似： CSPDarknet（主干） + PAN-FPN（颈） + Decoupled-Head（输出头部），但是在各模块的细节上有一些改进，并且整体上是基于anchor-free的思想，这与yolov5也有着本质上的不同。

#### 改进部分

（1）输入端

与YOLOv5类似。

（2）主干网络

Backbone部分采用的结构为Darknet53，其中包括基本卷积单元Conv、实现局部特征和全局特征的Feature Map级别的融合的空间金字塔池化模块SPPF、增加网络的深度和感受野，提高特征提取能力的`C2F模块`。

（3）颈部网络

与YOLOv5类似。

（4）输出端

在损失函数计算方面，采用了Task AlignedAssigner正样本分配策略。由分类损失VFL（Varifocal Loss）和回归损失CIOU（Complete-IOU）+DFL（Deep Feature Loss）两部分的三个损失函数加权组合而成