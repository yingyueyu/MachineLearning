# YOLOv6



> yolov6的U版结构有问题，实际使用，需要使用美团源代码



## 下载

```
git clone https://github.com/meituan/YOLOv6
```

或者直接下载源码文件



## 下载预训练权重

> 由于yolov6 的代码更新到0.4.0，所以权重需要使用0.4.0权重

下载地址：[Release YOLOv6 4.0 · meituan/YOLOv6 · GitHub](https://github.com/meituan/YOLOv6/releases/tag/0.4.0)



## 运行

按照官网的执行流程执行预测代码

```
python tools/infer.py --weights ./yolov6s.pt --source imgs
```

