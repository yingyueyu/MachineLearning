# onnx 跨平台部署

onnx （Open Neural Network Exchange） 开方神经网络交换，是由微软提出的一种模型数据格式，用于在不同深度学习框架之间进行模型转换和部署。

## pytorch 导出 onnx

导出 onnx 需要安装对应环境

```shell
pip install torch onnx
```

然后进行导出，参考 [《导出onnx.py》](./导出onnx.py)

## 导入 onnx

在其他设备环境中，安装 python，并安装 onnx 运行时

```shell
pip install onnxruntime
```

然后导入模型，创建会话并推理，参考 [《导入onnx.py》](./导入onnx.py)
