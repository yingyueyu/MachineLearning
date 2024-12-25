import onnx
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
model_onnx = onnx.load("simple_cnn.onnx")

# 创建ONNX Runtime会话
session = ort.InferenceSession("simple_cnn.onnx")

# 准备输入数据
dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)

# 运行推理
# 第一个参数为 None 则返回模型的所有输出
# outputs = session.run(['out1'], {"input": dummy_input})
# outputs = session.run(['out1', 'out2'], {"input": dummy_input})
outputs = session.run(None, {"input": dummy_input})

print(outputs)  # 输出推理结果
