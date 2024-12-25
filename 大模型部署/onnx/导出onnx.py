import torch
import torch.nn as nn


# 准备简单模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)
        self.fc2 = nn.Linear(10, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        return y1, y2


model = SimpleCNN()

# 准备输入校准数据
dummy_input = torch.randn(1, 1, 28, 28)  # 假设输入是28x28的单通道图像

# 导出模型为ONNX格式
torch.onnx.export(
    # 要导出的模型
    model,
    # 用于导出的虚拟输入
    dummy_input,
    # 输出文件的名称
    "simple_cnn.onnx",
    # 是否导出模型的参数
    export_params=True,
    # 要使用的ONNX操作集的版本
    opset_version=11,
    # 是否在导出时进行常量折叠优化
    do_constant_folding=True,
    # 输入节点的名称
    input_names=['input'],
    # 输出节点的名称
    # 模型若输出多个值，则此处定义获取不同值时的节点名
    output_names=['out1', 'out2']
)
print("模型已导出为ONNX格式")
# 导出成功后，您可以使用ONNX Runtime或其他支持ONNX格式的工具来加载和运行模型。
