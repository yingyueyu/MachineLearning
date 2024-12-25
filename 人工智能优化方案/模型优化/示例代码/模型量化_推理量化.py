import torch
import torch.nn as nn
import torch.quantization


# 定义一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 模型包含两个模块
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


# 初始化模型
model = SimpleModel()
model.eval()

# 融合线性层和ReLU层
# 融合结果成为一个新的层
fused_model = torch.quantization.fuse_modules(model, ['linear', 'relu'])

# 检查融合后的模型
# SimpleModel(
#   (linear): Linear(in_features=10, out_features=5, bias=True)
#   (relu): ReLU()
# )
print(model)

# SimpleModel(
#   (linear): LinearReLU(
#     (0): Linear(in_features=10, out_features=5, bias=True)
#     (1): ReLU()
#   )
#   (relu): Identity()
# )
# 可以看到融合后 relu 变成了 Identity 恒等映射，那就说明融合后的 relu 实际上没有做任何操作
print(fused_model)

# 准备校准数据
calibration_data = torch.randn(10, 10)

# 定义量化配置
quantization_config = torch.quantization.get_default_qconfig('fbgemm')

# 为模型设置量化配置
fused_model.qconfig = quantization_config

# 准备模型进行量化
torch.quantization.prepare(fused_model, inplace=True)

# 校准模型
fused_model(calibration_data)

# 对模型进行量化
torch.quantization.convert(fused_model, inplace=True)
# 量化时会收到警告：
# D:\Program Files\anaconda3\envs\pytorch-env\lib\site-packages\torch\ao\quantization\observer.py:220: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
#   warnings.warn(
# 这是因为 api 参数reduce_range改动导致的，我们可以将reduce_range参数替换为quant_min和quant_max。这些参数允许您更明确地指定观察器的量化范围。
# 为了解决警告问题，需要使用观察器
# from torch.ao.quantization import MinMaxObserver
# 创建观察器，指定量化的最小和最大值
# observer = MinMaxObserver(quant_min=-128, quant_max=127)
# 注册观察器到模型中的某个层
# model.linear.register_forward_pre_hook(observer)



# 检查量化模型
# SimpleModel(
#   (linear): QuantizedLinearReLU(in_features=10, out_features=5, scale=0.013005306012928486, zero_point=0, qscheme=torch.per_channel_affine)
#   (relu): Identity()
# )
# 此处的 linear 已经是量化后的 QuantizedLinearReLU
print(fused_model)
