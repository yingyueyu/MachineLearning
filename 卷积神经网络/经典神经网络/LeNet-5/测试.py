# 训练好模型之后，用模拟真实场景的数据来测试模型，这个过程就是测试
import torch
import os
from PIL import Image
from LeNet5 import LeNet5
from torchvision.transforms import Compose, ToTensor, Grayscale

# 导入模型
# map_location='cpu': 转换模型格式到对应设备上
# weights_only=True: 添加模型的可信任度，告诉程序这是个安全的可信任的模型
state_dict = torch.load('weights/LeNet5_500.pt', map_location='cpu', weights_only=True)
model = LeNet5()
model.load_state_dict(state_dict)
model.eval()  # 开启评估模式

transform = Compose([
    Grayscale(),
    ToTensor()
])
# 加载数据
images = []
entries = os.scandir('./test_data')
for entry in entries:
    print(entry.path)
    # 打开图片
    image = Image.open(entry.path)
    # 张量转换
    image = transform(image)
    images.append(image)

# 将列表转换成张量
inputs = torch.stack(images)
print(inputs)
print(inputs.shape)

# 预测结果
# torch.inference_mode(): 开启推理模式，提高推理速度
with torch.inference_mode():
    y = model(inputs)
    print(y)
    print(y.shape)
    values, indices = torch.topk(y, k=1, dim=1)
    print(values)
    print(indices)
