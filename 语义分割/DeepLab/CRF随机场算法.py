"""
CRF 随机场算法的包
pip install torchcrf (pydensecrf 已经失效了)
"""
"""
torchcrf 案例
"""
import torch
from torchcrf import CRF

# 像素分类的类别数量
num_tags = 5
# 获取CRF的模型
model = CRF(num_tags)

# 输入（20,3,224,224） 输出 (20,2,224,224)
# 按照上述的训练方式定义hid_out = 输出图.reshape(20, 224x224, 2)
# torch.randint(batch_size, w * h, 类别数量)
hid_out = torch.randint(8, 100, 5)
# y_tag 就是实际参与训练的标注图（此处二分类是一个黑白图）
# labels的图像 (20,224,224)
# y_tag = 标注图.reshape(20,224,224)
y_tag = torch.randint(5, size=(8, 100))

# 两张图的损失
loss = model(hid_out, y_tag)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = model(hid_out, y_tag)
    loss.backward()
    optimizer.step()

