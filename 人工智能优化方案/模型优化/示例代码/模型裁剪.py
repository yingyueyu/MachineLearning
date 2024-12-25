from torch import nn
import torch.nn.utils.prune as prune


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 4)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


model = SimpleModel()

# print('裁剪前')
# print(model.fc1.weight.data)
#
# # 模型裁剪只能对指定的层进行裁剪
# for name, module in model.named_modules():
#     if isinstance(module, nn.Linear):
#         print(name)
#         # 调用裁剪 api
#         # 按比例裁剪
#         # prune.l1_unstructured(module, name='weight', amount=0.5)
#         # 按个数裁剪
#         prune.l1_unstructured(module, name='weight', amount=2)
#
# print('裁剪后')
# print(model.fc1.weight.data)
#
# print('裁剪后的掩码')
# print(model.fc1.weight_mask)
#
# for name, module in model.named_modules():
#     if isinstance(module, nn.Linear):
#         # 应用重参数
#         prune.remove(module, name='weight')
#
# print('应用重参数后')
# print(model.fc1.weight.data)
# # 应用重参数后，weight_mask 就被移除了，裁剪的效果将永久保存到模型中
# # print(model.fc1.weight_mask)


# 全局裁剪
# 官方笔记: 由于全局结构化剪枝没有多大意义，除非规范通过参数的大小进行标准化，所以我们现在将全局剪枝的范围限制为非结构化方法。
prune.global_unstructured(
    [(model.fc1, 'weight'), (model.fc2, 'weight')],
    pruning_method=prune.L1Unstructured,
    amount=0.5
)

print(model.fc1.weight)
print(model.fc2.weight)
