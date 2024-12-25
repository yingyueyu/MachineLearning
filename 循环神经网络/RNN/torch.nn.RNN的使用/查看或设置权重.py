import torch
import torch.nn as nn

# 官方文档:
# Variables
# weight_ih_l[k] – the learnable input-hidden weights of the k-th layer, of shape (hidden_size, input_size) for k = 0. Otherwise, the shape is (hidden_size, num_directions * hidden_size)
# weight_hh_l[k] – the learnable hidden-hidden weights of the k-th layer, of shape (hidden_size, hidden_size)
# bias_ih_l[k] – the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
# bias_hh_l[k] – the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)
# 权重对应的就是更新隐藏状态的公式
# hidden_t = tanh(weight_{hidden}*hidden_{t-1} + weight_{input}*input_t)
# weight_ih_l[k] 对应的就是公式中的 weight_{input}
# weight_hh_l[k] 对应的就是公式中的 weight_{hidden}

model = nn.RNN(2, 10, 3)

# 方法一: 获取一个参数
state_dict = model.state_dict()
# 获取输入对隐藏线性变换的权重，也可以设置
w_ih_0 = state_dict['weight_ih_l0']
print(w_ih_0)
print(w_ih_0.shape)
w = torch.ones(10, 2)
# 设置权重
state_dict['weight_ih_l0'] = w
# 加载权重
model.load_state_dict(state_dict)

# 方法二: 迭代所有参数
# 获取模型参数名和参数值
named_parameters = model.named_parameters()
# name: 参数名
# data: 参数值
for name, data in named_parameters:
    print(name)
    print(data)
