import torch
import torch.nn.init as init

from torch import nn

from 数字识别模型_resnet import NumberRecognition


class TextRecognition(nn.Module):
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        # 加载预训练模型
        self.nr = NumberRecognition()
        try:
            state_dict = torch.load('weights/model_resnet.pt', map_location=device)
            self.nr.load_state_dict(state_dict)
        except:
            print('未找到预训练模型')
        # 备份原来的输出层参数
        params = dict(self.nr.resnet.named_parameters())
        src_out_weight = params['fc.weight'].detach()
        src_out_bias = params['fc.bias'].detach()
        # print(src_out_weight.shape)
        # print(src_out_bias.shape)
        # print(src_out_bias)
        # 修改输出层
        self.nr.resnet.fc = nn.Linear(512, 36)
        params = dict(self.nr.resnet.named_parameters())
        # print(params['fc.weight'].shape)
        # print(params['fc.bias'].shape)

        # 初始化权重
        new_weight = init.xavier_uniform_(torch.zeros_like(params['fc.weight']))
        # print(new_weight.shape)
        # 将一部分已经训练的权重赋值给新权重
        new_weight[:10, :] = src_out_weight
        # print(new_weight)
        # 初始化函数不能在一维数据上初始化，这里稍微操作下
        new_bias = init.xavier_uniform_(torch.zeros_like(params['fc.bias']).unsqueeze(0)).squeeze(0)
        new_bias[:10] = src_out_bias
        # print(new_bias)
        # 修改新的输出层参数
        self.nr.resnet.fc.weight = nn.Parameter(new_weight, requires_grad=True)
        self.nr.resnet.fc.bias = nn.Parameter(new_bias, requires_grad=True)

    def forward(self, x):
        return self.nr(x)


if __name__ == '__main__':
    model = TextRecognition()
    x = torch.randn(5, 1, 224, 224)
    print(model(x).shape)
