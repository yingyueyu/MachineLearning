import torch
from torch import nn

from 数字识别模型 import NumberRecognition


class TextRecognition(NumberRecognition):
    def __init__(self, device):
        super().__init__()
        # 加载预训练模型
        state_dict = torch.load('weights/model.pt', map_location=device)
        self.load_state_dict(state_dict)
        self.dict = dict(self.named_parameters())
        # print(self.dict.keys())
        # print(self.dict['fc3.weight'].shape)
        # print(self.dict['fc3.weight'])
        # print(self.dict['fc3.bias'].shape)
        # print(self.dict['fc3.bias'])

        # 扩展输出层的长度，数字 10 个 字母 26 个，共 36 个
        # 创建零张量，将预训练的数字分类的权重赋值上去
        out_weight = torch.zeros(36, 256)
        out_weight[:10, :] = self.dict['fc3.weight'].detach()
        out_bias = torch.zeros(36)
        out_bias[:10] = self.dict['fc3.bias'].detach()

        self.fc3 = nn.Linear(256, 36)
        self.fc3.weight = nn.Parameter(out_weight, requires_grad=True)
        self.fc3.bias = nn.Parameter(out_bias, requires_grad=True)

        # 冻结卷积部分参数
        for name, param in self.named_parameters():
            if name.startswith('conv') or name.startswith('bn'):
                param.requires_grad = False
            else:
                param.requires_grad = True


if __name__ == '__main__':
    model = TextRecognition(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(model.fc3.weight.requires_grad)
    print(model.conv1.weight.requires_grad)
    d = dict(model.named_parameters())
    print(d.keys())
    print(d['fc3.weight'])
    print(d['fc3.weight'].shape)
    print(d['fc3.weight'].requires_grad)
    print(d['fc3.weight'].data.requires_grad)

    x = torch.randn(5, 1, 28, 28)
    print(model(x).shape)
