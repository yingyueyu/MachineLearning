import peft
import torch
import copy

from peft import LoraModel
from torch import nn
import torch.nn.functional as F

use_peft = True

torch.manual_seed(0)

# 数据集
X = torch.rand((1000, 20))
y = (X.sum(1) > 10).long()

print(y)
print(y.shape)

# 超参数
n_train = 800
batch_size = 64

# 加载器
train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X[:n_train], y[:n_train]),
    batch_size=batch_size,
    shuffle=True,
)
eval_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X[n_train:], y[n_train:]),
    batch_size=batch_size,
)


# 模型
class MLP(nn.Module):
    def __init__(self, num_units_hidden=2000):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(20, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        return self.seq(X)


n_train = 800
lr = 0.002
batch_size = 64
max_epochs = 30
device = 'cpu' if not torch.cuda.is_available() else 'cuda'


# 训练
def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, (xb, yb) in enumerate(train_dataloader):
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        for step, (xb, yb) in enumerate(eval_dataloader):
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                outputs = model(xb)
            loss = criterion(outputs, yb)
            eval_loss += loss.detach().float()

        eval_loss = (eval_loss / len(eval_dataloader)).item()
        train_loss = (total_loss / len(train_dataloader)).item()
        print(f"{epoch=:<2}: {train_loss=:.4f}  {eval_loss=:.4f}")


# 无 peft 训练
if use_peft is False:
    print('无 peft 训练')
    module = MLP().to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # 最后三周期的训练结果
    # epoch = 27: train_loss = 0.0010 eval_loss = 0.2583
    # epoch = 28: train_loss = 0.0009 eval_loss = 0.2842
    # epoch = 29: train_loss = 0.0008 eval_loss = 0.2634
    train(module, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)

# 使用 peft 训练
else:
    print('使用 peft 训练')

    # 查看模块名称
    # 选择层 seq.0 和 seq.2 用于 LoRA 微调
    # *注意：并非所有层类型都可以使用 LoRA 进行微调。目前，支持:
    # linear layers, embeddings, Conv2D and transformers.pytorch_utils.Conv1D
    print([(n, type(m)) for n, m in MLP().named_modules()])

    # 添加 LoRA 微调配置
    config = peft.LoraConfig(
        r=8,
        target_modules=["seq.0", "seq.2"],
        modules_to_save=["seq.4"],
    )

    # 创建模型
    module = MLP().to(device)
    module_copy = copy.deepcopy(module)  # 留一个原始模型，备用
    # 包装一个 peft_model
    peft_model = peft.get_peft_model(module, config)
    # peft_model = LoraModel(module, config, 'lora')
    optimizer = torch.optim.Adam(peft_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # 查看可训练参数量
    peft_model.print_trainable_parameters()
    # 最后三周期的训练结果
    # epoch = 27: train_loss = 0.0164 eval_loss = 0.4015
    # epoch = 28: train_loss = 0.0058 eval_loss = 0.3373
    # epoch = 29: train_loss = 0.0012 eval_loss = 0.3502
    train(peft_model, optimizer, criterion, train_dataloader, eval_dataloader, epochs=max_epochs)

    # 检查更新了哪些权重，哪些权重没有更新
    for name, param in peft_model.base_model.named_parameters():
        if "lora" not in name:
            print(f'not LoRA: {name}')
            continue
        print(f'is LoRA: {name}')
        print(f"New parameter {name:<13} | {param.numel():>5} parameters | updated")

    params_before = dict(module_copy.named_parameters())
    for name, param in peft_model.base_model.named_parameters():
        print(name)
        continue

        if "lora" in name:
            continue

        # 获取原始模型中的参数名
        name_before = name.partition(".")[-1].replace("original_", "").replace("module.", "").replace(
            "modules_to_save.default.", "").replace('base_layer.', '')
        # 获取原始模型中的参数
        param_before = params_before[name_before]

        # 比较 peft 模型和原始模型，是否近似相等
        if torch.allclose(param, param_before):
            # 近似相当的当作没有更新的参数
            print(f"Parameter {name_before:<13} | {param.numel():>7} parameters | not updated")
        else:
            # 否则当作被更新的参数
            print(f"Parameter {name_before:<13} | {param.numel():>7} parameters | updated")

        # 结果如下:
        # Parameter seq.0.weight | 40000 parameters   | not updated
        # Parameter seq.0.bias   | 2000 parameters    | not updated
        # Parameter seq.2.weight | 4000000 parameters | not updated
        # Parameter seq.2.bias   | 2000 parameters    | not updated
        # Parameter seq.4.weight | 4000 parameters    | not updated
        # Parameter seq.4.bias   | 2 parameters       | not updated
        # Parameter seq.4.weight | 4000 parameters    | updated
        # Parameter seq.4.bias   | 2 parameters       | updated

        # 可以看到 LoRA 只更新了最后一层的参数，这些内容在 LoRA 的配置中被规定

# 结论:
# 可以看到不使用 peft 和使用 peft 的情况下训练效果相当，但是使用 peft 时训练的参数仅原来的 1% 左右
