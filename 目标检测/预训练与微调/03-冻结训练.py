import torch
import torchvision
import numpy as np

model = torchvision.models.resnet18()
model_dict = model.state_dict()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pretrained_dict = torch.load("best_acc_80%.pth", weights_only=True, map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 冻结阶段训练参数，learning_rate和batch_size可以设置大一点
Init_Epoch = 0
Freeze_Epoch = 50
Freeze_batch_size = 8
Freeze_lr = 1e-3
# 解冻阶段训练参数，learning_rate和batch_size设置小一点
UnFreeze_Epoch = 100
Unfreeze_batch_size = 4
Unfreeze_lr = 1e-4
# 可以加一个变量控制是否进行冻结训练
Freeze_Train = True
# 冻结一部分进行训练
batch_size = Freeze_batch_size
lr = Freeze_lr
start_epoch = Init_Epoch
end_epoch = Freeze_Epoch
if Freeze_Train:
    for param in model.backbone.parameters():
        param.requires_grad = False
# 解冻后训练
batch_size = Unfreeze_batch_size
lr = Unfreeze_lr
start_epoch = Freeze_Epoch
end_epoch = UnFreeze_Epoch
if Freeze_Train:
    for param in model.backbone.parameters():
        param.requires_grad = True
