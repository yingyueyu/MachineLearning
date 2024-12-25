import torch
import torchvision
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from dataset.utils import generate_batch_size

device = torch.device("cuda:0")

backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
backbone = backbone.to(device)
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((64, 128, 256),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
roi_pooler = roi_pooler.to(device)

model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
model = model.to(device)

train_data = torch.load("./data/dog_train.pth", weights_only=False)
valid_data = torch.load("./data/dog_val.pth", weights_only=False)
batch_images, batch_gits = generate_batch_size(train_data, 1)
batch_val_images, batch_val_gits = generate_batch_size(valid_data, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# ------------------------训练------------------------
epochs = 1
TP = 0
TF = 0
for epoch in range(epochs):
    model.train()
    losses = []
    for images, gts in zip(batch_images, batch_gits):
        images = images.to(device)
        optimizer.zero_grad()
        result = model(images.float(), gts)
        loss_classifier = result['loss_classifier']
        loss_box_reg = result['loss_box_reg']
        loss_objectness = result['loss_objectness']
        loss_rpn_box_reg = result['loss_rpn_box_reg']
        loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(loss.item())

    # 使用验证集测试结果
    # model.eval()
    # for images, gts in zip(batch_val_images, batch_val_gits):
    #     predictions = model(images.float())
    #     for i, prediction in enumerate(predictions):
    #         print(prediction)
            # TP += sum(prediction['scores'] > 0.7)
            # TF += len(prediction)

    total_loss = sum(losses) / len(losses)
    print(f"epoch {epoch + 1} / {epochs} -- loss:{total_loss:.4f} -- acc:{(TP / TF):.4f}")
