import torchvision.transforms as transforms

from PIL import Image

image_path = './Lenna.png'

image = Image.open('./Lenna.png')

# 随机水平反转图像
# p: 概率
transform = transforms.RandomHorizontalFlip(p=0.5)

# 随机垂直翻转图像
transform = transforms.RandomVerticalFlip(p=0.5)

# 随机旋转图像
transform = transforms.RandomRotation(degrees=30)
transform = transforms.RandomRotation(degrees=[0, 90])

# 随机裁剪并调整图像大小。
# size: 指定裁剪后图片的 高x宽；并且裁剪后的图片还原回原图比例时，回得到正确的图片比例
# 例如下图裁剪后图片为 64x256，但重置大小为 512x512 时，图片比例是正确的
transform = transforms.RandomResizedCrop(size=(256, 64), scale=(0.2, 0.5))

# 随机改变图像的亮度、对比度、饱和度和色调
# brightness (float 或者 float 的元组 (min, max)): 调整亮度的幅度。
# contrast (float 或者 float 的元组 (min, max)): 调整对比度的幅度。
# saturation (float 或者 float 的元组 (min, max)): 调整饱和度的幅度。
# hue (float 或者 float 的元组 (min, max)): 调整色调的幅度。
transform = transforms.ColorJitter(brightness=(0.8, 1.3))
transform = transforms.ColorJitter(contrast=(2, 5))
transform = transforms.ColorJitter(saturation=(0.2, 0.4))
transform = transforms.ColorJitter(hue=(-0.2, 0.2))

# 随机仿射变换图像
# degrees: 旋转角度区间
# translate: 水平和竖直平移的幅度，是个百分比值
# scale: 缩放区间，是个百分比值
# shear 是平行错位的幅度，填一个值代表水平幅度随机 [-shear, shear] 之间的值，也可以填两个值，代表下限和上限值；但是填4个值，前两个代表水平幅度，后两个代表竖直幅度
# transform = transforms.RandomAffine(degrees=15, translate=(0.1, 0.2), scale=(0.5, 1.5))
transform = transforms.RandomAffine(degrees=0, translate=(0.1, 0.2), scale=(0.5, 1.5), shear=[0, 0, 15, 30])

# 随机将图像转换为灰度图像
transform = transforms.RandomGrayscale(p=0.5)


# 标准化图像
# 标准化公式: output= (input - mean) / std
# mean: 每个颜色通道上指定的均值
# std: 每个颜色通道上指定的标准差，标准差描述了这个颜色通道的每个像素值离均值的一个平均波动幅度
# 标准化后的图像，在各通道上每个像素点将变成接近均值且均匀分布
# 为什么要标准化？
# 标准化的主要原因是使所有特征（像素值）具有相似的尺度，从而加速神经网络的训练过程。对于预训练的神经网络（如使用在 ImageNet 上预训练的模型），通常需要使用相应的 mean 和 std 参数，以匹配模型训练时使用的数据分布。
# 那么 mean 和 std 该如何指定呢？
# 若我们从预训练模型上迁移过来进行训练，那么可以先获取预训练时使用的数据集，求出数据集中每张图片每个通道的均值和标准差
# 相关api调用和思路: (这里 images 是一个批次图片的张量，下面的参数仅供参考)
# mean += images.mean(2).sum(0) 求均值并叠加
# std += images.std(2).sum(0) 求标准差并叠加
# 然后用总值除以总图片数得到平均的 mean 和 std
# mean /= total_images_count
# std /= total_images_count
transform = transforms.Compose([
    transforms.ToTensor(),
    # 只能用在张量上
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 组合使用示例:
transform = transforms.Compose([
    # 先做变换
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.5),
    # 再转张量
    transforms.ToTensor(),
    # 最后标准化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image = transform(image)
print(image.shape)
print(image)
transforms.ToPILImage()(image).show()

# image.show()
# image.save('lenna_test.png')
