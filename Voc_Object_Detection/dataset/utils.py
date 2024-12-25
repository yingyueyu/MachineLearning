from xml.dom.minidom import parse
import os
import cv2
import torch

# classes_list = ['background', 'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#                 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike',
#                 'train', 'bottle', 'chair', 'dining table', 'potted plant',
#                 'sofa', 'tv/monitor']

classes_list = ['background', 'dog']

classes_dict = {key: index for index, key in enumerate(classes_list)}

root = "../../voc/VOCdevkit/VOC2012"


def get_data_from_xml(filename, scale_x, scale_y):
    path = os.path.join(root, "Annotations", f"{filename}.xml")
    dom = parse(path)
    # 获取文档元素对象
    data = dom.documentElement
    # 获取获取所有的objects标签
    objects = data.getElementsByTagName('object')
    ground_truth = []
    # 从xml文件中解析信息
    for obj in objects:
        name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
        if name not in classes_list:
            continue
        idx = classes_dict[name]
        xmin = obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
        ymin = obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
        xmax = obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
        ymax = obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
        xmin = int(int(xmin) * scale_x)
        ymin = int(int(ymin) * scale_y)
        xmax = int(int(xmax) * scale_x)
        ymax = int(int(ymax) * scale_y)
        if xmin > xmax or ymin > ymax:
            continue
        # TODO idx 也可以单独设置到labels中
        ground_truth.append([idx, xmin, ymin, xmax, ymax])
    return ground_truth


def generate_image_gts(device, is_train=True):
    path = os.path.join(root, "ImageSets", "Main", "dog_train.txt" if is_train else "dog_val.txt")
    images = []
    gts = []
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        filename = line.split()[0]
        # -------------只生成带有分类目标的图像----------------
        label = int(line.split()[1])
        if label < 0:
            continue
        image = cv2.imread(os.path.join(root, "JPEGImages", f"{filename}.jpg"))
        h, w = image.shape[:2]
        scale_x = 600 / w
        scale_y = 600 / h

        # 获取ground_truth
        ground_truth = get_data_from_xml(filename, scale_x, scale_y)
        ground_truth = torch.tensor(ground_truth).to(device)
        if ground_truth.shape[-1] != 5:
            continue
        # ------------ 归一化的左上角坐标和右下角坐标 ----------------
        ground_truth = ground_truth / torch.tensor([1, w, h, w, h],device=device)
        gts.append(ground_truth)

        # 获取images图像数据
        image = cv2.resize(image, (600, 600))
        image = torch.from_numpy(image).permute([2, 0, 1])
        images.append(image.to(device))

    return {"images": images, "gts": gts}


def generate_batch_size(data, batch_size):
    images, gts = data['images'], data['gts']
    images = torch.stack(images, dim=0)
    len_images = len(images)
    num_batch = len_images // batch_size
    batch_images = []
    batch_gits = []
    for i in range(num_batch):
        batch_data = images[i * batch_size: (i + 1) * batch_size]
        batch_images.append(batch_data)
        boxes = []
        for j in range(i * batch_size, (i + 1) * batch_size):
            box = gts[j][:, 1:]
            label = gts[j][:, 0]
            # ------- 将每一张图片的所有真实框的位置信息与它的标签一起放置
            boxes.append({"boxes": box, "labels": label.long()})
        batch_gits.append(boxes)
    return batch_images, batch_gits


if __name__ == '__main__':
    device = torch.device("cuda:0")
    obj = generate_image_gts(device=device, is_train=False)
    torch.save(obj, "../data/dog_val.pth")

    # data = torch.load("../data/train.pth", weights_only=False)
    # batch_images, batch_gits = generate_batch_size(data, 1)
    # print(batch_images[0], batch_gits[0])
    # print(batch_gits[0]['labels'])
