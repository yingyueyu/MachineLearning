import os
import cv2

ROOT_dir = "./VOCdevkit/ccpd_valid"


def _label_content(f1):
    image = cv2.imread(os.path.join(ROOT_dir, f1))
    h, w = image.shape[:2]
    points = "0"  # 由于只有车牌这一个区域，因此此处写0
    for point in f1.split("-")[3].split("_"):
        x, y = point.split("&")
        x = int(x) / w
        y = int(y) / h
        # 归一化
        points += f" {x:.6f} {y:.6f}"
    return points


def _generate_images_labels(ROOT_dir):
    files = os.listdir(ROOT_dir)
    count = 1
    for file in files:
        # 保存图片到指定位置
        image = cv2.imread(os.path.join(ROOT_dir, file))
        cv2.imwrite(f"./VOCdevkit/panels/images/valid/{count}.jpg", image)
        # 保存位置信息
        content = _label_content(file)
        with open(f"./VOCdevkit/panels/labels/valid/{count}.txt", "w") as f:
            f.write(content)
        f.close()
        count += 1


if __name__ == '__main__':
    _generate_images_labels(ROOT_dir)
