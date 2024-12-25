import os.path
import cv2
from random import randint, random
from tqdm import tqdm


def generate_panels(panel_num, save_path="./panels_hr", hsv_random=False):
    for pn in tqdm(range(panel_num)):
        bg = cv2.imread("./panels/bg/blue_140.PNG")
        bg = cv2.resize(bg, (440, 140))
        panels = []
        city_element = randint(0, 30)
        panels.append(city_element)
        letter_element = randint(41, 64)
        panels.append(letter_element)
        for _ in range(5):
            chr_element = randint(31, 64)
            panels.append(chr_element)
        start = 20
        for i, panel_chr in enumerate(panels):
            city = cv2.imread(f"./panels/detect/{str(panel_chr)}.jpg")
            city = cv2.resize(city, (40, 80))
            city = 255 - city
            mask = city > bg[30:110, start:40 + start]
            bg[30:110, start:40 + start][mask] = city[mask]
            if i == 1:
                start += 40 + 40
            else:
                start += 40 + 12

        if hsv_random:
            hsv_img = cv2.cvtColor(bg.copy(), cv2.COLOR_BGR2HSV)
            random_v = randint(0, 100)
            hsv_img[:, :, 2] -= random_v
            bg = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join(save_path, "images", f"{pn}.jpg"), bg)
        with open(os.path.join(save_path, "labels", f"{pn}.txt"), "w") as f:
            f.write(" ".join([str(item) for item in panels]))
        f.close()


def generate_lr_panels(from_path="./panels_hsv",save_path="./panels_lr"):
    # lr panels is 35x110
    for file in tqdm(os.listdir(os.path.join(from_path, "images"))):
        image = cv2.imread(os.path.join(from_path,"images", f"{file}"))
        image = cv2.resize(image, (110, 35))
        kernel = randint(6, 12)
        image = cv2.blur(image, (kernel, kernel))
        cv2.imwrite(os.path.join(save_path, "images", f"{file}"), image)


if __name__ == '__main__':
    # generate_panels(3000, save_path="./panels_hsv", hsv_random=True)
    generate_lr_panels()
