import numpy as np
import cv2


def add_gaussian_noise(image, mean=0, sigma=10):
    h, w, c = image.shape
    gauss = np.random.normal(mean, sigma, (h, w, c))
    noise_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noise_image


image = cv2.imread("./assets/example.png")
noise_image = add_gaussian_noise(image)
cv2.imwrite("../assets/gauss_noise_example.png", noise_image)
