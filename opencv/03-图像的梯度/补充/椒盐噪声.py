import cv2
import numpy as np


def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = np.copy(image)
    salt = np.random.rand(*image.shape[:2]) < salt_prob
    pepper = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[salt] = 255
    noisy_image[pepper] = 0
    return noisy_image


image = cv2.imread("./assets/example.png")
noisy_image = add_salt_and_pepper_noise(image,0.001,0.001)
cv2.imwrite("../assets/salt_pepper_noise_example2.png", noisy_image)
