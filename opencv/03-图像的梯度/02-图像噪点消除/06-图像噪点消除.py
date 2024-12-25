import cv2

# cv2.GaussianBlur() # 高斯滤波器
# cv2.medianBlur() # 中值滤波器

# 抑制椒盐噪声
# salt_pepper_noisy_image = cv2.imread("../assets/salt_pepper_noise_example.png")
# result = cv2.medianBlur(salt_pepper_noisy_image, ksize=3)
# cv2.imshow("noisy image",salt_pepper_noisy_image)
# cv2.imshow("processing result",result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 抑制高斯噪声
gauss_noisy_image = cv2.imread("../assets/gauss_noise_example.png")
result = cv2.GaussianBlur(gauss_noisy_image, ksize=(3, 3), sigmaX=0.8, sigmaY=0.8)
cv2.imshow("noisy image", gauss_noisy_image)
cv2.imshow("processing result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
