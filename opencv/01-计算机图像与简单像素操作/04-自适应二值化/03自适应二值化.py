import cv2

srcImage = cv2.imread("../assets/example.png")
srcGray = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
# 初始化相关变量
# 初始化自适应阈值参数
maxVal = 255
blockSize = 3
constValue = 10
# 自适应阈值算法
# 0 ADAPTIVE_THRESH_MEAN_C
# 1 ADAPTIVE_THRESH_GAUSSIAN_C
adaptiveMethod = 0

# 阈值类型
# 0:THRESH_BINARY
# 1:THRESH_BINARY_INV
thresholdType = 0
# 图像自适应阈值操作
distImage = cv2.adaptiveThreshold(srcGray, maxVal, adaptiveMethod, thresholdType, blockSize, constValue)
cv2.imshow("srcImage", srcImage)
cv2.imshow("grayImage", srcGray)
cv2.imshow("Adaptive threshold", distImage)
cv2.waitKey(0)
