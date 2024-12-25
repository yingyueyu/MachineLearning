import cv2
import numpy as np
import os

# 需要测试的图片
test_pic = cv2.imread("./test/8.png")
test_pic = cv2.resize(test_pic, dsize=(125, 250))
test_pic_gray = cv2.cvtColor(test_pic, cv2.COLOR_BGR2GRAY)
_,test_pic_gray = cv2.threshold(test_pic_gray,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow("img",test_pic_gray)

# 准备模板
files = os.listdir("./numbers")
match_score = np.zeros((10,))
for i, filename in enumerate(files):
    test = np.zeros((500, 1000), dtype=np.uint8)
    template_pic = cv2.imread(f"./numbers/{filename}", 0)
    template_pic = cv2.resize(template_pic, dsize=(125, 250))
    # cv2.imshow(f"img{i}", template_pic)
    h, w = template_pic.shape
    test[:h, :w] = test_pic_gray
    result = cv2.matchTemplate(test, template_pic, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    match_score[i] = max_val

result = np.argmax(match_score)
print("结果为：",result)
cv2.waitKey(0)
