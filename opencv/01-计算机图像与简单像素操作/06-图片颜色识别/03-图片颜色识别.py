import cv2
import numpy as np


def color_detection(frame):
    # 定义颜色范围（在HSV颜色空间中）
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    # 将帧转换为HSV颜色空间
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 根据颜色范围创建掩膜
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # 对掩膜进行形态学操作，以去除噪声
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # 在原始帧中找到颜色区域并绘制方框
    contours, _ = cv2.findContours(red_mask + blue_mask + green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        color = ""
        if cv2.contourArea(contour) > 500:  # 设置最小区域面积以排除噪声
            if np.any(red_mask[y:y + h, x:x + w]):
                color = "red"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            elif np.any(blue_mask[y:y + h, x:x + w]):
                color = "blue"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif np.any(green_mask[y:y + h, x:x + w]):
                color = "green"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    # 进行颜色识别
    result = color_detection(frame)

    # 显示结果帧
    cv2.imshow("Color Detection", result)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
