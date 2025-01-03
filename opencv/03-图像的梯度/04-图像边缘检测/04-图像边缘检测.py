import cv2
import numpy as np

if __name__ == '__main__':

    def nothing(*arg):
        pass


    cv2.namedWindow('edge')
    cv2.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    cv2.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

    cap = cv2.VideoCapture(0)
    while True:
        flag, img = cap.read()
        img = cv2.GaussianBlur(img, (3, 3), 0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thrs1 = cv2.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv2.getTrackbarPos('thrs2', 'edge')
        edge = cv2.Canny(gray, thrs1, thrs2, apertureSize=5)
        vis = img.copy()
        vis = np.uint8(vis / 2.)
        vis[edge != 0] = (0, 255, 0)
        cv2.imshow('edge', vis)
        ch = cv2.waitKey(5) & 0xFF
        if ch == 27:
            break
    cv2.destroyAllWindows()