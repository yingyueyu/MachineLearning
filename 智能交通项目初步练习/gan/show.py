import torch
import cv2

images = torch.load("save/g_images_epoch2_batch57.pt", weights_only=False)
show_img = images[:10].permute([0, 2, 3, 1]).reshape(-1, 28).detach().numpy()
cv2.imshow("show",show_img)
cv2.waitKey(0)