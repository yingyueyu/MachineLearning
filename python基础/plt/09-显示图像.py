import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as im

# 显示一个灰度矩阵图象
# img = np.random.rand(100, 100)
# plt.imshow(img, cmap='gray')
# # 颜色深度参考
# plt.colorbar()
# plt.show()

# 显示本地图片
img = im.open('下载.png')
print(img)
data = np.array(img)
plt.imshow(data)
# 不显示坐标轴
plt.axis('off')
plt.show()


