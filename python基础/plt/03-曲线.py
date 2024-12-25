import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
# 加载字库文件
# zk = matplotlib.font_manager.FontProperties(fname='SourceHanSansSC-Normal.otf')

x = np.arange(0, 10, 0.1)
# print(x)
y = np.sin(x)
y2 = np.cos(x)
# print(y)
plt.plot(x, y)
# 一个坐标系可以设置多组数据，会画出多个图形
plt.plot(x, y2)

# 设置fontproperties的值为加载的字库名
# plt.xlabel('0~0的等距的点', fontproperties=zk)
# plt.ylabel('三角函数', fontproperties=zk)
# plt.title('正弦余弦曲线多图显示', fontproperties=zk)
plt.xlabel('0~0的等距的点')
plt.ylabel('三角函数')
plt.title('正弦余弦曲线多图显示')


plt.grid(True ,which='both', axis='both')

plt.show()