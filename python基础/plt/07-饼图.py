import matplotlib.pyplot as plt
import numpy as np

value = np.array([80, 70, 60, 40, 90])
label = np.array(['Python', 'c', 'Java', 'C++', 'JavaScript'])

# autopct 显示百分比，.xf保留几位小数
plt.pie(value, labels=label, autopct='%.2f%%')

plt.show()