"""
bar()方法及实例
matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
"""
import matplotlib.pyplot as plt
import numpy as np
x = np.array(["R-1", "R-2", "R-3", "R-4"])
y = np.array([12, 22, 6, 18])
plt.subplot(2, 2, 1)
plt.bar(x, y, color = 'r') # 设置柱形图的同时指定其颜色
plt.subplot(2, 2, 2)
plt.barh(x, y) # 设置垂直方向的柱形图
c = np.array(['k', 'g', 'r', 'y'])
plt.subplot(2, 2, 3)
plt.bar(x, y, color = c) # 自定义各个柱形的颜色
# 设置柱形图宽度，bar() 方法使用 width 设置，barh() 方法使用 height 设置 height
w = np.array([0.3, 0.4, 0.1, 0.9]) # 自定义各个柱形的宽度
plt.subplot(2, 2, 4)
plt.bar(x, y, width = w)
plt.show()
