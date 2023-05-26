"""
scatter() 方法语法及实例
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)
"""
import matplotlib.pyplot as plt
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9, 16, 7, 11, 23, 18])
sizes = np.array([20, 50, 100, 200, 500, 1000, 60, 90]) # 设置图标大小
colors = np.array(['r', 'g', 'y', 'k', 'm', 'c', 'g', 'r']) # 自定义点颜色
plt.scatter(x, y, s = sizes, c = colors)
plt.show()
