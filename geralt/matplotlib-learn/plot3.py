"""
绘制坐标 (1, 3) 和 (8, 10) 的两个点
"""
import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints, 'o') # 使用参数o（实心圈）来绘制点
plt.show()
