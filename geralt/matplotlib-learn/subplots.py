"""
subplots()方法语法及实例
fig, ax = plt.subplots(nrows, ncols, sharex, sharey, figsize)
subplots() 函数返回一个图像对象 fig 和一个子图数组 ax，数组的维度为 (nrows, ncols)。
"""
import numpy as np
import matplotlib.pyplot as plt
# 创建一些数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
# 创建一个包含两个子图的图表
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 6))
# 在第一个子图中绘制 sin 函数
ax1.plot(x, y1, 'r', label='sin(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
# 在第二个子图中绘制 cos 函数
ax2.plot(x, y2, 'b', label='cos(x)')
ax2.set_xlabel('x')
