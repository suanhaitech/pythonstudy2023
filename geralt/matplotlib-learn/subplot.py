"""
subplot()语法及实例
subplot(nrows, ncols, index, **kwargs)
以上函数将整个绘图区域分成 nrows 行和 ncols 列，然后从左到右，从上到下的顺序对每个子区域进行编号 1...N ，左上的子区域的编号为 1、右下的区域编号为 N，编号可以通过参数 index 来设置。
如设置 numRows ＝ 1，numCols ＝ 2，就是将图表绘制成 1x2 的图片区域
"""
import matplotlib.pyplot as plt
import numpy as np
#plot 1:
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])
plt.subplot(1, 2, 1)
plt.plot(xpoints,ypoints)
plt.title("plot 1")
#plot 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.subplot(1, 2, 2)
plt.plot(x,y)
plt.title("plot 2")
plt.suptitle("RUNOOB subplot Test")
plt.show()
