# 绘制一个4*4的子图
import matplotlib.pyplot as plt
import numpy as np                # 引入所需要的模块
# 绘制第一个子图
x = np.array([0, 12, 14, 18])
y = np.array([1,2,5,8])
plt.subplot(2, 2 ,1)
plt.title("plot1")
plt.plot(x, y)
plt.grid( axis='x',  color = 'r', linestyle = '--', linewidth = 0.5)
# 绘制第二个子图
x = np.array([0, 2, 4, 8])
y = np.array([0,7,14,20])
plt.subplot(2, 2 ,2)
plt.title("plot2")
plt.plot(x, y)
plt.grid( axis='y',  color = 'g', linewidth = 2)
# 绘制第三个子图
x = np.array([0, 1, 2, 5, 8, 12, 17])
y = np.array([4, 16, 25, 38, 41, 52, 64])
plt.subplot(2, 2 ,3)
plt.title("plot3")
plt.plot(x, y)
plt.grid(color = 'y', linestyle = '-', linewidth = 0.1)
# 绘制第四个子图
x = np.array([0, 60])
y = np.array([1, 52])
plt.subplot(2, 2 ,4)
plt.title("plot4")
plt.plot(x, y)
plt.grid(color = 'c', linestyle = '-.', linewidth = 0.5)
# 开始作图
plt.suptitle("RUNOOB subplot Test")
plt.show()

