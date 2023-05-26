# 引入所需要的模块
import matplotlib.pyplot as plt
import numpy as np

# 创建一些数据
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# 创建一个画像和子图 
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Simple plot")

# 创建两个子图
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
ax1.plot(x, y)
ax1.set_title("Sharing Y axis")
ax2.scatter(x, y)

# 创建四个子图
fig, axs = plt.subplots(2 , 2, subplot_kw = dict(projection = "polar"))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

# 共享x轴
plt.subplots(2, 2, sharex = "col")

# 共享y轴
plt.subplots(2, 2, sharey = "row")

# 共享x轴和y轴
plt.subplots(2, 2, sharex = "all", sharey = "all")

# 创建标识为 10 的图，已存在的则删除
fig, ax = plt.subplots(num = 10, clear = True)

plt.show()


