import matplotlib.pyplot as plt
import numpy as np

# 创建一些测试数据 -- 图1
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# 创建一个画像和子图 -- 图2
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# 创建两个子图 -- 图3
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# 创建四个子图 -- 图4
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))#画极坐标图
axs[0, 0].plot(x, y)#(0,0)位置画线型图
axs[1, 1].scatter(x, y)#(1,1)位置画散点图
# 共享 x 轴
plt.subplots(2, 2, sharex='col')
# 共享 y 轴
plt.subplots(2, 2, sharey='row')
# 共享 x 轴和 y 轴
plt.subplots(2, 2, sharex='all', sharey='all')
# 这个也是共享 x 轴和 y 轴
#plt.subplots(2, 2, sharex=True, sharey=True)

# 创建标识为 10 的图，已经存在的则删除
fig, ax = plt.subplots(num=10, clear=True)
plt.show()
