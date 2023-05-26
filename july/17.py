import numpy as np
import matplotlib.pyplot as plt     # 引入所需要的模块

x = np.array([1, 2, 3, 4])
y = np.array([1, 5, 12, 25])        # 创建数组

plt.title("RUNOOB grid() Test")
plt.xlabel("x - label")
plt.ylabel("y - label")            # 设置图形的标题，坐标轴

plt.plot(x, y)
plt.grid( axis='x',  color = 'r', linestyle = '--', linewidth = 0.5)  # 设置参数

plt.show()        # 作出图形
