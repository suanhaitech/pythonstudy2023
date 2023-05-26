import numpy as np
import matplotlib.pyplot as plt      # 导入所需要的模块

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.plot(x, y)                     # 作出图形

plt.title("RUNOOB TEST TITLE")    # 标题
plt.xlabel("x - label")
plt.ylabel("y - label")           # 坐标轴

plt.show()

