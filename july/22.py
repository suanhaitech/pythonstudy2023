# 引入所需要的模块
import matplotlib.pyplot as plt
import numpy as np

# 创建数组
x = np.array(["Runoob-1", "Runoob-2", "Runoob-3", "Runoob-4"])
y = np.array([12, 22, 6, 18])

plt.barh(x, y, color = ["#4CAF50", "red", "hotpink", "#556B2F"])
plt.show()

