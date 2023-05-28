"""
网格线grid() 方法的语法及实例
matplotlib.pyplot.grid(b = None, which = 'major', axis = 'both'， **kwargs)
"""
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.title("RUNOOB grid() Test")
plt.xlabel("x - label")
plt.ylabel("y - label")
plt.plot(x, y)
plt.grid(axis='x') # 设置就在y轴方向显示网格线
plt.show()
