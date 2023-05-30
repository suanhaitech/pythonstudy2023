"""
fmt用法及实例
fmt = '[marker][line][color]'
"""
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([6, 2, 13, 10]) 
# 不指定 x 轴上的点，则 x 会根据 y 的值来设置
plt.plot(ypoints, 'o:r') # o 表示实心圆标记，: 表示虚线，r 表示颜色为红色
plt.show()
