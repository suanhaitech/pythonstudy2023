"""
线的类型可以使用 linestyle 参数来定义，简写为 ls
线的颜色可以使用 color 参数来定义，简写为 c
线的宽度可以使用 linewidth 参数来定义，简写为 lw，值可以是浮点数，如：1、2.0等
"""
import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, ls = ':', c = 'r', lw = '13.2')
plt.show()
