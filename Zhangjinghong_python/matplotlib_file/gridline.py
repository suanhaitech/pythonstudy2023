import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4])
y = np.array([1,4,9,16])
plt.plot(x,y)

plt.grid()
#设置 y 就在轴方向显示网格线
#plt.grid(axis='x')
#设置颜色为红色，破折线，宽度为0.5的网格线
#plt.grid(color = 'r', linestyle = '--', linewidth = 0.5)
plt.show()


