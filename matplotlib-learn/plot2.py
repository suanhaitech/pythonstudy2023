import matplotlib.pyplot as plt
import numpy as np
# 创建数据
x = np.linspace(0, 2*np.pi, 100) # 其语法为首项，尾项，项数，详见numpy学习笔记
y = np.sin(x)
# 绘制折线图
plt.plot(x, y, 'r-', linewidth=2, label='sin(x)')
# 添加标题、坐标轴标签和图例
plt.title('Sin(x) Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# 显示图形
plt.show()
