# 引入所需要的模块
import matplotlib.pyplot as plt
import numpy as np

# 随机数生成种子
np.random.seed(458654125)

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2   # 0 to 15 point radii

plt.scatter(x, y, s = area, c = colors, alpha = 0.5)  # 设置颜色和透明度

plt.title("Simple drawing")

plt.show()


