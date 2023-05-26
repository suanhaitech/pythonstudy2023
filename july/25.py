import matplotlib.pyplot as plt
import numpy as np

# 生成一个随机的彩色图像
img = np.random.rand(10, 10, 3)

# 绘制彩色图像
plt.imshow(img)

# 显示图像
plt.show()
