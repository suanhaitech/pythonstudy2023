import matplotlib.pyplot as plt
import numpy as np
# 生成一组随机数据
data = np.random.randn(1000)
# 绘制直方图
plt.hist(data, bins=30, color='skyblue', alpha=0.8)
# 设置图表属性
plt.title('hist() Test')
plt.xlabel('Value')
plt.ylabel('Frequency')
# 显示图表
plt.show()
