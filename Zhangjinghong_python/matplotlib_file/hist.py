import matplotlib.pyplot as plt
import numpy as np

# 生成一组随机数据
data = np.random.randn(1000)
# 绘制直方图，箱数是10，颜色透明度为0.8
plt.hist(data, bins=30, color='skyblue', alpha=0.8)
# 设置图表属性
plt.title('RUNOOB hist() Test')
plt.xlabel('Value')
plt.ylabel('Frequency')
# 显示图表
plt.show()

#两组直方图
import matplotlib.pyplot as plt
import numpy as np

# 生成三组随机数据
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1, 1000)
data3 = np.random.normal(-2, 1, 1000)

# 绘制直方图
plt.hist(data1, bins=30, alpha=0.5, label='Data 1')
plt.hist(data2, bins=30, alpha=0.5, label='Data 2')
plt.hist(data3, bins=30, alpha=0.5, label='Data 3')

# 设置图表属性
plt.title('RUNOOB hist() TEST')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# 显示图表
plt.show()
