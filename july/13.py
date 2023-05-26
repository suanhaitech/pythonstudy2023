from scipy.sparse import csc_matrix     # 导入所需要的模块
import numpy as np 
row = np.array([0, 2, 2, 0, 1, 2])
col = np.array([0, 0, 1, 2, 2, 2])
data = np.array([1, 2, 3, 4, 5, 6])     # 生成数据

c = csc_matrix((data, (row, col)), shape=(3, 3))  # 创建矩阵

# 转为array
d = c.toarray()
print(d)     # 转为array

