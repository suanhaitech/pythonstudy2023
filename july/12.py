# 建立一个CSR稀疏矩阵
from scipy.sparse  import csr_matrix
import numpy as np                          # 引入所需要的数据库
indptr = np.array([0, 2, 3, 3, 3, 6, 6, 7])
indices = np.array([0, 2, 2, 2, 3, 4, 3])
data = np.array([8, 2, 5, 7, 1, 2, 9])     # 生成csr格式的矩阵
t = csr_matrix((data, indices, indptr))    # 生成矩阵
print(t)
a = t.toarray()      # 转为array
print(a)

