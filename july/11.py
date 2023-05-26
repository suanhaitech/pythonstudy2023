# 建立COO 稀疏矩阵
from scipy.sparse import coo_matrix   # 引入所需要的库
row = [0, 1, 2, 2]
col = [0, 1, 2, 3]
data = [1, 2, 3, 4]                 # 建立矩阵的参数
c = coo_matrix((data, (row, col)), shape=(4, 4)) # 构建4*4的稀疏矩阵
print(c)

d =  c.todense()      # 稀疏矩阵转化为密集矩阵
print(d)
e = coo_matrix(d)    # 将一个0值很多的矩阵转为稀疏矩阵
print(e)

f = e.tocsr()        # 将COO 稀疏矩阵转化为CSR稀疏矩阵
print(f)
print("\n")
g = e.tocsc()        # 将COO 稀疏矩阵转化为CSC稀疏矩阵 
print(g)

