import numpy as np
from scipy.sparse import coo_matrix

A=np.array([[0,0,1],[3,0,2],[0,6,0]])
B=coo_matrix((A),shape=(3,3))
print('A=\n',A)
print(B)                                      #按坐标索引非零元素
print('所有非零元素为\n',B.data)              #给出所有非零元素
print('非零元总数为\n',B.count_nonzero())     #给出非零元的数量
print('删去零元后\n',B.eliminate_zeros())     #删除所有零元
