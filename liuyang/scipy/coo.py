import numpy as np
from scipy.sparse import coo_matrix
row  = np.array([0, 3, 1, 0])
col  = np.array([0, 3, 1, 2])
data = np.array([4, 5, 7, 9])
coo = coo_matrix((data, (row, col)), shape=(4, 4))
A=coo.todense()  # 通过toarray方法转化成密集矩阵(numpy.matrix)
print(A)
