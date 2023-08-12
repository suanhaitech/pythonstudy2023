
from scipy import constants# 一英亩等于多少平方米
print(constants.acre)

import numpy as np

from scipy.sparse import coo_matrix
_row  = np.array([0,3,1,0])
_col  = np.array([0,3,1,2])
_data = np.array([4,5,7,9])
coo = coo_matrix((_data,(_row,_col)),shape=(4,4),dtype=np.int64)
print(coo.todense())
print(coo.toarray())


from scipy.sparse import csr_matrix
indptr = np.array([0, 2, 3, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr = csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
print(csr)


from scipy.sparse import csc_matrix
indptr = np.array([0, 2, 3, 5, 7])
indices = np.array([1, 3, 3, 1, 2, 0, 3])
data = np.array([21, 12, 1, 33, 3, 10, 4])
csc = csc_matrix((data, indices, indptr), shape=(4, 4)).toarray()
print(csc)


from scipy.sparse  import coo_matrix, csr_matrix, csc_matrix
I = np.array([0, 0, 3, 1, 0])
J = np.array([0, 0, 3, 1, 2])
V = np.array([4, 2, 5, 7, 9])
A = coo_matrix((V, (I, J)), shape=(4, 4)).toarray()
B = csr_matrix((V, (I, J)), shape=(4, 4)).toarray()
C = csc_matrix((V, (I, J)), shape=(4, 4)).toarray()
print(A)
print(B)
print(C)






