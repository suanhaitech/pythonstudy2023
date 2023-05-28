from scipy.sparse  import coo_matrix, csr_matrix, csc_matrix
import numpy as np
I = np.array([0, 0, 3, 1, 0])
J = np.array([0, 0, 3, 1, 2])
V = np.array([4, 2, 5, 7, 9])
A = coo_matrix((V, (I, J)), shape=(4, 4))
B = csr_matrix((V, (I, J)), shape=(4, 4))
C = csc_matrix((V, (I, J)), shape=(4, 4))
print(A, B, C)
