import numpy as np
A = np.arange(0, 9).reshape(3, 3)
print(A)
B = np.einsum('ij->ji', A)
print(B)

#矩阵所有元素求和
A = np.arange(0, 9).reshape(3, 3)
print(A)
B = np.einsum('ij->', A)
print(B)


#矩阵相乘求和
A = np.arange(0, 12).reshape(3, 4)
print(A)
B = np.arange(0, 12).reshape(3, 4)
C = np.einsum('ij,ij->', A, B)
print(C)
