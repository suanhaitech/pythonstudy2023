import numpy as np
A = np.arange(0, 9).reshape(3, 3)
print(A)
B = np.einsum('ij->ji', A)#矩阵转置
C = np.einsum('ij->', A)#所有元素求和
print(B)
print(C)
