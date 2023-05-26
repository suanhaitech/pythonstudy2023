import numpy as np
A=np.array([[1,1,1],[2,2,2],[3,3,3]])
print(A)
#生成一个3*3矩阵A
print('\n')

B=np.einsum('ij->ji',A)
print(B)
#输出一个A的转置
print('\n')

C=np.einsum('ii->i',A)
print('A的对角线元素：\n',C)
#A的对角线元素
print('\n')

D=np.einsum('ii->',A)
print('A的迹：\n',D)
print('\n')

E=np.einsum('ij->j',A)
print(E)
#按列求和
print('\n')
F=np.einsum('ik,kj->ij',A,B)
print('AB=\n',F)
#A与B做矩阵乘法
