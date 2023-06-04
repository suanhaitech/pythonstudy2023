from sympy import *
M = Matrix([[1, 3], [-2, 3]])
A = M**2     #求二次幂
B = M**-1    #求逆
C = M.det()  #求行列式
print(A)
print(B)
print(C)
