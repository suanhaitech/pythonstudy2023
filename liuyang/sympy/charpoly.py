from sympy import *
M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
lamda = symbols('lamda')
p = M.charpoly(lamda)
print(M.eigenvals())
print(factor(p))
