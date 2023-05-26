from sympy import *

x=symbols('x')
a=sin(x).series(x,0,7)
print(a)
#泰勒展开到第7项
print('-------------')

b=sin(x).series(x,5)
print(b)
#泰勒展开到x=5处
print('-------------')

c=Matrix([[12,15,14],[13,17,19],[4,5,6]])**-1
print(c)
#求逆矩阵
print('-------------')

d=Matrix([[64,65,66],[44,33,46],[12,13,14]]).det()
print(d)
#求行列式
print('-------------')

e=Matrix([[24,6,0,2],[2,0,7,8],[7,-3,-5,6]]).rref()
print(e)
#化为行阶梯型
print('-------------')

f=Matrix([[2,2,5],[1,8,4],[4,5,6]]).columnspace()
print(f)
#齐次线性方程组解空间
print('-------------')

y=Matrix([[-1,0,-1],[0,-3,6],[6,0,7]])

g=y.eigenvals()
print(g)
#求特征值
print('-------------')

h=y.eigenvects()
print(h)
#求特征向量
print('-------------')

i=y.diagonalize()
print(i)
#对角化

