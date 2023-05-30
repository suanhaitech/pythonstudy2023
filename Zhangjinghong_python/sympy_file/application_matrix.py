from sympy import * # 导入 sympy 中所有的功能
x, y = symbols('x, y') # 定义两个 Python 变量，分别指向 

Matrix([[1, -1], [3, 4], [0, 2]])#构造矩阵
Matrix([1, 2, 3])#构造列向量
Matrix([[1], [2], [3]]).T# 构造行向量
eye(3)#构造单位矩阵
zeros(3)#构造0矩阵
ones(4)#构造1矩阵
diag(1, 2, 3, 4)# 构造对角矩阵

a = Matrix([[1,-1],[3,4],[0,2]])
a.T

M = Matrix([[1,3],[-2,3]])
M**2

M**-1#求逆

M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
M.det()#行列式

M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2],[5, -2, -3,  3]])
M.eigenvals()#求特征值
lamda = symbols('lamda')
p = M.charpoly(lamda)
factor(p)#求特征多项式
