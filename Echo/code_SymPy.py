from sympy import *
import numpy as np

x,y,z = symbols('x y z')



expr = cos(x) + 1
print(expr.subs(x, 0))    



str_expr = 'x**2 + 2*x + 1'
expr = sympify(str_expr)
print(expr)                



a = np.pi / 3
expr = sin(x)
f = lambdify(x, expr, 'numpy')
print(f(a))
print(expr.subs(x, pi/3))



print(simplify(sin(x)**2 + cos(x)**2))



x_1 = symbols('x_1')
print(expand((x_1 + 1)**2))



print(factor(x**3 - x**2 + x - 1))



expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
print(collect(expr, x))



print(cancel((x**2 + 2*x + 1)/(x**2 + x)))



print(cancel((x**2 + 2*x + 1)/(x**2 + x)))



print(diff(cos(x), x))# 求一阶导数
print(diff(x**4, x, 3))# 求 3 阶导数

expr = cos(x)   #求微分
print(expr.diff(x, 2))



expr = exp(x*y*z)
print(diff(expr, x))



print(integrate(cos(x), x))# 求不定积分



print(limit(sin(x)/x, x, 0))



expr = sin(x)
print(expr.series(x, 0, 4))



#求解一元二次方程
Eq(x**2 - x, 0)
print(solveset(Eq(x**2 - x, 0), x, domain = S.Reals))#使用 Eq 函数构造等式



f = symbols('f', cls = Function)#建立符号函数变量
diffeq = Eq(f(x).diff(x, 2) - 2*f(x).diff(x) + f(x), sin(x))
print(dsolve(diffeq, f(x)))



print(Matrix([[1, -1], [3, 4], [0, 2]]))# 构造矩阵M
print(Matrix([1, 2, 3]))# 构造列向量
print(Matrix([[1], [2], [3]]).T)# 构造行向量，矩阵转置用矩阵变量的 T 方法。
print(eye(4))# 构造单位矩阵
print(zeros(4))# 构造零矩阵
print(ones(4))# 构造壹矩阵
print(diag(1, 2, 3, 4))# 构造对角矩阵



M = Matrix([[1, 3], [-2, 3]])
print(M**2) # 求矩阵 M 的 2 次幂
print(M**-1)# 求矩阵 M 的逆



M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
print(M.det())



M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
print(M.eigenvals())

lamda = symbols('lamda')
p = M.charpoly(lamda)
print(factor(p))



from sympy.abc import t, s
expr = sin(t)
laplace_transform(expr, t, s)
expr = 1/(s - 1)
print(inverse_laplace_transform(expr, s, t))



from sympy.plotting import plot
from sympy.abc import x
plot(x**2, (x, -2, 2))   #绘制二维函数图像

from sympy import plot_implicit
from sympy import Eq
from sympy.abc import x, y
plot_implicit(Eq(x**2 + y**2, 1))#绘制隐函数图像

from sympy.plotting import plot3d
from sympy.abc import x, y
from sympy import exp
plot3d(x*exp(-x**2 - y**2), (x, -3, 3), (y, -2, 2))#画出三维函数图像



a = (x + 1)**2
b = x**2 + 2*x + 1
print(simplify(a - b))

c = x**2 - 2*x + 1
print(simplify(a - c))



a = cos(x)**2 - sin(x)**2
b = cos(2*x)
print(a.equals(b))





