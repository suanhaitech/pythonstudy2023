"""
利用 lambdify 函数将 SymPy 表达式转换为 NumPy 可使用的函数
"""
import numpy as np
from sympy import *
a = np.pi/3
x= symbols('x')
e = sin(x)
f = lambdify(x, e, 'numpy')
print(f(a), e.subs(x, pi/3))


