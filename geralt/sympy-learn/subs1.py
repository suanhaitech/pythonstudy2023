"""
subs一次替换多个变量
"""
from sympy import *
x, y, z = symbols('x y z')
e = x**3 + 4*x*y - z
a = e.subs([(x, 2), (y, 4), (z, 0)])
print(a)
