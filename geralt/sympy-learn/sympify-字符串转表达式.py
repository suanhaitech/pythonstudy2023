"""
利用symplify函数转换字符串
注意其应该和化简函数simplify区分开来
"""
from sympy import *
str_expr = 'x**2 + 2*x + 1'
expr = sympify(str_expr)
print(expr)
