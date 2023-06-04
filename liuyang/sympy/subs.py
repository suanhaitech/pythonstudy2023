from sympy import *
x = symbols('x')
expr = cos(x) + 1
print(expr.subs(x, 0))
