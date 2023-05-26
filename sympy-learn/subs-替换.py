from sympy import *
x, y = symbols('x y')
expr = cos(x) + 1
a = expr.subs(x, y) # expr.subs(old, new)
print(a)
