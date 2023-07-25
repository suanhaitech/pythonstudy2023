from sympy import *
x,y,z = symbols('x y z')
expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
print(collect(expr, x))
