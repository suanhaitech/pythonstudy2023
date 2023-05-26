from sympy import *
x = symbols('x')
t = (x**2 + 2*x + 1)/(x**2 + x)
y = diff(t, x, 2)
print(y)



