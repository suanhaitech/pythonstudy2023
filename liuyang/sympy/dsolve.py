from sympy import *
f = symbols('f', cls = Function)
x = symbols('x')
diffeq = Eq(f(x).diff(x, 2) - 2*f(x).diff(x) + f(x), sin(x))
print(dsolve(diffeq, f(x)))
