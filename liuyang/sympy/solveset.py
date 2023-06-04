from sympy import *
x = symbols('x')
Eq(x**2 - x, 0)
print(solveset(Eq(x**2 - x, 0), x, domain = S.Reals))
