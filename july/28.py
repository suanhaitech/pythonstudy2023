from sympy import *
x = symbols("x")
y = limit(sin(x)/x, x, 0)
z = limit(sin(x)/x, x, 0, "+")

print(y)
print(z)

