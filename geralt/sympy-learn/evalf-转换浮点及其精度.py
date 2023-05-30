from sympy import *
x = symbols('x')
e = sqrt(8)
b = e.evalf()
c = e.evalf(3) # 保留3位有效数字 evalf方法
print(e, b, c)
e = cos(2*x)
a = e.evalf(subs = {x: 2.4})
print(a)
