from sympy import *

x=symbols('x')
expr=tan(x)+2
y=expr.subs(x,pi/3)
print('tan(pi/3+1=')
print(y)

a=pi.evalf(4)
print('pi的四位有效数字')
print(a)

c=simplify((x+1)**3+(x-1)**3)
print('化简x+1的立方加x-1的立方')
print(c)

d=expand((x-1)**3)
print('展开x-1的立方')
print(d)
print()
z=symbols('z')
e=collect(3*x**3+x*z+3*z-x**3+7*x**2,x)
print(e)

f=(3*x**4+5*x**3+4*x+14)/(x**3+3*x**2+8*x)
g=apart(f)
print('部分分式展开')
print(g)

h=diff(1/x,x,3)
print('1/x的三阶导数')
print(h)

print('x**(x*z)关于x的偏导数')
print(diff(x**(x*z),x))

print('x平方在(0,4)上的定积分')
print(integrate(x**2,(x,0,4)))

print('(1+1/x)**x当x趋于无穷时的极限')
print(limit((1+1/x)**x,x,00))

