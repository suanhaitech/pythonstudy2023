from sympy import * # 导入 sympy 中所有的功能
x, y = symbols('x, y') # 定义两个 Python 变量，分别指向

#因式分解
factor(x**3 - x**2 + x - 1)

#合并同类项
expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
collect(expr, x)

#有理分式化简
cancel((x**2 + 2*x + 1)/(x**2 + x))

#分式展开
expr = (4*x**3 + 21*x**2 + 10*x + 12)/(x**4 + 5*x**3 + 5*x**2 + 4*x)
apart(expr)

#求导
diff(cos(x),x)
diff(x**4,x,3)
#用 符号变量的 diff 方法 求微分
expr = cos(x)
expr.diff(x,2)

#多元函数求偏导
expr = exp(x*y*z)
diff(expr, x)

#积分
integrate(exp(-x), (x, 0, oo))#在0到无穷上对x积分
integrate(exp(-x**2 - y**2), (x, -oo, oo), (y, -oo, oo))

#求极限
limit(sin(x)/x, x, 0)#x趋于0

#级数展开
expr = sin(x)
expr.series(x, 0, 4)

#解方程
Eq(x**2 - x, 0)
solveset(Eq(x**2 - x, 0), x, domain = S.Reals)#domain是定义域

#求解微分方程
f = symbols('f', cls = Function)
diffeq = Eq(f(x).diff(x, 2) - 2*f(x).diff(x) + f(x), sin(x))
dsolve(diffeq, f(x))


