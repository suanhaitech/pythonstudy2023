# 导入所需要的库
from sympy import *

# 新建符号变量
x, y, z = symbols("x y z")

# expand() 是展开函数
y = expand((x + 1) ** 2)

# 构造分数 1/2
z = Rational(1, 2) 

print(y)
print(z)

