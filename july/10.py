# SciPy 的 optimize 模块的应用
# 查找 x + cos(x) 方程的根

from scipy.optimize import root  #导入所需要的模块
from math import cos
def eqn(x):
    return x + cos(x)      # 定义函数
myroot = root(eqn,0)       # 调用函数
print(myroot.x)            # 得出方程的根
print(myroot)              # 查看更多信息



