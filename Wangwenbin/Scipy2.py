from scipy.optimize import root          #最优化算法函数实现
def eqn(x):
    return x**3 -3*x**2 -x +3
myroot=root(eqn,[-2,0.5,4])
print(myroot.x)                          #求函数的零点
#为什么只有一个零点显示了？？？？？
#取大致的近似值可以取到所有？
