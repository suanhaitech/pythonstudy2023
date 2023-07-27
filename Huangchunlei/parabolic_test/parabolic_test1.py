import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.parabolic_1d import SinExpPDEData
from fealpy.mesh import UniformMesh1d
from fealpy.decorator import cartesian

class HeatConductionPDEData:

    def __init__(self, D=[0, 1], T=[0, 1], k=1):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        @param[in] k 热传导系数
        """
        self._domain = D
        self._duration = T
        self._k = k
        self._L = D[1] - D[0] 

    def domain(self):
        """
        @brief 空间区间
        """
        return self._domain

    def duration(self):
        """
        @brief 时间区间
        """
        return self._duration

    @cartesian
    def solution(self, p, t):
        """
        @brief 真解函数

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 真解函数值
        """
        return np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.sin(np.pi * p / self._L)

    @cartesian
    def init_solution(self, p):
        """
        @brief 初始解函数

        @param[in] x float, 空间点

        @return 初始解函数值
        """
        return np.sin(np.pi * p / self._L)

    @cartesian
    def source(self, p, t):
        """
        @brief 方程右端项

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 方程右端函数值
        """
        return 0
        
    @cartesian
    def gradient(self, p, t):
        """
        @brief 真解空间导数

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 真解空间导函数值
        """
        return (np.pi / self._L) * np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.cos(np.pi * p / self._L)

    @cartesian
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p float, 空间点
        @param[in] t float, 时间点
        """
        return self.solution(p, t)

# PDE 模型
pde = HeatConductionPDEData()



# 空间离散
domain = pde.domain()
nx = 40 
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 3200 
tau = (duration[1] - duration[0])/nt 

#准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')



#时间步进-----向前差分格式
def advance_forward(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f
        
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t


#时间步进---------向后差分格式
def advance_backward(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为向后欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += uh0
        
        gD = lambda p: pde.dirichlet(p, t)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t

#时间步进---------- Crank-Nicholson差分格式
def advance_crank_nicholson(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为 CN 方法
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += B@uh0

        gD = lambda p: pde.dirichlet(p, t)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t


"""
#制作动画---1

fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance_forward,fname='test1_1.mp4', frames=nt + 1) 
plt.show()

"""
#制作动画---2
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance_backward,fname='test1_2.mp4', frames=nt + 1)
plt.show()

"""
#制作动画---3
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance_crank_nicholson, fname='test1_3.mp4',frames=nt + 1)
plt.show()

"""



