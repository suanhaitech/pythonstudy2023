import numpy as np
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from typing import List,Tuple
from scipy.sparse.linalg import spsolve
# l=1
class SinExpPDEData: # 点击这里可查看 FEALPy 仓库中的代码
    def __init__(self, D=[0, 1], T=[0, 1]):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D
        self._duration = T

    def domain(self)->List[int]:
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
    def solution(self, p, t)->np.ndarray:
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点

        @return 真解函数值
        """
        pi = np.pi
        return np.sin(pi*p/l)*np.exp((-t*pi**2)/l**2)

    @cartesian
    def init_solution(self, p):
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点

        @return 真解函数值
        """
        pi = np.pi
        return np.sin(pi*p/l)

    @cartesian
    def source(self, p, t):
        """
        @brief 方程右端项

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点

        @return 方程右端函数值
        """
        pi = np.pi
        return 0

    @cartesian
    def gradient(self, p, t):
        """
        @brief 真解导数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点

        @return 真解导函数值
        """
        pi = np.pi
        return (pi/l)*np.exp((t*pi**2)/l**2)*np.cos(pi*p/l)


    @cartesian
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点
        """
        return self.solution(p, t)
        
# PDE 模型
pde = SinExpPDEData()
l=1

# 空间离散
domain = pde.domain()
nx = 40 
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 3200 
tau = (duration[1] - duration[0])/nt 

uh0 = mesh.interpolate(pde.init_solution, intertype='node')

def advance_forward(n: np.int_) -> Tuple[np.ndarray, np.float64]: # 点击这里查看 FEALPy 中的代码

    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 n 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f
        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
        
fig, axes = plt.subplots()
box = [0, 1, -0.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance_forward, fname='advance_forward.mp4', 
                    frames=nt+1, lw=2, interval=50, linestyle='--', color='red')
plt.show()





