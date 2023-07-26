import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fealpy.mesh import UniformMesh2d
from fealpy.decorator import cartesian
from typing import Callable, Tuple, Any

class SinSinExpPDEData: # 点击这里可查看 FEALPy 仓库中的代码
    def __init__(self, D=[0, 1, 0, 1], T=[0, 1]):
        """
        @brief 模型初始化函数
        
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D 
        self._duration = T 

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

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.exp(-2*(pi**2)*t)*np.sin(pi*x)*np.sin(pi*y)

    @cartesian
    def init_solution(self, p):
        """
        @brief 初始解

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(pi*x)*np.sin(pi*y)
        
    @cartesian
    def source(self, p, t):
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.zeros(x.shape)
        
    @cartesian
    def gradient(self, p, t):
        """
        @brief 真解导数 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解导函数值
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)*np.exp(-2(pi**2)*t)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)*np.exp(-2(pi**2)*t)
        return val
    
    @cartesian    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 
        """
        return self.solution(p, t)
# PDE 模型
pde = SinSinExpPDEData()

# 空间离散
domain = pde.domain()
nx = 20
ny = 20
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))

# 时间离散
duration = pde.duration()
nt = 3200 
tau = (duration[1] - duration[0])/nt 
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
   
def advance_backward(n: np.int_) -> Tuple[np.ndarray, np.float64]: # 点击这里可以查看 FEALPy 仓库中的代码
    """
    @brief 时间步进格式为向后欧拉方法
    
    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau)
        
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += uh0

        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0.flat = spsolve(A, f)
        
        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
        
#画图
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_backward, 
                    fname='parabolic_ab.mp4', plot_type='surface', frames=nt + 1)
plt.show()
