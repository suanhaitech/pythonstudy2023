import numpy as np
#from ..decorator import cartesian
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from fealpy.pde.parabolic_2d import SinSinExpPDEData
from fealpy.mesh import UniformMesh2d
class SinSinExpPDEData: 
    def __init__(self, D=[0, 1, 0, 1], T=[0, 0.1]):
        """
        @brief 模型初始化函数
        `
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

    #@cartesian
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
        return np.sin(pi*x)*np.sin(pi*y)*np.exp(-2*pi**2*t) 

    #@cartesian
    def init_solution(self, p):
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(pi*x)*np.sin(pi*y)
        
    #@cartesian
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
        return 0
    #@cartesian
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
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)*np.exp(-2*pi**2*t)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)*np.exp(-2*pi**2*t)
        return val
    
    #@cartesian    
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

uh0 = mesh.interpolate(pde.init_solution, intertype='node') #(nx+1, ny+1)

def advance_forward(n, *fargs): # 点击这里可以查看 FEALPy 仓库中的代码
    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)

        source = lambda p: pde.source(p, t+tau)
        f = mesh.interpolate(source, intertype='node')

        uh0.flat = A@uh0.flat + (tau*f)
        gD = lambda p: pde.dirichlet(p, t+tau)
        mesh.update_dirichlet_bc(gD, uh0)

        solution = lambda p: pde.solution(p, t+tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

def advance_backward(n, *fargs): # 点击这里可以查看 FEALPy 仓库中的代码
    """
    @brief 时间步进格式为向后欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau)
        
        source = lambda p: pde.source(p, t+tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += uh0
        
        gD = lambda p: pde.dirichlet(p, t+tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0.flat = spsolve(A, f)
        
        solution = lambda p: pde.solution(p, t+tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
def advance_crank_nicholson(n, *fargs): # 点击这里可以查看 FEALPy 仓库中的代码
    """
    @brief 时间步进格式为 CN 方法  
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t+tau)
        f = mesh.interpolate(source, intertype='node') 
        f *= tau
        f += B@uh0.flat
         
        gD = lambda p: pde.dirichlet(p, t+tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0.flat = spsolve(A, f)

        solution = lambda p: pde.solution(p, t+tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t
#向前
"""
fig, axes = plt.subplots()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_forward, fname='parabolic_af.mp4', plot_type='imshow', frames=nt + 1)
plt.show()

#向后

fig, axes = plt.subplots()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_backward, fname='parabolic_ab.mp4', plot_type='imshow', frames=nt + 1)
plt.show()
"""
fig, axes = plt.subplots()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_crank_nicholson, fname='parabolic_acn.mp4', plot_type='imshow', frames=nt + 1)
plt.show()

