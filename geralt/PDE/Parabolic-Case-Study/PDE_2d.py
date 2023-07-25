import numpy as np
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.pde.parabolic_2d import SinSinExpPDEData
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
from typing import Callable, Tuple, Any
from scipy.sparse.linalg import spsolve

# 原包内不含对应的函数，因此对SinSinExpPDEData类进行复写
class HeatConductionPDEDataInstance(SinSinExpPDEData):
    @cartesian
    def solution(self, p, t):
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(pi*x)*np.sin(pi*y)*np.exp(-2*(pi**2)*t)
    
    @cartesian
    def init_solution(self, p):
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(pi*x)*np.sin(pi*y)
    
    @cartesian
    def source(self, p, t):
        return np.zeros_like(p[..., 0])
    
    @cartesian
    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)*np.exp(-2*(pi**2)*t)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)*np.exp(-2*(pi**2)*t)
        return val
    

# PDE 模型
pde = HeatConductionPDEDataInstance(D=[0, 1, 0, 1], T=[0, 0.1])

# 空间离散
domain = pde.domain()
nx = 20
ny = 20
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))

# 时间离散
duration = pde.duration()
nt = 6400 
tau = (duration[1] - duration[0])/nt 

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

# 时间步进
# 向前欧拉
def advance_forward(n: np.int_) -> Tuple[np.ndarray, np.float64]: 
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
        uh0[:].flat = A@uh0[:].flat + (tau*f[:]).flat
        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
    


# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

# 向后欧拉
def advance_backward(n: np.int_) -> Tuple[np.ndarray, np.float64]: 
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
    

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

# Crank-Nicholson
def advance_crank_nicholson(n: np.int_) -> Tuple[np.ndarray, np.float64]: # 点击这里可以查看 FEALPy 仓库中的代码
    """
    @brief 时间步进格式为 CN 方法
    
    @param[in] n int, 表示第 n 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node') # f.shape = (nx+1,ny+1)
        f *= tau
        f.flat[:] += B@uh0.flat[:]
         
        gD = lambda p: pde.dirichlet(p, t+tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0.flat = spsolve(A, f)

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t
    

# 制作向前离散动画
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
fig, axes = plt.subplots()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_forward, 
                    fname='parabolic_af.mp4',plot_type='imshow', frames=nt + 1)
plt.show()


# 制作向后离散动画
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_backward, 
                    fname='parabolic_ab.mp4', plot_type='surface', frames=nt + 1)
plt.show()


# Crank-Nicholson 离散动画
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
fig, axes = plt.subplots()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_crank_nicholson, 
                    fname='parabolic_cn.mp4', plot_type='contourf', frames=nt + 1)
plt.show()

