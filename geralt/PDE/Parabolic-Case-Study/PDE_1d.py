import numpy as np
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.pde.parabolic_1d import HeatConductionPDEData
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from typing import Callable, Tuple, Any
from scipy.sparse.linalg import spsolve

# 右端函数为0，复写HeatConductionPDEData中的右端函数
class HeatConductionPDEDataInstance(HeatConductionPDEData):
    @cartesian
    def source(self, p, t):
        return np.zeros_like(p)
    

# PDE 模型 由于未规定物体长度l，因此这里依旧设置为1
pde = HeatConductionPDEDataInstance()

# 空间离散
domain = pde.domain()
nx = 40 
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 3200 
tau = (duration[1] - duration[0])/nt 

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')  # 根据指定的插值类型对初始解函数进行插值

# 时间步进
# 向前欧拉
def advance_forward(n: np.int_) -> Tuple[np.ndarray, np.float64]: 

    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 n 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau  # 当前时间步
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t + tau)  # 方程右端函数项
        f = mesh.interpolate(source, intertype='node')  # 根据指定的插值类型对右端函数进行插值，返回右端函数在网格节点上的插值值
        uh0[:] = A@uh0 + tau*f
        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh0)
        
        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t


# 向后欧拉
def advance_backward(n: np.int_) -> Tuple[np.ndarray, np.float64]: # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为向后欧拉方法

    @param[in] n int, 表示第 n 个时间步（当前时间步） 
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
        uh0[:] = spsolve(A, f)

        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t
    

# Crank-Nicholson
def advance_crank_nicholson(n: np.int_) -> Tuple[np.ndarray, np.float64]: # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为 CN 方法
    
    @param[in] n int, 表示第 n 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += B@uh0
        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t


# 制作向前离散动画
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance_forward, fname='advance_forward.mp4', 
                    frames=nt+1, lw=2, interval=50, linestyle='--', color='red')
plt.show()


# 制作向后离散动画
# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance_backward, fname='advance_backward.mp4', 
                    frames=nt+1, lw=2, interval=50, linestyle='--', color='blue')
plt.show()


# Crank-Nicholson 离散动画
# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance_crank_nicholson, fname='advance_crank_nicholson.mp4', 
                    frames=nt+1, lw=2, interval=50, linestyle='--', color='green')
plt.show()