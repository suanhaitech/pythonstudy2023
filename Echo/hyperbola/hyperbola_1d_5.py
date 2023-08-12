# 导入所需要的模块
import numpy as np
from fealpy.decorator import cartesian
from typing import Union, Tuple, List 
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.pde.hyperbolic_1d import Hyperbolic1dPDEData
from fealpy.mesh import UniformMesh1d

# 创建类
class Hyperbolic1dPDEDataInstance(Hyperbolic1dPDEData):
    @cartesian
    def solution(self,p: np.ndarray,t: np.float64) -> np.ndarray:
        pi = np.pi
        val = 1 + np.sin(2 * pi * (p + 2 * t)) 
        return val
        
    @cartesian
    def init_solution(self,p: np.ndarray) -> np.ndarray:
        pi = np.pi
        val = 1+np.sin(2*pi*p)
        return val
        
    @cartesian
    def dirichlet(self,p: np.ndarray,t: np.float64) -> np.ndarray:
        pi = np.pi
        return 1 + np.sin(4 * pi * t)
        
    def a(self) -> np.float64:
        return -2
        
# PDE 模型
pde = Hyperbolic1dPDEDataInstance()

# 空间离散
domain = pde.domain()
nx = 40
hx = (domain[1] - domain[0]) / nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 1600
tau = (duration[1] - duration[0]) / nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

# 时间步长
def lax_wendroff(n, *fargs):
    t = duration[0] + n * tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_lax_wendroff(pde.a(), tau)

        u0, u1 = uh0[[0, 1]]
        uh0[:] = A @ uh0

       
        gD = lambda p, t=t: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=-1)

        # 更新数值边界条件，三选一
        #uh0[0] = u0 + 2 * tau * (u1 - u0) / hx
        #uh0[0] = uh0[1]
        uh0[0] = 2*uh0[1] - uh0[2]

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

"""
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, lax_wendroff, fname='hytest1_5.mp4', frames=nt + 1)
plt.show()
"""

for i in range(nt +1):
    u , t = lax_wendroff(i)
    if t in [0.5,1.0,1.5,2.0]:
        fig,axes = plt.subplots(2,1)
        x = mesh.entity('node').reshape(-1)
        true_solution = pde.solution(x, t)
        error = true_solution - u
        print(f"At time {t}, Error: {error}")

        axes[0].plot(x, u, label='Numerical Solution')
        axes[0].plot(x, true_solution, label='True Solution')
        axes[0].set_ylim(-2, 2)
        axes[0].legend(loc='best')
        axes[0].set_title(f't = {t}')

        # 画出误差
        axes[1].plot(x, np.abs(u - true_solution), label='Error')
        axes[1].legend(loc='best')
        axes[1].set_title(f'Error at t = {t}')

        plt.tight_layout()
        plt.show()

