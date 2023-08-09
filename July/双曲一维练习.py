# Lax-Wendroff 格式
# 导入所需要的模块
import numpy as np
from fealpy.decorator import cartesian
from typing import Union,Tuple,List    # Union: 将多个集合合并为一个集合
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d

# 创建类
class Hyperbolic1dPDEData1:

    def __init__(self,D: Union[Tuple[int,int],List[int]] = (0,1),T: Union[Tuple[int, int], List[int]] = (0, 2)):
        self._domain = D
        self._duration = T

    def domain(self) -> Union[Tuple[float,float],List[float]]:
        return self._domain

    def duration(self) -> Union[Tuple[float,float],List[float]]:
        return self._duration

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
    def source(self,p: np.ndarray, t: np.float64) -> np.ndarray:
        return 0.0

    @cartesian
    def dirichlet(self,p: np.ndarray,t: np.float64) -> np.ndarray:
        pi = np.pi
        return 1 + np.sin(4 * pi * t)

    def a(self) -> np.float64:
        return -2

# 创建对象
pde =Hyperbolic1dPDEData1()

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
        uh0[0] = u0 + 2 * tau * (u1 - u0) / hx
        # uh0[0] = uh0[1]
        # uh0[0] = 2*uh0[1] - uh0[2]

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t


box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, lax_wendroff, frames=nt + 1,color = 'red')
plt.show()
