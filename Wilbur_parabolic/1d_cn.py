import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from typing import List, Callable, Tuple, Any
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
#类
class SinExpPDEData:
    def __init__(self, T=[0, 1],D = [0,1]):
        self._domain = D
        self._duration = T
        self._l = D[1] - D[0]
    def domain(self) -> List[int]:

        return self._domain

    def duration(self) -> List[int] :

        return self._duration

    @cartesian
    def solution(self, p, t) -> np.ndarray:

        pi = np.pi
        return np.exp(-pi**2 * t /  self._l**2) * np.sin(pi * p /  self._l)

    @cartesian
    def init_solution(self, p) -> np.ndarray:

        pi = np.pi
        return np.sin(pi*p /  self._l)

    @cartesian
    def source(self, p, t) -> np.ndarray:

        return np.zeros(p.shape)

    @cartesian
    def gradient(self, p, t) -> np.ndarray:

        pi = np.pi
        return (pi/self._l) * np.exp(-pi**2 * t /  self._l**2) * np.cos(pi * p /  self._l)


    @cartesian
    def dirichlet(self, p, t) -> np.ndarray:

        return self.solution(p, t)
#实例
pde = SinExpPDEData()
domain = pde.domain()
#空间离散
nx = 40
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

#时间离散
duration = pde.duration()
nt = 3200
tau = (duration[1] - duration[0])/nt
#初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
#Crank-Nicholson 离散格式
def advance_crank_nicholson(n:np.int_)-> Tuple[np.ndarray, np.float64]:
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += B@uh0

        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t
#制作动画
fig, axes = plt.subplots()
box = [0, pde._l, 0, 1.5] # 图像显示的范围 0 <= x <= l, 0 <= y <= 1.5

mesh.show_animation(fig, axes, box, advance_crank_nicholson,linestyle='--', frames=nt + 1)
plt.show()
