import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d
from scipy.sparse.linalg import spsolve
from fealpy.decorator import cartesian

class MembraneOscillationPDEData:

    def __init__(self, D=[0, 1, 0, 1], T=[0, 5]):

        self._domain = D
        self._duration = T

    def domain(self):

        return self._domain

    def duration(self):

        return self._duration

    @cartesian
    def source(self, p, t):

        return np.zeros_like(p[..., 0])

    @cartesian
    def init_solution(self, p):

        x, y = p[..., 0], p[..., 1]
        val = x**2 * (x+y)
        return val

    @cartesian
    def init_solution_diff_t(self, p):

        return np.zeros_like(p[..., 0])

    @cartesian
    def dirichlet(self, p, t):

        return np.zeros_like(p[..., 0])

pde = MembraneOscillationPDEData()
 #空间离散
domain = pde.domain()
nx = 50
ny = 50
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))

# 时间离散
duration = pde.duration()
nt = 1000
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node') # （nx+1, ny+1)
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node') # (nx+1, ny+1)
uh1 = mesh.function('node') # (nx+1, ny+1)

def advance_explicit(n, *frags):
    """
    @brief 时间步进为显格式

    @param[in] n int, 表示第 n 个时间步
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx
        ry = tau/hy
        uh1[1:-1, 1:-1] = 0.5*rx**2*(uh0[0:-2, 1:-1] + uh0[2:, 1:-1]) + \
                0.5*ry**2*(uh0[1:-1, 0:-2] + uh0[1:-1, 2:]) + \
                (1 - rx**2 - ry**2)*uh0[1:-1, 1:-1] + tau*vh0[1:-1, 1:-1]
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        A = mesh.wave_operator_explicit(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        uh2 = A@uh1.flat - uh0.flat

        uh0[:] = uh1[:]
        uh1.flat = uh2

        gD = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh1)

        return uh1, t

#图像绘制
box = [0, 1, 0, 1, -2, 2]
fig= plt.figure()
axes = fig.add_subplot(111,projection='3d')
mesh.show_animation(fig, axes, box, advance_explicit, fname='3-3-4.mp4', plot_type='surface', frames=nt+1)
plt.show()
