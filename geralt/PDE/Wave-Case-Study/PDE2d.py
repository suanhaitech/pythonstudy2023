import numpy as np
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.mesh import UniformMesh2d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_2d import MembraneOscillationPDEData

class MembraneOscillationPDEDataInstance(MembraneOscillationPDEData):
    @cartesian
    def init_solution(self, p):
        x, y = p[..., 0], p[..., 1]
        pi = np.pi
        val = (x**2) * (x + y)
        return val

# 建立pde模型
pde = MembraneOscillationPDEDataInstance(T=[0, 5])

# 空间离散
domain = pde.domain()
nx = 100
ny = 100
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

# 时间步进
# 显格式
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
        A = mesh.wave_operator_explicit(tau, a=0.5066)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        uh2 = A@uh1.flat - uh0.flat

        uh0[:] = uh1[:]
        uh1.flat = uh2

        gD = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh1)

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh1, errortype='max')
        print(f"the max error is {e}")

        return uh1, t

# 隐格式
def advance_implicit(n, *frags):
    """
    @brief 时间步进为隐格式

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
        A0, A1, A2 = mesh.wave_operator_implicit(tau, a=1.414) 
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f.flat += A1@uh1.flat + A2@uh0.flat

        uh0[:] = uh1[:]
        gD = lambda p: pde.dirichlet(p, t + tau)
        A0, f = mesh.apply_dirichlet_bc(gD, A0, f)
        uh1.flat = spsolve(A0, f)

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh1, errortype='max')
        print(f"the max error is {e}")

        return uh1, t
    

# 制作动画
# 显格式
box = [0, 1, 0, 1, -1.4, 1.4]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_explicit, 
                    fname='explicit.mp4', plot_type='imshow', frames=nt+1)
plt.show()


box = [0, 1, 0, 1, -1.4, 1.4]
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.show_animation(fig, axes, box, advance_explicit, 
                    fname='explicit.mp4', plot_type='surface', frames=nt+1)
plt.show()

# 隐格式
# box = [0, 1, 0, 1, -1.4, 1.4]
# fig, axes = plt.subplots()
# mesh.show_animation(fig, axes, box, advance_implicit,
#                     fname='implicit.mp4', plot_type='imshow', frames=nt+1)
# plt.show()


# box = [0, 1, 0, 1, -1.4, 1.4]

# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh.show_animation(fig, axes, box, advance_implicit,
#                     fname='implicit.mp4', plot_type='surface', frames=nt+1)
# plt.show()