import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_1d import StringOscillationPDEData
pde = StringOscillationPDEData()
# 空间离散
domain = pde.domain()
nx = 100
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 1000
tau = (duration[1] - duration[0])/nt
# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')
def advance_implicit(n, theta=0.25, *frags):
    """
    @brief 时间步进格式为隐格式

    @param[in] n int, 表示第 n 个时间步（当前时间步） 
    """
    # theta = 0.25
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx 
        uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        A0, A1, A2 = mesh.wave_operator_implicit(tau, theta=theta)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f += A1@uh1 + A2@uh0

        uh0[:] = uh1[:]
        gD = lambda p: pde.dirichlet(p, t + tau)
        A0, f = mesh.apply_dirichlet_bc(gD, A0, f)
        uh1[:] = spsolve(A0, f)
            
        return uh1, t

advance_implicit_theta = partial(advance_implicit, theta=0.25)
box = [0, 1, -0.1, 0.1]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_implicit_theta, fname='implicit.mp4', frames=nt+1)
plt.show()
