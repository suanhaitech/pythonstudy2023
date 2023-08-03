import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d
from fealpy.decorator import cartesian
from scipy.sparse.linalg import spsolve
class StringOscillationPDEData:
    def __init__(self, D=[0, 1], T=[0, 2]): #nt = 40

        self._domain = D
        self._duration = T

    def domain(self):

        return self._domain

    def duration(self):

        return self._duration

    @cartesian
    def solution(self, p, t):
        pi = np.pi
        return (np.sin(pi * (p - t)) + np.sin(pi * (p + t))) / 2 - (np.sin(pi * (p - t)) - np.sin(pi * (p + t))) / (2 * pi)

    @cartesian
    def init_solution(self, p):

        pi = np.pi
        val = np.sin(pi * p)
        return val

    @cartesian
    def init_solution_diff_t(self, p):

        pi = np.pi
        val = np.cos(pi * p)
        return val

    @cartesian
    def source(self, p, t):

        return np.zeros_like(p)

    @cartesian
    def gradient(self, p, t):

        pi = np.pi
        val = pi / 2 * (np.cos(pi * (p - t)) + np.cos(pi * (p + t))) - (np.cos(pi * (p -t)) - np.cos(pi * (p + t))) / 2

        return val


    @cartesian
    def dirichlet(self, p, t):

        return self.solution(p,t)

#实例生成
pde = StringOscillationPDEData()
# 空间离散
domain = pde.domain()
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 40
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')

def advance_explicit(n, *frags):
    """
    @brief 时间步进格式为显格式

    @param[in] n int, 表示第 n 个时间步
    """
    a = 1
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
        A = mesh.wave_operator_explicit(tau, a)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        uh2 = A@uh1 - uh0 + f  # 根据差分方程进行更新

        uh0[:] = uh1[:]
        uh1[:] = uh2
        gD = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh1)

        return uh1, t

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

for i in range(nt + 1):
    u,_ = advance_explicit(i)
    if (duration[0] + i * tau) in [0.5,1.0,1.5,2.0]:
        fig ,axes = plt.subplots()
        x = mesh.entity('node').flat
        axes.plot(x,u)
        axes.set_ylim(-2,2)
        axes.set_title(f't = {duration[0] + i * tau}')
        plt.show()

for i in range(nt +1):
    u , t = advance_explicit(i)
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
