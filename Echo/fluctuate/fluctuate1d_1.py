#格式为显格式
import numpy as np
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_1d import StringOscillationPDEData

class StringOscillationPDEData1: 
    def __init__(self, D=[0, 1], T=[0, 2]): 

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
        val = pi / 2 * (np.cos(pi * (p - t)) + np.cos(pi * (p + t))) - (np.cos(pi * (p -t) - np.cos(pi * (p + t)))) / 2

        return val

    
    @cartesian    
    def dirichlet(self, p, t):

        return self.solution(p,t) 

pde = StringOscillationPDEData1()


# 空间离散
domain = pde.domain()
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
#nt = 40  #情况一
nt = 20   #情况二
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')


def advance_explicit(n, *frags):

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

"""
#制作动图
box = [0, 1, -2, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_explicit, fname='fluctuate1d_1.mp4', frames=nt+1)
plt.show()


#获取特定图层
fig, axs = plt.subplots(2, 2)

times = [0.5, 1.0, 1.5, 2.0]
x = mesh.entity("node")
for n in range(nt + 1):
    t = duration[0] + n*tau
    y, t1 = advance_explicit(n)
    if t in times:
        if t == times[0]:
            axs[0, 0].plot(x, y)
            axs[0, 0].set_title(f't = {times[0]}')
        elif t == times[1]:
            axs[0, 1].plot(x, y)
            axs[0, 1].set_title(f't = {times[1]}')
        elif t == times[2]:
            axs[1, 0].plot(x, y)
            axs[1, 0].set_title(f't = {times[2]}')
        elif t == times[3]:
            axs[1, 1].plot(x, y)
            axs[1, 1].set_title(f't = {times[3]}')
# 调整子图布局
plt.tight_layout()
plt.show()
"""

#计算误差
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
