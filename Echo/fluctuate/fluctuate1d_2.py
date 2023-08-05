#格式为隐格式
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
nt = 40  #情况一
#nt = 20   #情况二
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')


#时间步进
def advance_implicit(n, theta=0.25, *frags):

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


#制作动图
box = [0, 1, -2, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_implicit ,fname='fluctuate1d_2.mp4', frames=nt+1)
plt.show()


