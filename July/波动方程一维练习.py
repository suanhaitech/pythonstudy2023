# 导入相关的模块
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d
from scipy.sparse.linalg import spsolve
from fealpy.decorator import cartesian

# 创建类
class StringOscillationPDEData:
    def  __init__(self,D=[0,1],T=[0,0.5]):
        self._domain = D
        self._duration = T

    def domain(self):
        return self._domain

    def duration(self):
        return self._duration

    @cartesian
    def solution(self,p,t):
        pi = np.pi
        return (np.sin(pi*(p - t))+np.sin(pi*(p+t)))/2 -(np.sin(pi*(p - t))-np.sin(pi*(p+t)))/2*pi
        
    @cartesian
    def init_solution(self,p):
        pi = np.pi
        return np.sin(pi*p)

    @cartesian
    def init_solution_diff_t(self,p):
        pi = np.pi
        return np.cos(pi*p)

    @cartesian
    def source(self,p,t):
        return 0.0

    @cartesian
    def gradient(self,p,t):
        pi = np.pi
        return pi*(np.cos(pi*(p - t))+np.cos(pi*(p+t)))/2 -(np.cos(pi*(p - t))-np.cos(pi*(p+t)))/2

    @cartesian
    def dirichlet(self,p,t):
        return self.solution(p, t)

# 创建对象
pde = StringOscillationPDEData()

# 空间与时间离散
# 空间离散
domain = pde.domain()
nx = 10
hx = (domain[1]-domain[0])/nx
mesh = UniformMesh1d([0,nx], h=hx,origin=domain[0] )

# 时间离散
duration = pde.duration()
nt = 10
tau = (duration[1] - duration[0])/nt

# 初值准备
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')

# 时间步长
def advance(n,*frags):
    theta =  0.0
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        r = tau / hx
        uh1[1:-1] = r**2*(uh0[0:-2] + uh0[2:])/2.0 + (1 - r**2)*uh0[1:-1] + tau*vh0[1:-1]
        gD = lambda p: pde.dirichlet(p,t)
        mesh.update_dirichlet_bc(gD,uh1)
        return uh1,t
    else:
        A0,A1,A2 = mesh.wave_operator(tau, theta =  theta)
        source = lambda p: pde.source(p,t)
        f = mesh.interpolate(source, intertype = 'node')
        f *= tau**2
        f += A1@uh1 +A2@uh0

        uh0[:] = uh1
        gD = lambda p: pde.dirichlet(p,t)
        if theta == 0.0:
            uh1[:] = f
            mesh.update_dirichlet_bc(gD, uh1)
        else:
            A0, f = mesh.apply_dirichlet_bc(gD,A0,f)
            uh1[:] = spsolve(A0, f)
            
        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh1, errortype='max')
        print(f"the max error is {e}")
        return uh1,t

# 制作动画
box = [0,1,-5,5]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance,fname='parabolic_ab.mp4',frames = nt + 1 )
plt.show()

