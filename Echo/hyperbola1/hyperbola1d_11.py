import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from typing import Union,Tuple,List 
from fealpy.pde.hyperbolic_1d import Hyperbolic1dPDEData
from fealpy.mesh import UniformMesh1d

# PDE 模型
class Hyperbolic1dPDEDataInstance(Hyperbolic1dPDEData):
    def a(self) -> np.float64:
        return 2
           
pde = Hyperbolic1dPDEDataInstance()

# 空间离散
domain = pde.domain()
nx = 40
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 3200
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

#时间步进
def hyperbolic_windward(n, *fargs): 
 
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_explicity_upwind(pde.a(), tau)
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
        
#制作动画
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_windward, fname='hytest1_1.mp4',frames=nt+1)
plt.show()

#计算误差
for i in range(nt +1):
    u , t = hyperbolic_windward(i)
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
