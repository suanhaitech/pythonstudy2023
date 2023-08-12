import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.hyperbolic_1d import Hyperbolic1dPDEData
from fealpy.mesh import UniformMesh1d


# PDE 模型
pde = Hyperbolic1dPDEData()


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

from scipy.sparse import csr_matrix
from scipy.sparse import diags

# 组装矩阵
def hyperbolic_operator_explicity_upwind_with_viscous():
    
    a=1
    r=a*tau/hx 

    if r > 1.0:
        raise ValueError(f"The r: {r} should be smaller than 0.5")

    NN=nx+1
    k = np.arange(NN)

    A = diags([1-r], [0], shape=(NN, NN), format='csr')
    val0 = np.broadcast_to(0, (NN-1, ))
    val1 = np.broadcast_to(r, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += csr_matrix((val0, (J, I)), shape=(NN, NN))
    A += csr_matrix((val1, (I, J)), shape=(NN, NN))

    return A


#时间步进
def hyperbolic_windward_with_vicious(n, *fargs): 

    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = hyperbolic_operator_explicity_upwind_with_viscous()
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
mesh.show_animation(fig, axes, box, hyperbolic_windward_with_vicious, fname='hytest1_4.mp4',frames=nt+1)
plt.show()
