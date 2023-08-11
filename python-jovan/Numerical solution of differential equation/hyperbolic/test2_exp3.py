import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse import csr_matrix
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
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
def hyperbolic_operator_explicity_upwind_with_viscous(mesh, tau, a):
    """
    @brief 生成双曲方程的带粘性项的迎风迭代矩阵
    @param[in] tau float, 当前时间步长
    """
    r = a*tau/mesh.h
    
    if r > 1.0:
        raise ValueError(f"The r: {r} should be smaller than 0.5")
    
    NN = mesh.number_of_nodes()
    k = np.arange(NN)
    
    A = diags([1-r], [0], shape=(NN, NN), format='csr')
    val0 = np.broadcast_to(0, (NN-1, ))
    val1 = np.broadcast_to(r, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += csr_matrix((val0, (J, I)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val1, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    
    return A
def hyperbolic_windward_with_vicious(n, *fargs): 
    """
    @brief 时间步进格式为带粘性项的显式迎风格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = hyperbolic_operator_explicity_upwind_with_viscous(mesh,pde.a(), tau)
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_windward_with_vicious,fname='hyperbolic_windward_with_vicious_test2.mp4', frames=nt+1,color='purple')
plt.show()
