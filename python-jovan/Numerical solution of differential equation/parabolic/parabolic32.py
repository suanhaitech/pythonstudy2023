import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.sparse.linalg import spsolve
from fealpy.pde.parabolic_2d import SinSinExpPDEData
from fealpy.mesh import UniformMesh2d
# PDE 模型
pde = SinSinExpPDEData()
# 空间离散
domain = pde.domain()
nx = 40
ny = 40
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))

# 时间离散
duration = pde.duration()
nt = 6400
tau = (duration[1] - duration[0])/nt
uh0 = mesh.interpolate(pde.init_solution, intertype='node') #(nx+1, ny+1)
def advance_crank_nicholson(n: np.int_) -> Tuple[np.ndarray, np.float64]: # 点击这里可以查看 FEALPy 仓库中的代码
    """
    @brief 时间步进格式为 CN 方法
    
    @param[in] n int, 表示第 n 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node') # f.shape = (nx+1,ny+1)
        f *= tau
        f.flat[:] += B@uh0.flat[:]
         
        gD = lambda p: pde.dirichlet(p, t+tau)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0.flat = spsolve(A, f)

        solution = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")

        return uh0, t
        
fig, axes = plt.subplots()
box = [0, 1, 0, 1, -1, 1] # 图像显示的范围 0 <= x <= 1, 0 <= y <= 1, -1 <= uh <= 1
mesh.show_animation(fig, axes, box, advance_crank_nicholson, 
                    fname='parabolic_cn.mp4', plot_type='contourf', frames=nt + 1)
plt.show()
