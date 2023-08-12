import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from typing import Union,Tuple,List 
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


def hyperbolic_windward(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为迎风格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
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

box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_windward, fname='hytest1_1.mp4',frames=nt+1)
plt.show()
