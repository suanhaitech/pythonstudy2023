import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.pde.hyperbolic_1d import Hyperbolic1dPDEData
from fealpy.mesh import UniformMesh1d

# 根据条件对Hyperbolic1dPDEData类复写
class Hyperbolic1dPDEDataInstance(Hyperbolic1dPDEData):
    @cartesian
    def solution(self, p: np.ndarray, t: np.float64) -> np.ndarray:
        pi = np.pi
        val = 1 + np.sin(2*pi*(p + 2*t))
        return val
    
    @cartesian
    def init_solution(self, p: np.ndarray) -> np.ndarray:
        """
        @brief 初值函数

        @param[in] p numpy.ndarray, 空间点

        @return 初值函数值
        """
        pi = np.pi
        val = 1 + np.sin(2*pi*p)
        return val
    
    @cartesian
    def dirichlet(self, p: np.ndarray, t: np.float64) -> np.ndarray:
        pi = np.pi
        return 1 + np.sin(4*pi*t)
    
    @cartesian
    def a(self) -> np.float64:
        return -2
    
# pde模型
pde = Hyperbolic1dPDEDataInstance(D=[0, 1], T=[0, 2])

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

# 时间步进
def hyperbolic_lax_wendroff(n, *fargs): 
    """
    @brief 时间步进格式为 lax-wendroff 格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        # e = 0.0
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_lax_wendroff(pde.a(), tau)
        u0, u1 = uh0[[0, 1]]
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=-1)

        # 三种数据边值条件
        uh0[0] = u0 + 2*tau*(u1 - u0) / hx
        # uh0[0] = u1
        # uh0[0] = 2*uh0[1] - uh0[2]

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
    
# 制作动画
box = [0, 1, -0.05, 2.05]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box,hyperbolic_lax_wendroff, fname='hyperbolic_lax_wendroff.mp4', frames=nt+1,color='green')
plt.show()

# 计算误差
# s = 0.0
# for n in range(nt + 1):
#     t = duration[0] + n * tau
#     _, _, e = hyperbolic_lax_wendroff(n)

#     s += e

# print(f"第三种边值条件的平均误差为{s/(nt + 1)}")