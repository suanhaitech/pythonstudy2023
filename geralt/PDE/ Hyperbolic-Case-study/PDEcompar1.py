import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from fealpy.pde.hyperbolic_1d import Hyperbolic1dPDEData
from fealpy.mesh import UniformMesh1d


class Hyperbolic1dPDEDataInstance2(Hyperbolic1dPDEData):
    @cartesian
    def solution(self, p: np.ndarray, t: np.float64) -> np.ndarray:
        val = 1 - np.sin(p + t)
        return val
    
    @cartesian
    def init_solution(self, p: np.ndarray) -> np.ndarray:
        val = 1 - np.sin(p)
        return val
    
    @cartesian
    def source(self, p: np.ndarray , t: np.float64 ) -> np.float64:
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        return 0.0
    
    @cartesian
    def dirichlet(self, p: np.ndarray, t: np.float64) -> np.ndarray:
        return 1 - np.sin(t + 1)
    
    @cartesian
    def a(self) -> np.float64:
        return -1
    

# pde模型
pde = Hyperbolic1dPDEDataInstance2(D=[0, 1], T=[0, 1])

# 空间离散
domain = pde.domain()
nx = 480
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 1600
tau = (duration[1] - duration[0])/nt 

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

# 显式迎风格式
def hyperbolic_windward(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为迎风格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        e = 0.0
        return uh0, t, e
    else:
        A = mesh.hyperbolic_operator_explicity_upwind(pde.a(), tau)
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t, e


# 积分守恒型 Lax - Friedrichs 格式
# def hyperbolic_lax(n, *fargs): 
#     """
#     @brief 时间步进格式为守恒型 Lax 格式

#     @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
#     """
#     t = duration[0] + n*tau
#     if n == 0:
#         e = 0.0
#         return uh0, t, e
#     else:
#         A = mesh.hyperbolic_operator_explicity_lax_friedrichs(pde.a(), tau)
#         uh0[:] = A@uh0

#         gD = lambda p: pde.dirichlet(p, t)
#         mesh.update_dirichlet_bc(gD, uh0, threshold=0)

#         solution = lambda p: pde.solution(p, t)
#         e = mesh.error(solution, uh0, errortype='max')
#         print(f"the max error is {e}")
#         return uh0, t, e
    

    

# 带粘性项的显式迎风格式
# def hyperbolic_windward_with_vicious(n, *fargs): 
#     """
#     @brief 时间步进格式为带粘性项的显式迎风格式

#     @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
#     """
#     t = duration[0] + n*tau
#     if n == 0:
#         e = 0.0
#         return uh0, t, e
#     else:
#         A = mesh.hyperbolic_operator_explicity_upwind_with_viscous(pde.a(), tau)
#         uh0[:] = A@uh0

#         gD = lambda p: pde.dirichlet(p, t)
#         mesh.update_dirichlet_bc(gD, uh0, threshold=0)

#         solution = lambda p: pde.solution(p, t)
#         e = mesh.error(solution, uh0, errortype='max')
#         print(f"the max error is {e}")
#         return uh0, t, e
    

# def hyperbolic_lax_wendroff(n, *fargs): 
#     """
#     @brief 时间步进格式为 lax-wendroff 格式

#     @param[in] n int, 表示第 `n` 个时间步（当前时间步）
#     """
#     t = duration[0] + n*tau
#     if n == 0:
#         # e = 0.0
#         return uh0, t
#     else:
#         A = mesh.hyperbolic_operator_lax_wendroff(pde.a(), tau)
#         u0, u1 = uh0[[0, 1]]
#         uh0[:] = A@uh0

#         gD = lambda p: pde.dirichlet(p, t)
#         mesh.update_dirichlet_bc(gD, uh0, threshold=-1)

#         # 三种数据边值条件
#         # uh0[0] = u0 + 2*tau*(u1 - u0) / hx
#         # uh0[0] = u1
#         uh0[0] = 2*uh0[1] - uh0[2]

#         solution = lambda p: pde.solution(p, t)
#         e = mesh.error(solution, uh0, errortype='max')
#         print(f"the max error is {e}")
#         return uh0, t
    
# 制作动画
# box = [0, 1, -0.05, 2.05]
# fig, axes = plt.subplots()
# mesh.show_animation(fig, axes, box, hyperbolic_windward, frames=nt+1)
# plt.show()

# box = [0, 1, -0.05, 2.05]
# fig, axes = plt.subplots()
# mesh.show_animation(fig, axes, box, hyperbolic_lax, frames=nt+1)
# plt.show()


# box = [0, 1, -0.05, 2.05]
# fig, axes = plt.subplots()
# mesh.show_animation(fig, axes, box, hyperbolic_windward_with_vicious, frames=nt+1)
# plt.show()

# 制作动画
# box = [0, 1, -0.05, 2.05]
# fig, axes = plt.subplots()
# mesh.show_animation(fig, axes, box,hyperbolic_lax_wendroff, fname='hyperbolic_lax_wendroff.mp4', frames=nt+1,color='green')
# plt.show()   

# 计算误差
s = 0.0
for n in range(nt + 1):
    t = duration[0] + n * tau
    _, _, e = hyperbolic_windward(n)

    s += e

print(f"显示迎风格式的平均误差为{s/(nt + 1)}")
