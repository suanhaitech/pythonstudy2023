import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

#from fealpy.pde.hyperbolic_1d import Hyperbolic1dPDEData
from fealpy.mesh import UniformMesh1d
import statistics
import numpy as np # 具体代码可参考 FEALPy 仓库

from fealpy.decorator import cartesian
from typing import Union, Tuple, List

class Hyperbolic1dPDEData:
    def __init__(self, D: Union[Tuple[int, int], List[int]] = (0, 2), T: Union[Tuple[int, int], List[int]] = (0, 4)):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D
        self._duration = T

    def domain(self) -> Union[Tuple[float, float], List[float]]:
        """
        @brief 空间区间
        """
        return self._domain

    def duration(self)-> Union[Tuple[float, float], List[float]]:
        """
        @brief 时间区间
        """
        return self._duration

    @cartesian
    def solution(self, p: np.ndarray, t: np.float64) -> np.ndarray:
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点

        @return 真解函数值
        """
        val = np.zeros_like(p)
        flag1 = p <= t
        flag2 = p > t+1
        flag3 = ~flag1 & ~flag2

        val[flag1] = 1
        val[flag3] = 1 - p[flag3] + t
        val[flag2] = p[flag2] - t - 1

        return val

    @cartesian
    def init_solution(self, p: np.ndarray) -> np.ndarray:
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点

        @return 真解函数值
        """
        val = np.zeros_like(p)
        val = np.abs(p-1)

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
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点
        """
        return np.ones(p.shape)

    def a(self) -> np.float64:
        return 1

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

#显式迎风格式
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
        #print(A)
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t,e

#积分守恒
def hyperbolic_lax(n, *fargs):
    """
    @brief 时间步进格式为守恒型 Lax 格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_explicity_lax_friedrichs(pde.a(), tau)
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=-1)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

#中心差分
def hyperbolic_windward_2(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为中心差分格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_central(pde.a(), tau)
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
#带粘性项
def hyperbolic_windward_with_vicious(n, *fargs):
    """
    @brief 时间步进格式为带粘性项的显式迎风格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.hyperbolic_operator_explicity_upwind_with_viscous(pde.a(), tau)
        #print(A)
        uh0[:] = A@uh0

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t,e

#显式迎风格式
"""
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_windward, frames=nt+1)
plt.show()
"""
err = []
t_list = []
for i in range(1,nt+1):
    u , t ,e= hyperbolic_windward(i)
    err.append(e)
    t_list.append(t)
    #fig, axes = plt.subplots(2, 1)

fig=plt.figure(figsize=(8, 4))
plt.plot(t_list, err,c='r',alpha=0.5)
plt.plot()
plt.show()
print(statistics.mean(err))

"""
#积分守恒
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_lax, frames=nt+1)
plt.show()


#中心差分
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_windward_2, frames=nt+1)
plt.show()


#带粘性项
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, hyperbolic_windward_with_vicious, frames=nt+1)
plt.show()
"""
err = []
t_list = []
for i in range(1,nt+1):
    u , t ,e= hyperbolic_windward_with_vicious(i)
    err.append(e)
    t_list.append(t)
    #fig, axes = plt.subplots(2, 1)

fig=plt.figure(figsize=(8, 4))
plt.plot(t_list, err,c='r',alpha=0.5)
plt.plot()
plt.show()
print(statistics.mean(err))


