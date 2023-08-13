import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
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
        val = 1 + np.sin(2 * np.pi * (p + 2 * t))
        
        return val

    @cartesian
    def init_solution(self, p: np.ndarray) -> np.ndarray:
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点

        @return 真解函数值
        """
        val = 1 + np.sin(2 * np.pi * p)
        
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
        return 1 + np.sin(4 * np.pi * t)
        
    def a(self) -> np.float64:
        return -2

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

def lax_wendroff(n, *fargs):
    """
    @brief 时间步进格式为迎风格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n * tau
    if n == 0:
        e = 0
        return uh0, t,e
    else:
        # 构造矩阵

        A = mesh.hyperbolic_operator_lax_wendroff(pde.a(), tau)

        u0, u1 = uh0[[0, 1]]
        uh0[:] = A @ uh0
        
        gD = lambda p, t=t: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0, threshold=-1)

        # 更新数值边界条件，三选一
        #uh0[0] = u0 + 2 * tau * (u1 - u0) / hx
        #uh0[0] = uh0[1]
        uh0[0] = 2*uh0[1] - uh0[2]

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t ,e

"""
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, lax_wendroff, fname='hyperbolic1_test_2.mp4', frames=nt+1,color='yellow')
plt.show()
"""
err = []
t_list = []
for i in range(1,nt+1):
    u , t ,e= lax_wendroff(i)
    err.append(e)
    t_list.append(t)
    #fig, axes = plt.subplots(2, 1)
    
fig=plt.figure(figsize=(8, 4))
plt.plot(t_list, err,c='r',alpha=0.5)
plt.plot()
plt.show()
"""
        #true_solution = pde.solution(x, t)
        # 计算误差
        # E = mesh.error(true_solution, u)

        error = true_solution - u
        print(f"At time {t}, Error: {error}")

        # 画出数值解和真解
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

#print(err)
err=[]
for i in range(nt+1):
    u, t ,e= lax_wendroff(i)
    err.append(e)
    if t in [0.5, 1.0, 1.5, 2.0]:
        fig, axes = plt.subplots(2, 1)
        x = mesh.entity('node').reshape(-1)
        true_solution = pde.solution(x, t)
        # 计算误差
        # E = mesh.error(true_solution, u)
        error = true_solution - u
        print(f"At time {t}, Error: {error}")

        # 画出数值解和真解
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
print(err)
"""
