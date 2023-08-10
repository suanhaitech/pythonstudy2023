import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation

import numpy as np
from typing import List, Optional
from fealpy.decorator import cartesian

class StringOscillationPDEData: # 点击这里可以查看 FEALPy 代码
    def __init__(self, D = [0, 1], T = [0, 2]):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        """
        self._domain = D 
        self._duration = T 

    def domain(self):
        """
        @brief 空间区间
        """
        return self._domain

    def duration(self):
        """
        @brief 时间区间
        """
        return self._duration 
        
    @cartesian
    def solution(self, x: np.ndarray, t: np.float64) -> np.ndarray:
        """
        @brief 真解函数

        @param[in] x numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        return (np.sin(np.pi*(x - t)) + np.sin(np.pi*(x + t))) / 2 - (np.sin(np.pi*(x - t)) - np.sin(np.pi*(x + t))) / (2 * np.pi)
        
    @cartesian
    def init_solution(self, p):
        """
        @brief 初始解

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解函数值
        """
        pi=np.pi
        return np.sin(pi*p)

    @cartesian
    def init_solution_diff_t(self, p):
        """
        @brief 初始解的时间导数函数(初始速度条件)

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 初始解时间导数函数值
        """
        pi=np.pi
        return np.cos(pi*p)

    @cartesian
    def source(self, p, t):
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点

        @return 方程右端函数值
        """
        return np.zeros_like(p)
    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet
        """
        if t is None:
            t = 1.0  # 或者其他适当的默认值
        pi=np.pi
        val = np.zeros_like(p)
        val[0] = np.sin(pi * t) / pi
        val[-1] = -np.sin(pi * t) / pi
        return val
        
pde = StringOscillationPDEData()
# 空间离散
domain = pde.domain()
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt =20
tau = (duration[1] - duration[0])/nt
# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')
def advance(n, *frags):
    """
    @brief 时间步进

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    theta = 0.
    t = duration[0] + n*tau
    if n == 0: # 第 0 层
        return uh0, t
    elif n == 1: # 第 1 层
        rx = tau/hx 
        uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
        gD = lambda p, t: pde.dirichlet(p, t)
        print("gD:", gD)
        mesh.update_dirichlet_bc(gD, uh1, t)
        return uh1, t
    else:
        A0, A1, A2 = mesh.wave_operator(tau, theta=theta) # 差分格式为显格式
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f += A1@uh1 + A2@uh0

        uh0[:] = uh1[:]
        gD = lambda p, t: pde.dirichlet(p, t+tau)
        if theta == 0.0:
            uh1[:] = f
            mesh.update_dirichlet_bc(gD, uh1, t)
        else:
            A0, f = mesh.apply_dirichlet_bc(gD, A0, f)
            uh1[:] = spsolve(A0, f)
            
        return uh1, t

for i in range(nt+1):
    u, _ = advance(i)
    if (duration[0] + i*tau) in [0.5, 1.0, 1.5, 2.0]:
        fig, axes = plt.subplots()
        x = mesh.entity('node').flat
        axes.plot(x, u)
        axes.set_ylim(-2, 2)
        axes.set_title(f't = {duration[0] + i*tau}')
        plt.show()
        
for i in range(nt+1):
    u, t = advance(i)
    if t in [0.5, 1.0, 1.5, 2.0]:
        fig, axes = plt.subplots(2, 1)
        x = mesh.entity('node').reshape(-1)
        true_solution = pde.solution(x,t)
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
