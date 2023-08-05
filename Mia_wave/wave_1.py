import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from scipy.sparse.linalg import spsolve
from fealpy.pde.wave_1d import StringOscillationPDEData

class StringOscillationSinCosPDEData:
    def __init__(self, D = [0.0, 1.0], T= [0.0, 2.0]):
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

    def duration(self) :
        """
        @brief 时间区间
        """
        return self._duration 

    
    def source(self, p, t) :
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        return 0.0

    def solution(self, x,t) :
        """
        @brief 真解函数

        @param[in] x numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 真解函数值
        """
        return (np.sin(np.pi*(x - t)) + np.sin(np.pi*(x + t))) / 2 - (np.sin(np.pi*(x - t)) - np.sin(np.pi*(x + t))) / (2 * np.pi)

    def init_solution(self, x):
        """
        @brief 初始条件函数

        @param[in] x numpy.ndarray, 空间点

        @return 初始条件函数值
        """
        return np.sin(np.pi * x)

    def init_solution_diff_t(self, x):
        """
        @brief 初始条件时间导数函数(初始速度条件)

        @param[in] x numpy.ndarray, 空间点

        @return 初始条件时间导数函数值
        """
        return np.cos(np.pi * x)

    #def dirichlet(self, x: np.ndarray, t: np.float64) -> np.ndarray:
    #    """
    #    @brief Dirichlet
    #    """
     #   val = np.zeros_like(x)
      #  val[0] = np.sin(np.pi * t) / np.pi
       # val[-1] = -np.sin(np.pi * t) / np.pi
        #return val
 

    def dirichlet(self, x, t) :
        """
        @brief Dirichlet
        """
        if t is None:
            t = 1.0  # 或者其他适当的默认值
        val = np.zeros_like(x)
        val[0] = np.sin(np.pi * t) / np.pi
        val[-1] = -np.sin(np.pi * t) / np.pi
        return val

pde = StringOscillationSinCosPDEData()

# 空间离散
domain = pde.domain()
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 20
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, 'node')
vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node')
uh1 = mesh.function('node')

def advance_explicit(n, *frags):
    """
    @brief 时间步进格式为显格式

    @param[in] n int, 表示第 n 个时间步 
    """
    a = 1
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx 
        uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        A = mesh.wave_operator_explicit(tau, a)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        uh2 = A@uh1 - uh0 + f  # 根据差分方程进行更新

        uh0[:] = uh1[:]
        uh1[:] = uh2
        gD = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh1)
            
        return uh1, t

def advance_implicit(n, theta=0.25, *frags):
    """
    @brief 时间步进格式为隐格式

    @param[in] n int, 表示第 n 个时间步（当前时间步）
    """
    # theta = 0.25
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    elif n == 1:
        rx = tau/hx
        uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh1)
        return uh1, t
    else:
        A0, A1, A2 = mesh.wave_operator_implicit(tau, theta=theta)
        source = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        f *= tau**2
        f += A1@uh1 + A2@uh0

        uh0[:] = uh1[:]
        gD = lambda p: pde.dirichlet(p, t + tau)
        A0, f = mesh.apply_dirichlet_bc(gD, A0, f)
        uh1[:] = spsolve(A0, f)

        return uh1, t
"""
box = [0, 1, -2.0, 2.0]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_explicit, fname='explicit.mp4', frames=nt+1)
plt.show()

from functools import partial

advance_implicit_theta = partial(advance_implicit, theta=0.25)

box = [0, 1, 2.0, -2.0]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box, advance_implicit_theta, fname='implicit.mp4', frames=nt+1)
plt.show()
"""
"""
for i in range(nt+1):
    u, _ = advance_implicit(i)
    if (duration[0] + i*tau) in [0.5, 1.0, 1.5, 2.0]:
        fig, axes = plt.subplots()
        x = mesh.entity('node').flat
        axes.plot(x, u)
        axes.set_ylim(-2, 2)
        axes.set_title(f't = {duration[0] + i*tau}')
        plt.show()
"""
for i in range(nt+1):
    u, t = advance_implicit(i)
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
