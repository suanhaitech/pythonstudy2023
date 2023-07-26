from fealpy.decorator import cartesian
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.parabolic_1d import SinExpPDEData
from fealpy.mesh import UniformMesh1d

class HeatConductionPDEData:

    def __init__(self, D=[0, 1], T=[0, 1], k=1):
        """
        @brief 模型初始化函数
        @param[in] D 模型空间定义域
        @param[in] T 模型时间定义域
        @param[in] k 热传导系数
        """
        self._domain = D
        self._duration = T
        self._k = k
        self._L = D[1] - D[0] 

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
    def solution(self, p, t):
        """
        @brief 真解函数

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 真解函数值
        """
        return np.exp(-(np.pi**2) * t / self._L**2) * np.sin(np.pi * p / self._L)

    @cartesian
    def init_solution(self, p):
        """
        @brief 初始解函数

        @param[in] x float, 空间点

        @return 初始解函数值
        """
        return np.sin(np.pi * p / self._L)

    @cartesian
    def source(self, p, t):
        """
        @brief 方程右端项

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 方程右端函数值
        """
        return 0

    @cartesian
    def gradient(self, p, t):
        """
        @brief 真解空间导数

        @param[in] p float, 空间点
        @param[in] t float, 时间点

        @return 真解空间导函数值
        """
        return (np.pi / self._L) * np.exp(-(np.pi**2) * t / self._L**2) * np.cos(np.pi * p / self._L)

    @cartesian
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p float, 空间点
        @param[in] t float, 时间点
        """
        return self.solution(p, t)

pde = HeatConductionPDEData()

# 空间离散
domain = pde.domain()
nx = 40
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 3200
tau = (duration[1] - duration[0])/nt

#准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype='node')

def parabolic_operator_forward(self, tau):
    """
    @brief 生成抛物方程的向前差分迭代矩阵

    @param[in] tau float, 当前时间步长
    """
    r = tau/self.h**2 
    if r > 0.5:
        raise ValueError(f"The r: {r} should be smaller than 0.5")

    NN = self.number_of_nodes()
    k = np.arange(NN)

    A = diags([1 - 2 * r], [0], shape=(NN, NN), format='csr')

    val = np.broadcast_to(r, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)
    return A

def parabolic_operator_backward(self, tau): # 点击这里可查看 FEALPy 仓库中的代码
    """
    @brief 生成抛物方程的向后差分迭代矩阵

    @param[in] tau float, 当前时间步长
    """
    r = tau/self.h**2 

    NN = self.number_of_nodes()
    k = np.arange(NN)

    A = diags([1+2*r], [0], shape=(NN, NN), format='csr')

    val = np.broadcast_to(-r, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)
    return A

def parabolic_operator_crank_nicholson(self, tau):# 点击这里可查看 FEALPy 仓库中的代码
    """
    @brief 生成抛物方程的 CN 差分格式的迭代矩阵

    @param[in] tau float, 当前时间步长
    """
    r = tau/self.h**2

    NN = self.number_of_nodes()
    k = np.arange(NN)

    A = diags([1 + r], [0], shape=(NN, NN), format='csr')
    val = np.broadcast_to(-r/2, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

    B = diags([- 1 - r], [0], shape=(NN, NN), format='csr')
    B += csr_matrix((-val, (I, J)), shape=(NN, NN), dtype=self.ftype)
    B += csr_matrix((-val, (J, I)), shape=(NN, NN), dtype=self.ftype)
    return A, B

def advance_forward(n, *fargs): 
    """
    @brief 时间步进格式为向前欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f

        gD = lambda p: pde.dirichlet(p, t)
        mesh.update_dirichlet_bc(gD, uh0)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        return uh0, t

def advance_backward(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为向后欧拉方法

    @param[in] n int, 表示第 `n` 个时间步（当前时间步）
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += uh0

        gD = lambda p: pde.dirichlet(p, t)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')

        return uh0, t

def advance_crank_nicholson(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为 CN 方法
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A, B = mesh.parabolic_operator_crank_nicholson(tau)
        source = lambda p: pde.source(p, t)
        f = mesh.interpolate(source, intertype='node')
        f *= tau
        f += B@uh0

        gD = lambda p: pde.dirichlet(p, t)
        A, f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A, f)

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')

        return uh0, t

fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5] # 图像显示的范围 0 <= x <= 1, -1.5 <= y <= 1.5

#mesh.show_animation(fig, axes, box, advance_forward, fname='parabolic_1d_forward_1.mp4', frames=nt + 1)
mesh.show_animation(fig, axes, box, advance_backward, fname='parabolic_1d_backward_1.mp4', frames=nt + 1)
#mesh.show_animation(fig, axes, box, advance_crank_nicholson, fname='parabolic_1d_crank_nicholson_1.mp4', frames=nt + 1)
plt.show()


