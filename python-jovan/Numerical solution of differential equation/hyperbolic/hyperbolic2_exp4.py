import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh2d
from typing import List,Tuple,Union
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from fealpy.decorator import cartesian
from scipy.sparse import diags
class Hyperbolic2dPDEData: # 点击这里可以查看 FEALPy 中的代码

    def __init__(self, D=[0, 2, 0, 2], T=[0, 4]):
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

    
    def solution(self, p, t):
        """
        @brief 真解函数

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 返回 val
        """
        x, y = p[..., 0], p[..., 1]
        val = np.zeros_like(p[...,0])

        m = x*y
        nx , ny = m.shape
        for i in range(nx):
            for j in range(ny):
                if m[i,j] <= t:
                    val[i,j]= 1.0
                elif m[i,j] > t+1:
                    val[i,j] = m[i,j] - t - 1.0
                else:
                    val[i,j] = 1.0 - m[i,j] +t
        return val
    
    def source(self, p, t):
        """
        @brief 方程右端项 
        
        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 0
        """
        return np.zeros_like(p[..., 0])

    
    def init_solution(self, p):
        """
        @brief 初值条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 返回 val
        """

        x, y = p[..., 0], p[..., 1]
        val = np.zeros_like(p[...,0])
        m = x*y
        val = np.abs(m-1)
        return val

    
    def init_solution_diff_t(self, p):
        """
         @brief 初值条件的导数

         @param[in] p numpy.ndarray, 空间点
        """
        return np.zeros_like(p[..., 0])

    
    def dirichlet(self, p, t):
        """
        @brief Dirichlet 边界条件

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 边界条件函数值
        """
        return np.ones_like(p[..., 0])


        
    def a(self) -> np.float64:
        return 1
pde = Hyperbolic2dPDEData()
#空间离散
domain = pde.domain()
nx = 40
ny = 40
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(domain[0], domain[2]))

#时间离散
duration = pde.duration()
nt = 3200 
tau = (duration[1] - duration[0])/nt 
uh0 = mesh.interpolate(pde.init_solution, intertype='node')
def hyperbolic_operator_explicity_lax_friedrichs(mesh, tau, a):
    """
    @brief 积分守恒型 lax_friedrichs 格式
    """
    rx = a*tau/mesh.h[0]
    ry = a*tau/mesh.h[1]

    if rx + ry > 1.0:
        raise ValueError(f"The r: {r} should be smaller than 1.0")

    NN = mesh.number_of_nodes()

    n0 = mesh.nx + 1
    n1 = mesh.ny + 1
    k = np.arange(NN).reshape(n0, n1)
    A = diags([0], [0], shape=(NN, NN), format='csr')
    val0 = np.broadcast_to(1/2 *(1+ rx), (NN-n1, ))
    val1 = np.broadcast_to(1/2 *(1- rx), (NN-n1, ))
    I = k[1:, :].flat
    J = k[0:-1, :].flat
    A += csr_matrix((val0, (J, I)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val1, (I, J)), shape=(NN, NN), dtype=mesh.ftype)

    val0 = np.broadcast_to(1/2 *(1+ ry), (NN-n0, ))
    val1 = np.broadcast_to(1/2 *(1- ry), (NN-n0, ))
    I = k[1:, :].flat
    J = k[0:-1, :].flat
    A += csr_matrix((val0, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((val1, (J, I)), shape=(NN, NN), dtype=mesh.ftype)

    return A
def hyperbolic_lax(n, *fargs):
    """
    @brief 时间步进格式为守恒型 Lax 格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = hyperbolic_operator_explicity_lax_friedrichs(mesh,tau,a=1)
        uh0.flat = A@uh0.flat

        gD = lambda p: pde.dirichlet(p, t+tau)
        mesh.update_dirichlet_bc(gD, uh0)
        solution = lambda p: pde.solution(p, t + tau)                  
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        
        return uh0, t
box = [0, 2, 0, 2, 0, 4]
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.show_animation(fig, axes, box, hyperbolic_lax, plot_type='surface', frames=nt+1)
plt.show()


