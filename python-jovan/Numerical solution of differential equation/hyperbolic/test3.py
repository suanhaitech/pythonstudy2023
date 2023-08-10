# 积分守恒型 Lax - Friedrichs 格式
# 导入所需要的模块
import numpy as np
from fealpy.decorator import cartesian
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from typing import Union,Tuple,List    # Union: 将多个集合合并为一个集合
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d

# 创建类
class Hyperbolic1dPDEData:

    def __init__(self,D: Union[Tuple[int,int],List[int]] = (0,2),T: Union[Tuple[int, int], List[int]] = (0, 4)):
        self._domain = D
        self._duration = T

    def domain(self) -> Union[Tuple[float,float],List[float]]:
        return self._domain

    def duration(self) -> Union[Tuple[float,float],List[float]]:
        return self._duration

    @cartesian
    def solution(self,p: np.ndarray,t: np.float64) -> np.ndarray:
        val = np.zeros_like(p)
        flag1 = p <= t
        flag2 = p > t+1
        flag3 = ~flag1 & ~flag2

        val[flag1] = 1
        val[flag3] = 1 - p[flag3] + t
        val[flag2] = p[flag2] - t -1
        return val

    @cartesian
    def init_solution(self,p: np.ndarray) -> np.ndarray:
        val = np.zeros_like(p)
        val = np.abs(p-1)
        return val

    @cartesian
    def source(self,p: np.ndarray, t: np.float64) -> np.ndarray:
        return 0.0

    @cartesian
    def dirichlet(self,p: np.ndarray,t: np.float64) -> np.ndarray:
        return np.ones(p.shape)

    def a(self) -> np.float64:
        return 1

# 创建对象
pde =Hyperbolic1dPDEData()

# 空间离散
domain = pde.domain()
nx = 40
hx = (domain[1] - domain[0]) / nx
mesh = UniformMesh1d([0,nx], h = hx, origin = domain[0])

# 时间离散
duration = pde.duration()
nt = 3200
tau = (duration[1] - duration[0]) / nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype = 'node')

# 组装矩阵
def hyperbolic_operator_central(mesh, tau, a):
    """
    @brief 生成双曲方程的中心差分迭代矩阵
    @param[in] tau float, 当前时间步长
    """
    r = a*tau/mesh.h
    
    NN = mesh.number_of_nodes()
    k = np.arange(NN)
    
    A = diags([1], [0], shape=(NN, NN), format='csr')
    val = np.broadcast_to(-r/2, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=mesh.ftype)
    A += csr_matrix((-val, (J, I)), shape=(NN, NN), dtype=mesh.ftype)
    return A


# 时间步进
def hyperbolic_windward(n, *fargs): # 点击这里查看 FEALPy 中的代码
    """
    @brief 时间步进格式为中心差分格式

    @param[in] n int, 表示第 `n` 个时间步（当前时间步） 
    """
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = hyperbolic_operator_central(mesh,pde.a(), tau)
        uh0[:] = A@uh0

        c
        mesh.update_dirichlet_bc(gD, uh0, threshold=0)

        return uh0, t
  
# 制作动画
box = [0, 2, 0, 2]
fig, axes = plt.subplots()
mesh.show_animation(fig, axes, box,hyperbolic_windward, fname='hyperbolic_windward_central.mp4', frames=nt+1,color='green')
plt.show()
