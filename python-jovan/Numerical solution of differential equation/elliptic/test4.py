from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
from typing import List 
import numpy as np
from fealpy.pde.elliptic_1d import SinPDEData 
from scipy.sparse import csr_matrix
#类
class SinPDEData:
    def domain(self) -> List[int]:
        """
        @brief 得到 PDE 模型的区域

        @return: 表示 PDE 模型的区域的列表
        """
        return [-1 , 1]
    @cartesian
    def solution(self, p) -> np.ndarray:
        """
        @brief 计算 PDE 模型的精确解
        
        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点的精确解
        """
        return (np.e**(-p**2))*(1-p**2)
    @cartesian
    def source(self, p) -> np.ndarray:
        """
        @brief: 计算 PDE 模型的原项 

        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点处的源项
        """
        return (np.e**(-p**2))*(4*p**4-16*p**2+6)
    @cartesian
    def gradient(self, p) -> np.ndarray:
        """
        @brief: 计算 PDE 模型的真解的梯度

        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点处真解的梯度
        """
        return (np.e**(-p**2))*(-4*p**4+14*p**2-4)
    @cartesian
    def dirichlet(self, p) -> np.ndarray:
        """
        @brief: 模型的 Dirichlet 边界条件
        """
        return self.solution(p)
pde = SinPDEData()
domain = pde.domain()
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
def laplace_operator(self) -> csr_matrix: # 可以点击这里查看 FEALPy 仓库中的代码   
    """
    @brief 组装 Laplace 算子 ∆u 对应的有限差分离散矩阵

    @note 并未处理边界条件
    """
    h = self.h
    cx = 1/(h**2)
    NN = self.number_of_nodes()
    k = np.arange(NN)

    A = diags([2*cx], [0], shape=(NN, NN), format='csr')

    val = np.broadcast_to(-cx, (NN-1, ))
    I = k[1:]
    J = k[0:-1]
    A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
    A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)
    return A
    
A = mesh.laplace_operator()
# 处理特殊情况
B = diags([2],[0],shape=A.shape)
A = A+B
uh = mesh.function()
f = mesh.interpolate(pde.source, 'node')
A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f)

uh[:] = spsolve(A, f)
fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes, uh)

et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{1}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)

maxit = 5
em = np.zeros((len(et), maxit), dtype=np.float64)
egradm = np.zeros((len(et), maxit), dtype=np.float64)
for i in range(maxit):
    A = mesh.laplace_operator()+diags([2],[0],shape=mesh.laplace_operator().shape)
    f = mesh.interpolate(pde.source, 'node')
    uh = mesh.function()
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f)
    uh[:] = spsolve(A, f)
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)
    if i < maxit:
        mesh.uniform_refine()
print("em:\n", em)
print("\n")
print("em_ratio:", em[:, 0:-1]/em[:, 1:])

