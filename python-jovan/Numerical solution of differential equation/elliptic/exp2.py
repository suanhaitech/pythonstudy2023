import numpy as np
from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from scipy.sparse.linalg import spsolve
import time
from scipy.sparse import diags
class SinPDEData: # 可以点击这里查看FEALPy仓库中的代码。
    def domain(self):
        """
        @brief 得到 PDE 模型的区域

        @return: 表示 PDE 模型的区域的列表
        """
        return [0, 1]
        
    @cartesian
    def solution(self, p):
        """
        @brief 计算 PDE 模型的精确解
        
        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点的精确解
        """
        return np.sin(4*np.pi*p)
        
    @cartesian
    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项 

        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点处的源项
        """
        return 16*np.pi**2*np.sin(4*np.pi*p)
    
    @cartesian
    def gradient(self, p):
        """
        @brief: 计算 PDE 模型的真解的梯度

        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点处真解的梯度
        """
        return 4*np.pi*np.cos(4*np.pi*p)
    
    @cartesian    
    def dirichlet(self, p):
        """
        @brief: 模型的 Dirichlet 边界条件
        """
        return self.solution(p)
pde = SinPDEData()
domain = pde.domain()
nx = 100
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
start_time = time.perf_counter()
h=hx
cx = 1/(h**2)
A = diags([-cx, 2*cx, -cx ], [-1, 0, 1], shape=(NN, NN), format='csr')
print(A)
bytes = A.dtype.itemsize * A.size
print(bytes)
f = mesh.interpolate(pde.source, 'node')
node = mesh.node
uh = mesh.function()
A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f)
uh[:] = spsolve(A, f)
end_time = time.perf_counter()
fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes, uh)

print('B used time is: {:.6f}s\n'.format(end_time - start_time))
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{1}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))
