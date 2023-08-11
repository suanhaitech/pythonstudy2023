import numpy as np
from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
from fealpy.decorator import cartesian
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import time
class SinPDEData: # 可以点击这里查看FEALPy仓库中的代码。
    def domain(self):
        """
        @brief 得到 PDE 模型的区域

        @return: 表示 PDE 模型的区域的列表
        """
        return [-1, 1]
        
    @cartesian
    def solution(self, p):
        """
        @brief 计算 PDE 模型的精确解
        
        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点的精确解
        """
        return (np.e**(-p**2))*(1-p**2)
        
    @cartesian
    def source(self, p):
        """
        @brief: 计算 PDE 模型的原项 

        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点处的源项
        """
        return (np.e**(-p**2))*(4*p**4-14*p**2+4)
    
    @cartesian
    def gradient(self, p):
        """
        @brief: 计算 PDE 模型的真解的梯度

        @param p: 自变量 x 的数组
        @return: PDE 模型在给定点处真解的梯度
        """
        return (np.e**(-p**2))*(2*p**3-4*p)
    
    @cartesian    
    def dirichlet(self, p):
        """
        @brief: 模型的 Dirichlet 边界条件
        """
        return self.solution(p)
        
pde = SinPDEData()
domain = pde.domain()
#定义数据
nx = 100
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
#函数在网格上绘制插值函数
uI = mesh.interpolate(pde.solution, 'node')
fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes, uI)
plt.show()
#计算运行时间
start_time = time.perf_counter()
#矩阵组装
A = mesh.laplace_operator()
f = mesh.interpolate(pde.source, 'node')
node = mesh.node
uh = mesh.function()
B=diags([2], [0], shape=(NN, NN), format='csr')
A=A+B
A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f)
#数值求解
uh[:] = spsolve(A, f)
end_time = time.perf_counter()
#绘制结果图
fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes, uh)
plt.show()
print('作业 used time is: {:.6f}s\n'.format(end_time - start_time))
#误差分析
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{1}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack(eu))
