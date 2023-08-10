from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
import numpy as np

from fealpy.decorator import cartesian
from typing import List 

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
# 将十个点处的离散解插值到网格节点上
uI = mesh.interpolate(pde.solution, 'node')

fig = plt.figure()
axes = fig.gca()
# show_function 函数在网格上绘制插值函数
mesh.show_function(axes, uI)
plt.show()
# 画出真解的图像
x = np.linspace(domain[0], domain[1], 100)

u = pde.solution(x)

fig = plt.figure()
axes = fig.gca()
axes.plot(x, u)
plt.show()
