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
#实例
pde = SinPDEData()
domain = pde.domain()

#测试
print('domain :', domain)
print(np.abs(pde.solution(0)-1)< 1e-12)
print(np.abs(pde.solution(1)) < 1e-12)
print(np.abs(pde.solution(-1)) < 1e-12)
print(np.abs(pde.source(0) -6) < 1e-12)
