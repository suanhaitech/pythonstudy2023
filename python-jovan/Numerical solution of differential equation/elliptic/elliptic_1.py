import numpy as np

from fealpy.decorator import cartesian

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

# 测试
print('domain :', domain)
print(np.abs(pde.solution(0)) < 1e-12)
print(np.abs(pde.solution(1)) < 1e-12)
print(np.abs(pde.solution(1/8) - 1) < 1e-12)
