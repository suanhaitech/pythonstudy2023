import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.pde.elliptic_2d import CosCosPDEData
from scipy.sparse.linalg import spsolve

# pde模型
pde = CosCosPDEData()
domain = pde.domain()

# 网格剖分
nx = 5
ny = 5
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny

mesh = UniformMesh2d((0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=12, fontcolor='r')
mesh.find_edge(axes, showindex=True, fontsize=12, fontcolor='g') 
mesh.find_cell(axes, showindex=True, fontsize=12, fontcolor='b')
# plt.show()

# # 将36个点处的离散解插值到网格节点上
# uI = mesh.interpolate(pde.solution, 'node')

# # 创建一个 figure，并设置当前坐标轴已进行绘图
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')

# # show_function 函数在网格上绘制插值函数
# mesh.show_function(axes, uI)
# plt.show()

# # 画出真解的图像
# x = np.linspace(0, 1, 101)
# y = np.linspace(0, 1, 101)
# X, Y = np.meshgrid(x, y)
# p = np.array([X, Y]).T
# Z = pde.solution(p)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, Y, Z, cmap='jet')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.show()

# 矩阵组装
A = mesh.laplace_operator()

# 将稀疏矩阵转换成稠密矩阵输出
print(A.toarray())

# 处理边界条件
uh = mesh.function()
f = mesh.interpolate(pde.source, 'node')

A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
print(A.toarray())
print(f)

# 离散系统求解
uh = mesh.function()
uh.flat[:] = spsolve(A, f) # 返回网格节点处的数值解

fig = plt.figure(4)
axes = fig.add_subplot(111, projection='3d')
mesh.show_function(axes, uh.reshape(6, 6))
plt.title("Numerical solution after processing the boundary conditions")
plt.show()

# 计算误差
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{l2}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))
print("----------------------------------------------------------------------")

# 测试收敛阶
maxit = 5
em = np.zeros((3, maxit), dtype=np.float64)

for i in range(maxit):
    A = mesh.laplace_operator() 
    uh = mesh.function()
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f)
    uh.flat[:] = spsolve(A, f)  
    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

    if i < maxit:
        mesh.uniform_refine()

print("em_ratio:", em[:, 0:-1]/em[:, 1:])