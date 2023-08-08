import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
from fealpy.pde.elliptic_1d import ExpPDEData
from scipy.sparse.linalg import spsolve

# pde模型
pde = ExpPDEData()
domain = pde.domain()


# 网格剖分
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

# # 插值网络
# # 将十个点处的离散解插值到网格节点上
# uI = mesh.interpolate(pde.solution, 'node')

# fig = plt.figure()
# axes = fig.gca()
# # show_function 函数在网格上绘制插值函数
# mesh.show_function(axes, uI)
# plt.show()

# # 画出真解的图像
# x = np.linspace(domain[0], domain[1], 100)

# u = pde.solution(x)

# fig = plt.figure()
# axes = fig.gca()
# axes.plot(x, u)
# plt.show()

# 矩阵组装
A = mesh.laplace_operator()

# 处理边界条件
uh = mesh.function()
f = mesh.interpolate(pde.source, 'node')

A, f = mesh.apply_dirichlet_bc(pde.dirichlet, A, f)

# 离散系统求解
uh[:] = spsolve(A, f)

fig = plt.figure()
axes = fig.gca()
mesh.show_function(axes, uh)
plt.show()

# 计算误差
et = ['$|| u - u_h||_{\infty}$', '$|| u - u_h||_{0}$', '$|| u - u_h ||_{1}$']
eu = np.zeros(len(et), dtype=np.float64) 
eu[0], eu[1], eu[2] = mesh.error(pde.solution, uh)
et = np.array(et)
print(np.vstack((et, eu)))

# 测试收敛阶
maxit = 5 
em = np.zeros((len(et), maxit), dtype=np.float64)
egradm = np.zeros((len(et), maxit), dtype=np.float64) 

for i in range(maxit):
    A = mesh.laplace_operator() 
    uh = mesh.function() 
    f = mesh.interpolate(pde.source, 'node')
    A, f = mesh.apply_dirichlet_bc(gD=pde.dirichlet, A=A, f=f)
    uh[:] = spsolve(A, f) 

    em[0, i], em[1, i], em[2, i] = mesh.error(pde.solution, uh)

    if i < maxit:
        mesh.uniform_refine()

print("em:\n", em)
print("em_ratio:", em[:, 0:-1]/em[:, 1:])

