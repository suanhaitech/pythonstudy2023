from fealpy.mesh import UniformMesh1d
import matplotlib.pyplot as plt
from fealpy.pde.elliptic_1d import SinPDEData 
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

