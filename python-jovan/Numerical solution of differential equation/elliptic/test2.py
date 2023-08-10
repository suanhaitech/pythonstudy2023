from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
import matplotlib.pyplot as plt
from fealpy.pde.elliptic_1d import SinPDEData 
#实例
pde = SinPDEData()
domain = pde.domain()
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
