#情形b
import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.decorator import cartesian
from scipy.sparse import diags,csr_matrix
from fealpy.mesh.uniform_mesh_1d import UniformMesh1d
import matplotlib.pyplot as plt

class Hyperbolic1dPDEData:
    def __init__(self,D = [0,1] , T = [0,1]):
        self._domain = D
        self._duration = T

    def domain(self):

        return self._domain

    def duration(self):

        return self._duration

    @cartesian
    def solution(self, p: np.ndarray , t: np.float64) ->np.ndarray:

        pi = np.pi
        val = 1 + np.sin(2*pi*(p+2*t))

        return val

    @cartesian
    def init_solution(self,p: np.ndarray)->np.ndarray:

        pi = np.pi
        val = 1 + np.sin(2*pi*p)

        return val

    @cartesian
    def source(self,p: np.ndarray,t: np.float64)->np.ndarray:

        return np.zeros_like(p)

    @cartesian
    def dirichlet(self,p: np.ndarray,t: np.float64)->np.ndarray:

        return 1+ np.sin(4*np.pi*t)

    def a(self)->np.float64:

        return -2

#实例化
pde = Hyperbolic1dPDEData()
#离散
domain = pde.domain()
nx = 30
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0,nx],h = hx , origin = domain[0])
duration = pde.duration()
nt = 800
tau = (duration[1] - duration[0])/nt

uh0 = mesh.interpolate(pde.init_solution, intertype = 'node')

def Lax_Wendroff(mesh , tau ,a ):
   r = a*tau/hx
   if np.abs(r) > 1.0:
      raise ValueError(f"The r: {r} should be smaller than 1")
   NN = mesh.number_of_nodes()
   k = np.arange(NN)

   A = diags([1-r**2], 0, shape = (NN,NN), format = 'csr')
   val0 = np.broadcast_to(1/2 * r *(r+1),(NN-1, ))
   val1 = np.broadcast_to(1/2 * r *(r-1),(NN-1, ))
   I = k[1:]
   J = k[0:-1]
   A += csr_matrix((val0,(I,J)),shape = (NN,NN),dtype = mesh.ftype)
   A += csr_matrix((val1,(J,I)),shape = (NN,NN),dtype = mesh.ftype)

   return A

def hyperbolic_lax_wendroff(n, *fargs):
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = Lax_Wendroff(mesh, tau,pde.a())
        u0 , u1 = uh0[[0,1]]
        uh0[:] = A@uh0
        gD = lambda p, t = t: pde.dirichlet(p,t)
        mesh.update_dirichlet_bc(gD,uh0,threshold = -1)
        uh0[0] = uh0[1]

        solution = lambda p: pde.solution(p, t)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t

box = [0,1,0,2]
fig, axes = plt.subplots()
mesh.show_animation(fig,axes,box,hyperbolic_lax_wendroff,frames = nt+1)
plt.show()
