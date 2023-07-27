# 向前欧拉公式
# 导入所需要的库
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d
from fealpy.decorator import cartesian

# 创建类
class HeatConductionPDEData:

    def __init__(self, D=[0, 1], T=[0, 1], k=1):
        self._domain = D
        self._duration = T
        self._k = k
        self._L = D[1] - D[0] 

    def domain(self):
        return self._domain

    def duration(self):
        return self._duration

    @cartesian
    def solution(self, p, t):
        return np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.sin(np.pi * p / self._L)

    @cartesian
    def init_solution(self, p):
        return np.sin(np.pi * p / self._L)

    @cartesian
    def source(self, p, t):
        return  0
        
    @cartesian
    def gradient(self, p, t):
        return (np.pi / self._L) * np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.cos(np.pi * p / self._L)

    @cartesian
    def dirichlet(self, p, t):
        return self.solution(p, t)

# 创建一个对象
pde = HeatConductionPDEData()
# 时间与空间离散(网格剖分)
# 空间离散
domain = pde.domain()
nx = 10
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 3200
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype= 'node')

# 向前欧拉
def advance_forward(n, *fargs):
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_forward(tau)
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype='node')
        uh0[:] = A@uh0 + tau*f

        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        mesh.update_dirichlet_bc(gD, uh0)

        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype='max')
        print(f"the max error is {e}")
        return uh0, t
# 制作动画
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5]
mesh.show_animation(fig, axes, box, advance_forward, fname='advance_forward.mp4', frames=nt+1)
plt.show()



# 向前欧拉公式
# 导入所需要的库
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d
from fealpy.decorator import cartesian


# 创建类
class HeatConductionPDEData:

    def __init__(self, D=[0, 1], T=[0, 1], k=1):
        self._domain = D
        self._duration = T
        self._k = k
        self._L = D[1] - D[0]

    def domain(self):
        return self._domain

    def duration(self):
        return self._duration

    @cartesian
    def solution(self, p, t):
        return np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.sin(np.pi * p / self._L)

    @cartesian
    def init_solution(self, p):
        return np.sin(np.pi * p / self._L)

    @cartesian
    def source(self, p, t):
        return  0

    @cartesian
    def gradient(self, p, t):
        return (np.pi / self._L) * np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.cos(np.pi * p / self._L)

    @cartesian
    def dirichlet(self, p, t):
        return self.solution(p, t)

# 创建一个对象
pde = HeatConductionPDEData()
# 时间与空间离散(网格剖分)
# 空间离散
domain = pde.domain()
nx = 20
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 3200
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype= 'node')

# 向后欧拉公式
def advance_backward(n, *fargs):
    t = duration[0] + n*tau
    if n== 0:
        return uh0, t
    else:
        A = mesh.parabolic_operator_backward(tau)
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype = 'node')
        f*= tau
        f+= uh0

        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        A ,f = mesh.apply_dirichlet_bc(gD, A, f)
        uh0[:] = spsolve(A,f)

        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype = 'max')
        print(f"the max error is {e}")

        return uh0, t
# 制作动画
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5]
mesh.show_animation(fig, axes, box, advance_backward,fname='advance_backward.mp4', frames=nt + 1,lw=2, interval=50, linestyle='--', color='red')
plt.show()



# # Crank-Nicholson 离散格式
# 导入所需要的库
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh1d
from fealpy.decorator import cartesian

# 创建类
class HeatConductionPDEData:

    def __init__(self, D=[0, 1], T=[0, 1], k=1):
        self._domain = D
        self._duration = T
        self._k = k
        self._L = D[1] - D[0]

    def domain(self):
        return self._domain

    def duration(self):
        return self._duration

    @cartesian
    def solution(self, p, t):
        return np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.sin(np.pi * p / self._L)

    @cartesian
    def init_solution(self, p):
        return np.sin(np.pi * p / self._L)

    @cartesian
    def source(self, p, t):
        return  0

    @cartesian
    def gradient(self, p, t):
        return (np.pi / self._L) * np.exp(-self._k * (np.pi**2) * t / self._L**2) * np.cos(np.pi * p / self._L)

    @cartesian
    def dirichlet(self, p, t):
        return self.solution(p, t)

# 创建一个对象
pde = HeatConductionPDEData()
# 时间与空间离散(网格剖分)
# 空间离散
domain = pde.domain()
nx = 40
hx = (domain[1] - domain[0])/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=domain[0])

# 时间离散
duration = pde.duration()
nt = 3200
tau = (duration[1] - duration[0])/nt

# 准备初值
uh0 = mesh.interpolate(pde.init_solution, intertype= 'node')

# Crank-Nicholson 离散格式
def advance_crank_nicholson(n, *fargs):
    t = duration[0] + n*tau
    if n == 0:
        return uh0, t
    else:
        A,B = mesh.parabolic_operator_crank_nicholson(tau)
        source: Callable[[np.ndarray], np.ndarray] = lambda p: pde.source(p, t + tau)
        f = mesh.interpolate(source, intertype = 'node')
        f *= tau
        f += B@uh0

        gD: Callable[[np.ndarray], np.ndarray] = lambda p: pde.dirichlet(p, t + tau)
        A , f = mesh.apply_dirichlet_bc(gD, A,f)
        uh0[:] = spsolve(A, f)

        solution: Callable[[np.ndarray], np.ndarray] = lambda p: pde.solution(p, t + tau)
        e = mesh.error(solution, uh0, errortype = 'max')
        print(f"the max error is {e}")

        return uh0, t

# 制作动画
fig, axes = plt.subplots()
box = [0, 1, -1.5, 1.5]
mesh.show_animation(fig, axes, box, advance_crank_nicholson, frames=nt + 1)
plt.show()
