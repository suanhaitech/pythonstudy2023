import numpy as np
import matplotlib.pyplot as plt
from fealpy.pde.elliptic_1d import SinPDEData 
pde = SinPDEData()
domain = pde.domain()
# 画出真解的图像
x = np.linspace(domain[0], domain[1], 100)
u = pde.solution(x)
fig = plt.figure()
axes = fig.gca()
axes.plot(x, u)
plt.show()
