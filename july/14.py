import matplotlib.pyplot as plt
import numpy as np             # 导入所需要的模板
x = np.arange(0,4*np.pi,0.1)   # start,stop,step
y = np.sin(x)
z = np.cos(x)
plt.plot(x,y,x,z)
plt.show()                  # 做出图形

