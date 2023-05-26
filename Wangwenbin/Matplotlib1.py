import matplotlib.pyplot as plt
import numpy as np
x=np.arange(-10,10,0.01)
y=1/(np.sin(x)+2)
z=1/(np.cos(x)+2)

plt.plot(x,y,x,z)                             #生成在一张图像上

fig2,(axs1,axs2)=plt.subplots(2,1)            #分配两个坐标轴并且按照(2,1)的形状
axs1.plot(x,y)
axs2.plot(x,z)                                #在两个轴上单独生成一次
plt.show()
