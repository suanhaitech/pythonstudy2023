import matplotlib.pyplot as plt
import numpy as np

x=np.random.rand(20)
y=np.random.rand(20)
plt.subplot(1,2,1)
plt.scatter(x,y)
plt.title("plot 1")

x=np.random.rand(40)
y=np.random.rand(40)
colors=np.random.rand(40)
size=(30*np.random.rand(40))**2
plt.subplot(1,2,2)
plt.scatter(x,y,size,c=colors,alpha=0.5)
plt.title("plot 2")

plt.colorbar()
plt.show()
