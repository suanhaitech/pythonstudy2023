import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,2*np.pi,0.1)
y=x*np.sin(x)
z=(x**2)*np.sin(x)
a=(x**(3/2))*np.sin(x)
b=(x**(1/2))*np.sin(x)

plt.plot(x,y,x,z,x,a,x,b)
plt.show()
