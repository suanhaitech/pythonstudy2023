import matplotlib.pyplot as plt
import numpy as np

#plot 1
x=np.arange(-8,8,0.1)
y=x**3
plt.subplot(2,2,1)
plt.plot(x,y)
plt.title("plot 1")

#plot 2
x=np.linspace(0,3*np.pi,400)
y=x/(1+(x**4)*(np.sin(x))**2)
plt.subplot(2,2,2)
plt.plot(x,y)
plt.title("plot 2")

#plot 3
x=np.linspace(1,10,400)
y=np.sin(1/(x**(1/2)))
plt.subplot(2,2,3)
plt.plot(x,y)
plt.title("plot 3")

#plot 4
x=np.linspace(0,2*np.pi,400)
y=x**(np.sin(x))
plt.subplot(2,2,4)
plt.plot(x,y)
plt.title("plot 4")

plt.show()
