"""
多条线，可写一个plot，也可以写两个
"""
import matplotlib.pyplot as plt
import numpy as np
x1 = np.array([0, 1, 2, 3])
y1 = np.array([3, 7, 5, 9])
x2 = np.array([0, 1, 2, 3])
y2 = np.array([6, 2, 13, 10])
plt.plot(x1, y1)
plt.plot(x2, y2) 
plt.show()
