import matplotlib.pyplot as plt
import numpy as np

x=np.array(["p1","p2","p3","p4","p5"])
y=np.array([14,5,10,6,20])

plt.bar(x,y,color=['r','g','b','y','k'],width=0.8)
plt.show()
