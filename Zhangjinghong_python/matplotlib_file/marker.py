import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([1,3,4,5,8,9,6,1,3,4,5,2,4])
plt.plot(ypoints, marker = 'o')
plt.show()

ypoints = np.array([6, 2, 13, 10])
#o表示实心圆，：表示虚线，r表示红色
plt.plot(ypoints, 'o:r')
plt.show()

ypoints = np.array([6, 2, 13, 10])
#标记大小为20，内部颜色为黄色，边框为红色
plt.plot(ypoints,marker = 'o',ms = 20,mec = 'r',mfc = 'y')
plt.show()

ypoints = np.array([6, 2, 13, 10])

plt.plot(ypoints, linestyle = 'dotted')
plt.show()

plt.plot(ypoints, color = 'r')
plt.show()

plt.plot(ypoints, linewidth = '12.5')
plt.show()
