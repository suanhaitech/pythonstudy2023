import matplotlib.pyplot as plt
import numpy as np

x = np.array(["Runoob-1", "Runoob-2", "Runoob-3", "C-RUNOOB"])
y = np.array([12, 22, 6, 18])
plt.bar(x, y, color = "r")
plt.title("title")
plt.xlabel("xlabel")
plt.ylabel("ylabel")
plt.rcParams['font.family']=['STFangsong']
#plt.barth(x,y)   #将图像旋转
plt.show()
