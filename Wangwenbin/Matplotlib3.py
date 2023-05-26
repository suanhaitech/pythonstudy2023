import matplotlib.pyplot as plt
import numpy as np

y=np.array([1,6,3,5,3,7,8,9])
plt.plot(y,'v:g',ms=15,mec='b')         #倒三角标记，虚线，绿色,标记的小15，外框颜色蓝色
plt.title("A Test")                     #取一个标题
plt.xlabel("x-label")                   #对两个轴命名
plt.ylabel("y-label")
plt.grid(color='y',linewidth=0.5)       #设置网格
plt.show()
