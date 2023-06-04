import numpy as np
 
a = np.arange(10)
b = np.array([[1,2,3],[3,4,5],[4,5,6]])
print (b[...,1])   # 第2列元素
print (b[1,...])   # 第2行元素
print (b[...,1:])  # 第2列及剩下的所有元素
s = slice(2,7,2)   # 从索引 2 开始到索引 7 停止，间隔为2
print (a[s])
