import numpy as np
x = np.arange(12).reshape(4,3)
y = x[:,1:]#取所有行，从第一列开始取
z = x[:,::2]#取所有行，步长为2取列
print(x)
print(y)
print(z)

a = np.arange(10)
# 从索引 2 开始到索引 7 停止，步长为 2
s = slice(2,7,2)

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a[1:])#输出索引为2及以后的行

print (a[...,1])   # 第1列元素
print (a[1,...])   # 第1行元素
print (a[...,1:])  # 第1列及剩下的所有元素
