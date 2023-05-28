#整数数组索引
import numpy as np
x = np.array([[1,2],[3,4],[5,6]])
y = x[[0,1,2],[0,1,0]]#就是取(0,0),(1,1),(2,0)这三个点
print(y)

print(x)


x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
print(x)
row = np.array([[0,0],[3,3]])
cols = np.array([[0,2],[0,2]])
print(row)
print(cols)

y = x[row,cols]#取(0,0),(0,2),(3,0),(3,2)三个位置的元素
print(y)

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = a[1:3,1:3]
c = a[1:3,[1,2]]
d = a[...,1:]
print(a)
print(b)
print(c)
print(d)
