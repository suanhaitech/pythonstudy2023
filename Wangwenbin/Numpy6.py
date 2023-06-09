import numpy as np
x=np.arange(36).reshape(6,6)
print("数组x:\n" ,x)
#输出一个6*6的0至35的数组
print("-------------")

y=x[:,1::2]
print(y)
#从第二列开始到最后一列，一隔一切片
print("-------------")

z=x[::3]
print(z)
#从第一行开始逢三做一次切片
print("-------------")

p=x[1:6:2,1:6:2]
print(p)
print("-------------")
#[7,9,11],[19,21,24],[31,33,35]

q=x[[0,3,5],[0,3,5]]
print("x的(0,0),(3,3),(5,5)位置的元素:\n",q)
print("-------------")
#对(0,0)(3,3)(5,5)这三个元素索引

rows=np.array([[0,0],[5,5]])
cols=np.array([[0,5],[0,5]])
a=x[rows,cols]
print("四个角上的元素：")
print(a)
