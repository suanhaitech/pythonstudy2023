import numpy as np
x=np.arange(25).reshape(5,5)
print(x)
print('-------------')

y=x[[0,1,2,3,4],[0,1,2,3,4]]
print(y)
print('-------------')
#索引对角线元素
z=np.sum(y,axis=0)
print(z)
#对角线元素求和
