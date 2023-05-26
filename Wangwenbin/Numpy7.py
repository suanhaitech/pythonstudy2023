#广播
import numpy as np
a=np.arange(12).reshape(4,3)
b=np.array([0,1,2])
print(a+b)
print('\n')
c=np.array([1,2,3,4]).reshape(4,1)
print(a+c)
print('\n')
#tile()函数
d=np.array([[1,2],[3,4]])
e=np.tile(d,(3,2))
print(e)
