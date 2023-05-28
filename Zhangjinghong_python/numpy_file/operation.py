import numpy as np
a = np.array([(0,1),(3,4),(6,7),(9,10)])
b = np.ones((4,2))
print(a)
print(b)
c = a + b#加
print(c)
 
d = a - b#减
print(d)
 
e = a*b#乘，是对应元素相乘
print (e)

f = a/b#除
print(f)

a = np.array([0.25,1.33,1,100])
print(a)
print (np.reciprocal(a))#返回倒数

a = np.array([10,100,1000])
print(a)
print(np.power(a,2))
b = np.array([1,2,3])
print(np.power(a,b))#计算幂

a = np.array([10,20,30])
print(np.mod(a,b))#取余
