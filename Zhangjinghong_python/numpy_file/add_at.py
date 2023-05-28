import numpy as np
F = np.zeros(5)
I = np.array([0,3,4,4]) # 索引数组
val = np.array([1.0,2.0,3.0,3.0]) # 索引对应的值
np.add.at(F, I, val)#在索引为0的地方加1，索引为3的地方加2，索引为4的地方加两次3
print(F)

#二维数组
x=([0,4,1],[3,2,4])
y=np.zeros((5,6),int)
np.add.at(y,x,1)
#分别在位置为第0行第3列；第4行第2列；第1行第4列的位置+1
print(y)
