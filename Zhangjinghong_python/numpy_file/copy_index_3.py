#花式索引
import numpy as np
#一维数组
x = np.arange(9)
print(x)
# 一维数组读取指定下标对应的元素
x2 = x[[0, 6]] # 使用花式索引
print(x2)#输出索引为0和6的元素

print(x2[0])
print(x2[1])

#二维数组
x = np.arange(32).reshape((8,4))
print(x)
print(x[[4,2,1,7]])
print(x[[-4,-2,-1,-7]])#倒序

#传入多个索引数组
x=np.arange(32).reshape((8,4))
print (x[np.ix_([1,5,7,2],[0,3,1,2])])#做笛卡尔积
