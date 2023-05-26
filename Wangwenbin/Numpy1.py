import numpy as np
x=np.arange(12).reshape(4,3)    #以arange函数从0到11创建数组赋值给x，并重塑为4*3的形状
print(type(x))                  #输出x的数据结构类型
print(x.dtype)                  #输出x的数据类型
print(x.shape)                  #输出数组的维度（4,3）
print(x.strides)                #输出跨越数组维度时需要越过的字节数
print(x.ndim)                   #输出维数或秩
print(x.itemsize)               #以字节为单位输出每个元素的大小
