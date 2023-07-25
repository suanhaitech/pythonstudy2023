import numpy as np
F = np.zeros(5)
I = np.array([0, 3, 4,4]) # 索引数组
val = np.array([1.0, 2.0, 3.0,3.0])   # 索引对应的值
np.add.at(F, I, val)    #在F上的I的位置加val
print(F)
