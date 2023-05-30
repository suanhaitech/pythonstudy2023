import numpy as np
x = np.arange(12).reshape(4,3)
print(x)
print(np.sum(x,axis=0))#列求和
print(np.sum(x,axis=1))#行求和
print(np.sum(x,axis=(0,1)))#求和
