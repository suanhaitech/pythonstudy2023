import numpy as np
x = np.arange(12).reshape(4, 3)

print(type(x)) #<class 'numpy.ndarray'>

print(x.dtype)
print(x.shape)   # (4, 3)
print(x.strides) # (24, 8)
