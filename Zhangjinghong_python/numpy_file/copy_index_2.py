#布尔索引
import numpy as np
x = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
print(x)
print(x[x>5])

a = np.array([np.nan,1,2,np.nan,3,4,5])#np.nan就是取空值
print(a[~np.isnan(a)])

a = np.array([1,2+6j,5,3.5+5j])
print(a[np.iscomplex(a)])
