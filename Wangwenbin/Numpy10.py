import numpy as np

a=np.array([0,120,145,55])
print('四个角度\n',a)

print('不同角度的正弦值：')
#利用pi/180转化为弧度
print(np.sin(a*np.pi/180))
print('-----------------')

print('cos()')
print(np.cos(a*np.pi/180))
print('-----------------')

print('tan()')
print(np.tan(a*np.pi/180))

