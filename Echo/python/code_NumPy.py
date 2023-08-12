import numpy as np                    #引入库

data1 = [6, 7.5, 8, 0, -1 ]
arr1 = np.array(data1)
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)


a = np.array([1, 2, 3, 4])
b = np.array([1, 2, 3, 4])
print (a + b)  
print (a - b)  
print (a * b)  
print (a*3)    
print (a / b)  
print (a ** b) 


c= np.array([True, True, False, False])
d= np.array([True, False, True, False])
print (c & d) 
print (c | d) 
print (~c)  
print (c & True)  
print (d & False)  


e = 2
a = np.array([1, 2, 3, 4])
print (a != e)
print (a > e)   
print (a >= e)  
print (a < e)
print (a <= e)
print (a == e)      


f= np.array([1, 2, 3, 4])
g=np.array([True,True,False,False])
print(f[g])    
print(f[f<3])      
print(f[g==True])  


h= np.array([1, 2, 3, 4])
i = h
h+=h      
print( i )  # [2 4 6 8]
 
h= np.array([1, 2, 3, 4])
i = h
h=h+np.array([1, 2, 3, 4])  
print( i )                  


a = np.arange(10)
s = slice(2,7,2)   
print (a[s])
b = a[2:7:2]    
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print (a[...,1])
print (a[1,...])   
print (a[...,1:])


x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
print(x[[0,1,2],  [0,1,0]] )
print (x[x > 5])

a = np.array([np.nan,  1,2,np.nan,3,4,5])     
print (a[~np.isnan(a)])

a = np.array([1,  2+6j,  5,  3.5+5j])
print (a[np.iscomplex(a)])

x = np.arange(9)                
print(x[[0, 6]] )

x=np.arange(32).reshape((8,4))                 
print (x[[4,2,1,7]])


arr1 = np.array([[0, 0, 0],[1, 1, 1],[2, 2, 2], [3, 3, 3]]) 
arr2 = np.array([1, 2, 3])                                  
arr_sum = arr1 + arr2                                          
print(arr_sum)


x = np.arange(12).reshape(4, 3)
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))
print(np.sum(x, axis=(0, 1)))


arr = np.arange(10)
sum1 = np.einsum('i->', arr)          
print("\nsum1: {}".format(sum1))    


arr2 = np.arange(20).reshape(4, 5)
sum_col = np.einsum('ij->j', arr2)          
print("\nsum_col:\n{}".format(sum_col)) 
sum_row = np.einsum("ab->a", arr2)
print("\nsum_row:\n{}".format(sum_row))     
sum_all= np.einsum('ij->', arr2)            
print("\nsum_all:\n{}".format(sum_all)) 


A = np.array([[1, 1, 1], [2, 2, 2], [5, 5, 5]])
B = np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]])
result = A @ B
print("\nresult:\n{}".format(result))        #矩阵乘法
result2 = np.einsum('ij, jk->ik', A, B)
print("\nresult2:\n{}".format(result2))
result3 = np.einsum('ij,ij->', A, B)         #对应矩阵元素相乘求和
print("\nresult3:\n{}".format(result3))


a = np.array([[1, 2], [3, 4]])
b = np.ones(shape=(2, 2))
print("\narray:\n{}".format(np.einsum('ij,ij->ij', a, b)))
print(np.einsum('ij,ij->', a, b))


AA = np.array([[11, 12, 13, 14],
               [21, 22, 23, 24],
               [31, 32, 33, 34],
               [41, 42, 43, 44]])
print("\narray:\n{}".format(np.einsum('ii->i', AA)))
print(np.einsum('ii->', AA))
print(np.einsum('ij->ji', AA))


a=np.array([1,2,3,4])
np.add.at(a,[0,1,2,2],1)  
print(a)                  

a=np.array([1,2,3,4])
b=np.array([1,2])
np.add.at(a,[0,1],b)    
print(a)                 

x=([0,4,1],[3,2,4])
dW=np.zeros((5,6),int)
np.add.at(dW,x,1)
dW

