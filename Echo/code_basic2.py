###################################字典

data = {'语文':105, '数学':140, '英语':120}   #创建字典
print(data['语文'])
data = {'语文':105, '数学':140, '英语':120}   #创建字典
print(data['语文'])                             #访问字典的值
data['物理']=99                                 #添加键值对
del data['语文']                                #删除键值对
data['数学']=112                                #修改字典值
print('数学' in data)                           #判断键值对是否存在
data.clear()                                   #删除字典所有内容
print(data.get('历史'))                        #根据键来获取值（键不存在时，返回None，不报错）
data.update({'语文':130,'地理':90})             #更新已有的字典（已含的key-value对，覆盖；不含的key-value对，添加）
print(data.items())                           #以列表返回可遍历的(键, 值) 元组数组
print(data.keys())                            #以列表返回一个字典所有的键
print(data.values())                          #以列表返回字典中的所有值
data.pop('语文')#获取指定 key 对应的 value，并删除这个 key-value 对
print(data.popitem())                         #弹出字典中最后一个key-value对
data.setdefault('语文',100)                    #设置默认值（key 在字典中不存在时）
print(len(data))                              #计算字典元素个数（键总数）
print(type(data))                             #返回输入的变量类型


##############################数组
list1=[1,2,3]
tuple1 = ([1,2,3],[4,5,6],[7,8,9])
print(list1)
print(tuple1)

#############################元组
tuple1 = tuple((1,2,3))        
print(tuple1)
tuple2 = (1,2,3,4,5,6,7,8,9,10)
print(tuple2)
print(tuple2[5])
num = eval(input("请输入要查找元素的下标："))
print(tuple2[num])

