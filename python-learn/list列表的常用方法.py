my_list = ['itself', 'heima', 'python']
# 列表的查询功能（方法） 语法：列表.index(元素)    若不存在，会报错
index = my_list.index("itself")
print(index)
# 修改特定位置（索引）的元素值  语法：列表[下表]=值
my_list[0] = "ding"
print(my_list, type(my_list[0]))
# 插入元素  语法：列表.insert(下标， 元素)  在指定下标位置插入指定元素
my_list.insert(1, 'best')
print(my_list)
# 追加元素  语法：列表.append(元素)，将指定元素，追加到列表的尾部
my_list.append('quan')
print(my_list)
# 追加一批元素  语法：列表.extend(其他数据容器)，将其他数据容器的内容取出，依次追加到列表尾部
mylist2 = [1, 2, 3]
my_list.extend(mylist2)
print(my_list)
# 删除元素  语法1：del 列表[下标]  语法2：列表.pop(下标)
del my_list[2]  # del关键字
print(my_list)
element = my_list.pop(1)  # 返回删除值
print(my_list, element)
# 删除某元素在列表的第一个匹配项  语法：列表.remove(元素)
my_list = ['itself', 'heima', 'python', 'heima']
my_list.remove("heima")
print(my_list)
# 清空列表
my_list = ['itself', 'heima', 'python', 'heima']
my_list.clear()
print(my_list)
# 统计某元素在列表中的数量
my_list = [1, 2, 3, 45, 2, 1]
count = my_list.count(1)  # 注意数据类型
print(count)
# 统计列表中的全部元素数量
count = len(my_list)
print(count)