my_list = [21, 25, 21, 23, 22, 20]
my_list.append(31)
print(my_list)
new_list = [29, 33, 30]
my_list.extend(new_list)
print(my_list)
# 删去并取出第一个元素
a = my_list.pop(0)
# 删去并取出最后一个元素
z = my_list.pop(-1)
print(a, z)
# 查找31位置
index = my_list.index(31)
print(index)
