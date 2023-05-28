# 列表容量上限2**63-1 == 9223372036854775807个
name_list = ['itheima', 'itcast', 'python']
print(name_list)
print(type(name_list))
my_list = ['itheima', 666, True]
print(my_list)
print(type(my_list))
# 列表可以一次存储多个数据，且可以为不同的数据类型，支持嵌套
m_list = [[1, 2, 3], [3, 5]]
print(m_list)
print(type(m_list))
# 列表的下表（索引） 012...  语法：列表[下标索引]   或者-n...-1（反向）
print(my_list[2])
print(my_list[-2])
# 嵌套的索引
print(m_list[1][1])  # m_list[1]又是一个列表
