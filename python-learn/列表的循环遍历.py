def list_while_func():
    """
    使用while循环来遍历列表的演示函数
    :return: None
    """
    my_list = ['粗茶', '粗蛋白', 'AC世界观']
    # 循环控制变量通过下标索引变量控制，默认0
    # 每次+1
    index = 0
    while index < len(my_list):
        element = my_list[index]
        print(f"列表的元素：{element}")
        index += 1
list_while_func()
def list_for_func():
    """
    使用for循环遍历
    :return: None
    """
    my_list = ['粗茶', '粗蛋白', 'AC世界观']
    # for 临时变量 in 数据容器
    for i in my_list:
        print(f"列表的元素有{i}")
list_for_func()