my_list = [1, 2, 3, 4, 5, 6, 8, 9, 0, 12]
def list_while_func(m):
    """
    用while循环取出列表内的偶数
    :param m:     m表示列表
    :return: None
    """
    index = 0
    while index < len(m):
        element = m[index]
        index += 1
        if element % 2 == 0:
            print(f"列表中的偶数为{element}")
        else:
            continue
list_while_func(my_list)


def list_for_func(m):
    """
    用for循环取出列表内的偶数
    :param m: 列表
    :return: None
    """
    for i in m:
        if i % 2 == 0:
            print(f"列表中的偶数为{i}")
list_for_func(my_list)