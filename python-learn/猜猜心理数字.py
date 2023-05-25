m_num = int(input("请输入心理数字："))
num = int(input("请输入想猜想的数字："))
if num == m_num:
    print("猜想正确！")
else:
    num = int(input("不对，再猜一次："))
    if num == m_num:
        print("猜想正确！")
    else:
        num = int(input("不对，再猜最后一次："))
        if num == m_num:
            print("猜想正确！")
        else:
            print(f"sorry，全部猜错了，我想的是：{m_num}")

