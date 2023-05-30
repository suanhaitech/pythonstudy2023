"""
黑马ATM
定义全局变量money（记录余额）、name（记录客户姓名）
定义函数：
-查询余额
-存款函数
-取款函数
-主菜单函数
request：
-程序启动后输入客户姓名
-查询余额、取款、存款后返回主菜单
-存款、取款后，显示当前余额
-客户选择退出或输入错误，程序会退出，否则会一直运行
"""
money = 5000000
name = None
name = input("请输入您的姓名：")
def menu():
    print("----------主菜单----------")
    global name
    print(f"{name}，您好，欢迎来到黑马银行ATM，请选择操作：")
    print("查询余额\t[输入1]")
    print("存款  \t[输入2]")
    print("取款  \t[输入3]")
    print("退出  \t[输入4]")
    m = int(input("请输入您的选择："))
    if m == 1:
        check_money()
    elif m == 2:
        deposit_money()
    elif m == 3:
        withdrawal_money()
    else:
        print("程序已退出")
        name = None
def check_money():
    print("----------查询余额----------")
    print(f"{name}，您好，您的余额剩余：{money}")
    menu()
def deposit_money():
    print("----------存款----------")
    global money
    d_money = int(input("请输入您要存款的金额："))
    money += d_money
    print(f"{name}，您好，您存款{d_money}元成功")
    print(f"{name}，您好，您的余额剩余{money}元")
    menu()
def withdrawal_money():
    print("----------取款----------")
    global money
    w_money = int(input("请输入您要取款的金额："))
    test_money = money - w_money
    if test_money >= 0:
        money = test_money
        print(f"{name}，您好，您取款{w_money}成功")
        print(f"{name}，您好，您的余额剩余{money}")
        menu()
    else:
        print(f"对不起，您的余额不足，当前余额为{money}元")
        menu()

menu()




