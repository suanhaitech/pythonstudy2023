def poiu():
    print("请选择功能-------------")
    print("1,添加学员")
    print("2,删除学员")
    print("3,修改学员")
    print("4,查询学员")
    print("5,显示所有学员")
    print("6,退出系统")
    print("-" * 20)
info = []
def add():
    """添加学员"""
    newid = input("请输入学号：")
    newname = input("请输入姓名：")
    newtel = input("请输入手机号：")
    global info
    for i in info:
        if newname == i["name"]:
            print('此用户已经拥有')
            return
    dictt = {}
    dictt["id"] = newid
    dictt["name"] = newname
    dictt["tel"] = newtel
    info.append(dictt)
    print(info)
def del1():
    """删除学员"""
    del1name = input("请输入删除学员姓名：")
    global info
    for i in info:
        if del1name == i["name"]:
            info.remove(i)
            break
    else:
        print("该学员不存在")
    print(info)
def modify1():
    """修改学员电话"""
    modifyname = input("请输入学员姓名：")
    global info
    for i in info:
        if modifyname == i["name"]:
            i["tel"] = input("请输入新的手机号:")
            break
    else:
        print("该学员不存在")
    print(info)
def seacher():
    """查找学员"""
    seachername = input("请输入学员姓名：")
    global info
    for i in info:
        if seachername == i["name"]:
            print("查找到的学员信息如下-------")
            print(f"学员的学号为{i['id']}, 姓名为{i['name']}, 电话为{i['tel']}")
            break
    else:
        print("该学员不存在")
def all():
    """显示所有学员信息"""
    print('学号\t姓名\t电话')
    for i in info:
        print(f'{i["id"]}\t, {i["name"]}\t, {i["tel"]}\t')
while True:
    poiu()
    user = int(input("请选择功能:"))
    if user == 1:
       add()
    elif user == 2:
       del1()
    elif user == 3:
        modify1()
    elif user == 4:
        seacher()
    elif user == 5:
        all()
    elif user == 6:
        flag = input("确定要退出系统，yes or no :")
        if flag == "yes":
            break
    else:
        print("你输入的数字有误，请重新输入")
