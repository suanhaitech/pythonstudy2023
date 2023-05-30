##########################################定义
#查看python关键词（变量定义不可用）

import keyword
print(keyword.kwlist)


###########################################列表
#输入一年中的某一天，判断这一天是这一年的第几天：【输入格式：YYYY-MM-DD】
User_input = input('输入：年-月-日')
Year = int(User_input.split('-')[0])   ##得到年份
Month = int(User_input.split('-')[1])  ##得到月份
Day = int(User_input.split('-')[2])    ##得到天

li = [31,28,31,30,31,30,31,31,30,31,30,31]   ##所有平年各个月份的天数
num = 0    ##记录天数
if ((Year % 4 == 0) and (Year % 100 != 0) or (Year % 400 == 0)):    ##当闰年时：
    li[1] = 29   ##将二月的天数改为29
for i in range(12):  ##遍历月份
	if Month > i + 1:   ##i从0开始，假如是5月的某一天，i循环到3停止，经过0-1-2-3四次循环，取4个月份即取1-2-3-4月的所有天
		num += li[i]   ##将1-4月总天数求和
	else:            ##退出if判断后，当下一次循环时，i=4，i+1不满足if的条件，进入else，将最后5月的第几天加入总天数中
		num += Day
		break
print('这一天是%d年的第%d天' %(Year,num))


#########################################
#修改用户登陆系统：用户名和用户密码存放在两个列表里。用admin超级用户登陆后，可以进行添加，删除，查看用户的操作。1.后台管理员admin 密码admin2.管理员才能看到会员信息3.会员信息包含（添加会员信息，删除会员信息，查看会员信息，退出）
inuser = input('UserName: ')
inpasswd = input('Password: ')
users = ['root', 'westos']
passwds = ['123', '456']

if inuser == 'admin' and inpasswd == 'admin':
    while True:
        print("""
            菜单
        1.添加会员信息
        2.删除会员信息
        3.查看会员信息
        4.退出
        """)
        choice = input('请输入选择： ')
        if choice == '1':
            Add_Name = input('要添加的会员名: ')
            Add_Passwd = input('设置会员的密码为： ')
            users = users + [Add_Name]
            passwds = passwds + [Add_Passwd]
            print('添加成功！')

        elif choice == '2':
            Remove_Name = input('请输入要删除的会员名： ')
            if Remove_Name in users:
                Remove_Passwd = input('请输入该会员的密码： ')
                SuoYinZhi = int(users.index(Remove_Name))
                if Remove_Passwd == passwds[SuoYinZhi]:
                    users.remove(Remove_Name)
                    passwds.pop(SuoYinZhi)
                    print('成功删除！')
                else:
                    print('用户密码错误,无法验证身份,删除失败')
            else:
                print('用户错误！请输入正确的用户名')


        elif choice == '3':
            print('查看会员信息'.center(50,'*'))
            print('\t用户名\t密码')
            usercount = len(users)
            for i in range(usercount):
                print('\t%s\t%s' %(users[i],passwds[i]))


        elif choice == '4':
            exit()
        else:
            print('请输入正确选择！')

