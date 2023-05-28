"""
公司账户余额 1W，给 20名员工发工资。
-员工编号1-20依次领取工资，每日可领取1000元.
-领工资时财务判断绩效1-10（随机），低于5，不发工资，下一位、
-发完工资后结束发工资。
"""
money = 10000
for i in range(1, 21):
    if money == 0:
        break
    else:
        import random
        num = random.randint(1, 10)
        if num >= 5:
            money = money - 1000
            print(f"员工{i}，绩效分{num}，发放工资1000，账户余额还有{money}")
        else:
            print(f"员工{i}，绩效分{num}，低于5，不发工资")
if money == 0:
    print("工资发完了，下个月领取吧")
else:
    print(f"工资尚有余额{money},希望绩效低于5的员工再接再厉")

