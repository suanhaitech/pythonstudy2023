# 狗狗年龄计算
age = int(input("请输入你家狗狗年龄："))
print("")
if age <= 0:
    print("你是在开玩笑吧！")
elif age == 1:
    print("相当于14岁的人")
elif age == 2:
    print("相当于22岁的人")
elif age >= 2:
    human = 22 + (age -2)*5
    print("对应人的年龄为：", human)

### 退出提示
input("点击enter键退出")











