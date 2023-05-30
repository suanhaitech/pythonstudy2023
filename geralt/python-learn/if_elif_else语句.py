height = int(input("请输入你的身高(cm):"))
vip_level = int(input("请输入你的vip等级（1-5）："))
day = int(input("请告诉我今天几号："))
# if int(input("请输入你的身高（cm）：")) <= 120:
if height <= 120:
    print("身高小于120cm，可以免费")
elif vip_level >= 3:
    print("vip等级大于等于3，可以免费")
elif day == 1:
    print("今天是1号，可以免费")
else:
    print("不好意思，需要买票十元")

