if int(input("你的身高是多少：")) > 120:
    print("你的身高超出限制，不可以免费")
    print("但是，如果vip级别大于3，可以免费")
    if int(input("你的vip级别：")) > 3:
        print("恭喜你，可以免费")
    else:
        print("你的vip等级不够，不能免费")
else:
    print("欢迎小朋友免费游玩")
