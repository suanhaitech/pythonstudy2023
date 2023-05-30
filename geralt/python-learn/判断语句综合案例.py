# 构建一个随机变量
import random
num = random.randint(1,10)
guess_num = int(input("输入你要猜测的数字："))
if guess_num == num:
    print("恭喜你，第一次就猜中了")
else:
    if guess_num > num:
        print("你猜测的数字大了")
    else:
        print("你猜测的数字小了")
    guess_num = int(input("再次输入你要猜测的数字："))
    if guess_num == num:
        print("恭喜你第二次猜中了")
    else:
        if guess_num > num:
            print("你猜测的数字大了")
        else:
            print("你猜测的数字小了")
        guess_num = int(input("请第三次输入你要猜测的数字："))
        if guess_num == num:
            print("恭喜你第三次猜中了")
        else:
            print("三次机会已耗尽，没有猜中")
