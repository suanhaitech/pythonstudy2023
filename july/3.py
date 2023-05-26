# 猜字谜游戏
number = 7
guess = 3
while guess != number:
    guess = int(input("请输入你猜的数字："))
    
    if guess == number:
        print("恭喜你，猜对了")
    elif guess < number:
        print("猜的数字小了")
    elif guess > number:
        print("猜的数字大了")

