print("欢迎来到黑马儿童游乐场，儿童免费，成人收费")
age = input("请输入你的年龄：")
age = int(age)
if age >= 18:
    print("您已成年，补票需要10元")
    print("祝您游玩愉快")
if age < 18:
    print("您未成年，游玩免费")
    print("祝您游玩愉快")
