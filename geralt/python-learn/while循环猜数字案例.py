"""
设置1-100随机数，通过while循环，配合input语句，判断是否相等
-无限次
-提示大小
-猜了几次
"""
import random
num = random.randint(1, 100)
guess_num = int(input("请输入你要猜测的数字（1-100）："))
i = 1
while num != guess_num:
    if guess_num > num:
        print("大了")
    else:
        print("小了")
    guess_num = int(input("请再次输入你要猜测的数字（1-100）："))
    i += 1
print(f"恭喜你，您猜中了，一共猜了：{i}次，数字为{num}")

