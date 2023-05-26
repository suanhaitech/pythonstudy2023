"""
表白 100天，每天送十枝玫瑰花，玩点随机...
"""
import random
day = random.randint(1, 100)
i = 1
while i <= day:
    print(f"今天是第{i}天，准备表白......")
    # 内层循环的控制变量
    j = 1
    while j <= 10:
        print(f"送给小明的第{j}枝玫瑰花")
        j += 1
        print("小明，我喜欢你！")
    i += 1
print(f"坚持到第{i-1}天，表白成功")

