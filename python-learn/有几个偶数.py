num = int(input("请输入数字："))
i = 0
for x in range(1, num + 1):
    if x % 2 == 0:
        i += 1
print(f"1到{num}中有{i}个偶数")
