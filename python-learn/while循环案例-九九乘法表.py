"""
print("Hello", end='')
print("world", end='')
补充：print语句输出不换行
print("Hello world")
print("it best")
print("Hello\tworld")
print("itself\tbest")
补充：制表符\t——效果等同于按下tab键，可让多行字符串进行对齐，差距大可能对不齐
"""
i = 1
while i <= 9:
    j = 1
    while i >= j:
        print(f"{i}*{j}={i*j}\t", end='')
        j += 1
    i += 1
    print()  # print空内容，就是输出一个换行
