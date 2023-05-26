"""
continue:中断本次循环，直接进入下一次循环，可用于 while和 for循环，效果一致
break:结束本次循环
"""
# for i in range(1, 6):
#     print("a")
#     for j in range(1, 6):
#         print("b")
#         continue
#         print("c")
#     print("d")
for i in range(1, 101):
    print("a")
    break
    print("b")
print("c")

