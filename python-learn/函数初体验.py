"""
函数，如len()为 python内置的函数
-提前写好的
-可重复使用
-实现统计长度这一特定功能的代码段
"""
# 统计字符串的长度，且不使用内置的函数len()
str1 = "itheima"
str2 = "quanxin"
str3 = "dingrenjie"
def my_len(data):
    count = 0
    for i in data:
        count += 1
    print(f"字符串{data}的长度是{count}")
my_len(str1)
my_len(str2)
my_len(str3)

