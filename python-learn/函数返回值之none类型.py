# if 判断中，None（字面量）等同于false
def check_age(age):
    if age > 18:
        return "success"
    else:
        return None
    # 可不写
result = check_age(16)
# if语句true即运行
if not result:
    # 进入if表明result是None值，也就是False
    print("未成年，不可以进入")
# None用于声明无初始内容的变量
name = None
