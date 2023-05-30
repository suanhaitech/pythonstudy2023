"""
演示python的input语句
获取键盘输入信息
"""
print("请告诉我你是谁？")
name = input()
# name = input("请告诉我你是谁？")  提示语句可直接写入
print("我知道了，你是：%s" % name)
# 输入数字类型
num = input("请告诉我你的银行卡密码：")
# 接受默认为字符串
print("你的银行卡密码的类型是：", type(num))
