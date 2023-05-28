# 演示局部变量
# def test_a():
#     num = 100
#     print(num)
#
# test_a()
# 出了函数其局部变量无法使用
# print(num)
# 定义全局变量
# num = 200
# def test_a():
#     print(f"test_a:{num}")
# def test_b():
#     num = 500
#     print(f"test_b:{num}")
# test_a()
# test_b()
# print(num)
# global关键字
num = 200
def test_a():
    print(f"test_a:{num}")
def test_b():
    global num  # 设置为全局变量
    num = 500
    print(f"test_b:{num}")
test_a()
test_b()
print(num)


