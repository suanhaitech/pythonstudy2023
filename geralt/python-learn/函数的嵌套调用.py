"""
演示嵌套调用函数
"""
def fun_b():
    print("a")
def fun_a():
    print("b")
    # 嵌套调用fun_b
    fun_b()
    print("c")
# 调用fun_a
fun_a()
