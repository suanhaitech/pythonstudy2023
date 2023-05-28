"""
演示python中的各类运算符
"""
# 算术（数学）运算符
print("1 + 1 = ", 1 + 1)
print("2 - 1 = ", 2 - 1)
print("4 * 3 = ", 4 * 3)
print("4 / 2 = ", 4 / 2)
# 取整除
print("11 // 2 = ", 11 // 2)
# 取余
print("9 % 2 = ", 9 % 2)
# 指数
print("3 ** 2 = ", 3 ** 2)
# 赋值运算符
num = 1 + 2 * 3
# 复合赋值运算符
# +=
num1 = 1
num1 += 1  # num1 = num1 + 1
print("num1+=1:", num1)
# -=
num1 -= 1
print("num1-=1:", num1)
num1 *= 4
print("num1*=4:", num1)
num1 /= 2
print("num1/=2:", num1)
num = 3
num %= 2
print("num%=2:", num)
num **= 2
print("num**=2:",num)
num = 9
num //= 2
print("num//=2:", num)


