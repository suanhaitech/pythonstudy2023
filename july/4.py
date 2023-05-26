# 设计一个类
class Student:
    def __init__(self, name, age, tel):
        self.name = name
        self.age = age
        self.tel = tel
        print(f"我是{self.name}，我的年龄为{self.age}，我的电话号码为{self.tel}")
student_1 = Student("July", 23, "123654897")
print(student_1.name)
print(student_1.age)
print(student_1.tel)

