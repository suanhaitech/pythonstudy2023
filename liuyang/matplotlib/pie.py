import matplotlib.pyplot as plt
# 数据
sizes = [15, 30, 45, 10]
# 饼图的标签
labels = ['A', 'B', 'C', 'D']
# 饼图的颜色
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
# 突出显示第二个扇形，数值越大突出越远
explode = (0, 0.1, 0, 0)
# 绘制饼图
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# 标题
plt.title("RUNOOB Pie Test")
# 显示图形
plt.show()
