"""
突出显示第二个扇形，并格式化输出百分比
"""
import matplotlib.pyplot as plt
# 数据
sizes = [15, 30, 45, 10]
# 饼图的标签
labels = ['A', 'B', 'C', 'D']
# 饼图的颜色
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
# 突出显示第二个扇形
explode = (0, 0.1, 0, 0) # 值越大，距离中心越远
# 绘制饼图
plt.pie(sizes, explode = explode, labels = labels, colors = colors,
        autopct = '%1.1f%%', # 格式化输出百分比
        shadow = True, # 设置饼图阴影
        startangle = 90 # 用于指定饼图的起始角度，默认为从 x 轴正方向逆时针画起
        )
# 标题
plt.title("Pie Test")
# 显示图形
plt.show()
