import matplotlib.pyplot as plt
sizes=[10,20,30,15,25]
#大小
labels=['A','B','C','D','E']
#每个部分的标签
colors=['#65a479','g','lightskyblue','gold','#a564c9']
#颜色
explode=(0.1,0.1,0,0,0)
#突出第一第二个扇形
plt.pie(sizes,explode=explode,labels=labels,colors=colors,
        autopct='%1.1f%%',shadow=True,startangle=90)
#生成pie
plt.title("Test")
plt.show()
