import matplotlib                #通过 import 来导入 matplotlib 库
#print(matplotlib.__version__)
import matplotlib.pyplot as plt
import numpy as np



xpoints = np.array([0, 6])
ypoints = np.array([0, 100])
plt.plot(xpoints, ypoints, 'o')  
plt.show()

x = np.arange(0,4*np.pi,0.1)   
y = np.sin(x)
z = np.cos(x)
plt.plot(x,y,x,z)              
plt.show()



import matplotlib.markers
ypoints = np.array([1,3,4,5,8,9,6,1,3,4,5,2,4])
plt.plot(ypoints, marker = 'o')
plt.plot([1, 2, 3], marker=matplotlib.markers.CARETDOWNBASE)
plt.show()



ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, marker = 'o', ms = 20, mec = '#4CAF50', mfc = '#4CAF50')
plt.show()



ypoints = np.array([6,2,13,10])
plt.plot(ypoints,ls ='-.',c='SeaGreen',linewidth='12.5')
plt.show()

y1 = np.array([3, 7, 5, 9])
y2 = np.array([6, 2, 13, 10])
plt.plot(y1)
plt.plot(y2)                       
plt.show()



x = np.array([1,2,3,4])
y = np.array([1,4,9,16])
plt.plot(x, y)
plt.title("RUNOOB TEST TITLE")
plt.xlabel("x - label")
plt.ylabel("y - label")
plt.show()



import matplotlib
zhfont1 = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Bold.otf", size=18) 
x = np.arange(1,11)
y =  2  * x +  5
plt.title("测试", fontproperties=zhfont1)
plt.xlabel("x 轴", fontproperties=zhfont1)
plt.ylabel("y 轴", fontproperties=zhfont1)
plt.plot(x,y)
plt.show()



x = np.array([1,2,3,4])
y = np.array([1,4,9,16])
plt.title("RUNOOB grid() Test")
plt.xlabel("x - label")
plt.ylabel("y - label")
plt.plot(x, y)
plt.grid(color = 'r', linestyle = '--', linewidth = 0.5)
plt.show()



#图 1:
zhfont1 = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Bold.otf", size=18) 
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])
plt.subplot(1, 2, 1)     
plt.plot(xpoints,ypoints)
plt.title("图 1",fontproperties=zhfont1)

#图 2:
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.subplot(1, 2, 2)    
plt.plot(x,y)
plt.title("图 2",fontproperties=zhfont1)
plt.suptitle("RUNOOB subplot Test")
plt.show()



x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# 创建一个画像和子图 -- 图2
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# 创建两个子图 -- 图3
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# 创建四个子图 -- 图4
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)# 共享 x 轴
plt.subplots(2, 2, sharex='col')# 共享 y 轴
plt.subplots(2, 2, sharey='row')# 共享 x 轴和 y 轴
plt.subplots(2, 2, sharex='all', sharey='all')# 这个也是共享 x 轴和 y 轴
plt.subplots(2, 2, sharex=True, sharey=True)# 创建标识为10的图，已经存在的则删除
fig, ax = plt.subplots(num=10, clear=True)
plt.show()



x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9, 16, 7, 11, 23, 18])
sizes = np.array([20,50,100,200,500,1000,60,90])#设置图标大小
colors = np.array(["red","green","black","orange","purple","beige","cyan","magenta"])
plt.scatter(x,y,s=sizes,c=colors)
plt.colorbar()
plt.show()



x = np.array(["Runoob-1", "Runoob-2", "Runoob-3", "C-RUNOOB"])
y = np.array([12, 22, 6, 18])
plt.bar(x,y,color = ["#4CAF50","red","hotpink","#556B2F"],width = 0.1)
plt.show()



y = np.array([35, 25, 25, 15])
plt.pie(y,
        labels=['A','B','C','D'], 
        colors=["#d5695d", "#5d8ca8", "#65a479", "#a564c9"], 
        explode=(0, 0.2, 0, 0), 
        autopct='%.2f%%', 
       )
plt.title("RUNOOB Pie Test") 
plt.show()



data = np.random.randn(1000)
plt.hist(data, bins=30, alpha=0.5,color='skyblue')
plt.title('RUNOOB hist() Test')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()



# 生成三组随机数据
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1, 1000)
data3 = np.random.normal(-2, 1, 1000)
# 绘制直方图
plt.hist(data1, bins=30, alpha=0.5, label='Data 1')
plt.hist(data2, bins=30, alpha=0.5, label='Data 2')
plt.hist(data3, bins=30, alpha=0.5, label='Data 3')
# 设置图表属性
plt.title('RUNOOB hist() TEST')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
# 显示图表
plt.show()


"""
import pandas as pd
"""
# 使用 NumPy 生成随机数
random_data = np.random.normal(170, 10, 250)
# 将数据转换为 Pandas DataFrame
dataframe = pd.DataFrame(random_data)
# 使用 Pandas hist() 方法绘制直方图
dataframe.hist()
plt.title('RUNOOB hist() Test')
plt.xlabel('X-Value')
plt.ylabel('Y-Value')
plt.show()



data = pd.Series(np.random.normal(size=100))
plt.hist(data, bins=10)
plt.title('RUNOOB hist() Tes')
plt.xlabel('X-Values')
plt.ylabel('Y-Values')
plt.show()



# 生成一个随机的彩色图像
img = np.random.rand(10, 10, 3)
plt.imshow(img)
plt.show()



from PIL import Image
img = Image.open('map.jpeg')
# 转换为数组
data = np.array(img)
# 绘制地图
plt.imshow(data)
# 隐藏坐标轴
plt.axis('off')
plt.show()



# 生成一个随机矩阵
data = np.random.rand(10, 10)
plt.imshow(data)
plt.show()



n = 4
# 创建一个 n x n 的二维numpy数组
a = np.reshape(np.linspace(0,1,n**2), (n,n))
plt.figure(figsize=(12,4.5))

# 第一张图展示灰度的色彩映射方式，并且没有进行颜色的混合
plt.subplot(131)
plt.imshow(a, cmap='gray', interpolation='nearest')
plt.xticks(range(n))
plt.yticks(range(n))
# 灰度映射，无混合
plt.title('Gray color map, no blending', y=1.02, fontsize=12)

# 第二张图展示使用viridis颜色映射的图像，同样没有进行颜色的混合
plt.subplot(132)
plt.imshow(a, cmap='viridis', interpolation='nearest')
plt.yticks([])
plt.xticks(range(n))
# Viridis映射，无混合
plt.title('Viridis color map, no blending', y=1.02, fontsize=12)

# 第三张图展示使用viridis颜色映射的图像，并且使用了双立方插值方法进行颜色混合
plt.subplot(133)
plt.imshow(a, cmap='viridis', interpolation='bicubic')
plt.yticks([])
plt.xticks(range(n))
# Viridis 映射，双立方混合
plt.title('Viridis color map, bicubic blending', y=1.02, fontsize=12)
plt.show()



img_data = np.random.random((100, 100))
plt.imshow(img_data)
# 保存图像到磁盘上
plt.imsave('runoob-test.png', img_data)



# 创建一幅灰度图像
img_gray = np.random.random((100, 100))
# 创建一幅彩色图像
img_color = np.zeros((100, 100, 3))
img_color[:, :, 0] = np.random.random((100, 100))
img_color[:, :, 1] = np.random.random((100, 100))
img_color[:, :, 2] = np.random.random((100, 100))

plt.imshow(img_gray, cmap='gray')
# 保存灰度图像到磁盘上
plt.imsave('test_gray.png', img_gray, cmap='gray')

plt.imshow(img_color)
# 保存彩色图像到磁盘上
plt.imsave('test_color.jpg', img_color)



img = plt.imread('map.jpeg')
plt.imshow(img)
plt.show()



img_array = plt.imread('tiger.jpeg')
tiger = img_array/255

plt.figure(figsize=(10,6))
for i in range(1,5):
    plt.subplot(2,2,i)
    x = 1 - 0.2*(i-1)
    plt.axis('off') #hide coordinate axes
    plt.title('x={:.1f}'.format(x))
    plt.imshow(tiger*x)
plt.show()



#截图
img_array = plt.imread('tiger.jpeg')
tiger = img_array/255

plt.figure(figsize=(6,6))
plt.imshow(tiger[:300,100:400,:])
plt.axis('off')
plt.show()



img_array = plt.imread('tiger.jpeg')
tiger = img_array/255

red_tiger = tiger.copy()
red_tiger[:, :,[1,2]] = 0

plt.figure(figsize=(10,10))
plt.imshow(red_tiger)
plt.axis('off')
plt.show()




